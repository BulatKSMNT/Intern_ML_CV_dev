"""Осторожная постобработка результатов OCR внутреннего паспорта РФ.

При обработке документов обучающая разметка не используется.
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

FIELDS = (
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
    "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
    "Issue_organization_ru",
)
NAME_FIELDS = FIELDS[:3]
AS_OF = date(2026, 10, 1)
REFERENCE = Path(__file__).resolve().parent / "reference" / "subdivisions.csv"
CYRILLIC = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
MRZ_ALPHABET = "ABVGDE2JZIQKLMNOPRSTUFHC34WXY9678"
ENCODE = str.maketrans(CYRILLIC, MRZ_ALPHABET)
DECODE = str.maketrans(MRZ_ALPHABET, CYRILLIC)


@dataclass(frozen=True)
class Policy:
    """Настройки принятия документов и отключения отдельных проверок в экспериментах.

    Входные данные не изменяются. Способ объединения оценок уверенности применяется
    только к месту рождения и органу выдачи. Произведение оценок — эвристический
    показатель, а не откалиброванная вероятность.
    """

    score: str = "min"
    threshold: float = 0.95
    mobile_threshold: float = 1.0
    organization_mode: str = "off"
    repair_zones: bool = True
    check_geometry: bool = True
    check_checksums: bool = True

    def __post_init__(self):
        if self.score not in {"min", "mean", "product"}:
            raise ValueError("unknown confidence score")
        if self.organization_mode not in {"safe", "off", "legacy"}:
            raise ValueError("unknown organization correction mode")
        if not all(math.isfinite(x) and 0 < x <= 1 for x in (self.threshold, self.mobile_threshold)):
            raise ValueError("thresholds must be in (0, 1]")

    def text_threshold(self, source):
        return self.mobile_threshold if source == "mobile_v2" else self.threshold


def confidence_score(scores, method="min"):
    if not scores:
        return 0.0
    if method == "min":
        return min(scores)
    if method == "mean":
        return sum(scores) / len(scores)
    if method == "product":
        return math.prod(scores)
    raise ValueError("unknown confidence score")


def one_edit_away(observed, candidate):
    """Проверить отличие на одну замену, вставку или удаление символа."""
    if abs(len(observed) - len(candidate)) > 1:
        return False
    if len(observed) == len(candidate):
        return sum(a != b for a, b in zip(observed, candidate)) == 1
    shorter, longer = sorted((observed, candidate), key=len)
    for i, (a, b) in enumerate(zip(shorter, longer)):
        if a != b:
            return shorter[i:] == longer[i + 1:]
    return True  # One extra character at the end.


def norm(value):
    return " ".join(str(value or "").upper().split())


def anniversary(birth, age):
    try:
        return birth.replace(year=birth.year + age)
    except ValueError:
        # Do not consider a leap-day birthday reached before March 1.
        return date(birth.year + age, 3, 1)


def parse_date(value):
    if not re.fullmatch(r"[0-9]{2}\.[0-9]{2}\.[0-9]{4}", value):
        raise ValueError("invalid date format")
    day, month, year = map(int, value.split("."))
    return date(year, month, day)


def valid_dates(birth, issued, as_of=AS_OF):
    """Проверить действительность по доступным датам; статус в базе МВД не проверяется."""
    if not date(1900, 1, 1) <= birth <= as_of:
        return False
    if not max(date(1997, 10, 1), anniversary(birth, 14)) <= issued <= as_of:
        return False
    return all(
        not (issued < anniversary(birth, age)
             and as_of > anniversary(birth, age) + timedelta(days=90))
        for age in (20, 45)
    )


def checksum(text):
    """Вычислить контрольную цифру по ICAO 7–3–1; она не подтверждает подлинность."""
    total = 0
    for i, char in enumerate(text):
        if "0" <= char <= "9":
            value = int(char)
        elif "A" <= char <= "Z":
            value = ord(char) - ord("A") + 10
        elif char == "<":
            value = 0
        else:
            raise ValueError("invalid MRZ character")
        total += value * (7, 3, 1)[i % 3]
    return str(total % 10)


def encode_name(value):
    return value.translate(ENCODE).replace(" ", "<").replace("-", "<")


def name_line(fields):
    body = (encode_name(fields[NAME_FIELDS[0]]) + "<<"
            + encode_name(fields[NAME_FIELDS[1]]) + "<"
            + encode_name(fields[NAME_FIELDS[2]]))
    return ("PNRUS" + body).ljust(44, "<")[:44]


def valid_confidences(line):
    text, scores = line.get("text"), line.get("char_conf")
    return (isinstance(text, str) and isinstance(scores, list)
            and len(text) == len(scores)
            and all(isinstance(c, (int, float)) and math.isfinite(c)
                    and 0 <= c <= 1 for c in scores))


def complete_field(field, check_geometry=True):
    """Выявить пропуски первых, последних или промежуточных строк по рамке поля."""
    if not field or not field.get("lines"):
        return False
    box = field.get("box", [])
    lines = field["lines"]
    if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
        return False
    if any(not valid_confidences(line) or not line["text"].strip()
           or len(line.get("box", [])) != 4
           or line["box"][2] <= 0 or line["box"][3] <= 0 for line in lines):
        return False
    if not check_geometry:
        return True
    lines = sorted(lines, key=lambda line: line["box"][1])
    height = statistics.median(line["box"][3] for line in lines)
    if abs(lines[0]["box"][1] - box[1]) > height * 0.55:
        return False
    bottom = lines[-1]["box"][1] + lines[-1]["box"][3]
    if abs(box[1] + box[3] - bottom) > height * 0.55:
        return False
    if any(not 0.55 * height <= b["box"][1] - a["box"][1] <= 1.6 * height
           for a, b in zip(lines, lines[1:])):
        return False
    return True


def read_field(field):
    return norm(" ".join(line["text"] for line in sorted(
        field["lines"], key=lambda line: line["box"][1]))) if field else ""


def scored_text(field):
    """Нормализовать пробелы, сохранив соответствие символов их оценкам уверенности."""
    if field is None:
        return "", []
    pairs = []
    for line in sorted(field["lines"], key=lambda line: line["box"][1]):
        if pairs:
            pairs.append((" ", 1.0))
        pairs.extend(zip(line["text"].upper(), line["char_conf"]))
    compact = []
    for char, confidence in pairs:
        if char.isspace():
            if compact and compact[-1][0] != " ":
                compact.append((" ", confidence))
        else:
            compact.append((char, confidence))
    if compact and compact[-1][0] == " ":
        compact.pop()
    return "".join(char for char, _ in compact), [score for _, score in compact]


def low_threshold(source):
    return 0.98 if source == "mobile_v2" else 0.90


def corroborated_threshold(source):
    return 1.0 if source == "mobile_v2" else 0.98


def numeric_line(fields):
    """Сформировать нижнюю строку МЧЗ по визуальным полям с четырьмя контрольными цифрами."""
    birth = parse_date(fields["Birth_date"])
    issued = parse_date(fields["Issue_date"])
    if not re.fullmatch(r"[0-9]{2} [0-9]{2} [0-9]{6}", fields["Licence_number"]):
        raise ValueError("invalid passport number")
    if not re.fullmatch(r"[0-9]{3}-[0-9]{3}", fields["Issue_organisation_code"]):
        raise ValueError("invalid subdivision code")
    if fields["Sex_ru"] not in ("МУЖ.", "ЖЕН."):
        raise ValueError("invalid sex")
    number = fields["Licence_number"].replace(" ", "")
    serial = number[:3] + number[4:]
    birth_text = birth.strftime("%y%m%d")
    optional = number[3] + issued.strftime("%y%m%d") + fields["Issue_organisation_code"].replace("-", "") + "<"
    result = (serial + checksum(serial) + "RUS" + birth_text + checksum(birth_text)
              + ("M" if fields["Sex_ru"] == "МУЖ." else "F") + "<<<<<<<"
              + optional + checksum(optional))
    return result + checksum(result[:10] + result[13:20] + result[21:43])


def reconcile_names(first, mrz_scores, fields, viz, source):
    """Согласовать до двух замен символов при явном различии уверенности двух зон.

    Пробелы и дефисы не восстанавливаются по символу '<'. Окончание ФИО,
    отсутствующее из-за усечения МЧЗ, не достраивается.
    """
    expected = name_line(fields)
    if expected == first:
        return fields
    differences = [i for i, (a, b) in enumerate(zip(expected, first)) if a != b]
    if len(differences) > 2:
        return None
    positions = {}
    offset = 5
    for index, key in enumerate(NAME_FIELDS):
        _, scores = scored_text(viz[key])
        for j, score in enumerate(scores):
            positions[offset + j] = key, j, score
        offset += len(fields[key]) + (2 if index == 0 else 1)
    result = dict(fields)
    for i in differences:
        if i not in positions or first[i] not in MRZ_ALPHABET:
            return None
        key, j, visual_score = positions[i]
        if visual_score < low_threshold(source) and mrz_scores[i] >= corroborated_threshold(source):
            value = result[key]
            result[key] = value[:j] + first[i].translate(DECODE) + value[j + 1:]
        elif mrz_scores[i] < low_threshold(source) and visual_score >= corroborated_threshold(source):
            pass  # The visual character is supported by a confident reading.
        else:
            return None
    return result


def minimum_confidence(field):
    return min((c for line in field["lines"] for c in line["char_conf"]), default=0) if field else 0


def read_mrz(mrz, check_checksums=True):
    if not mrz or len(mrz.get("lines", [])) != 2:
        return None
    lines = mrz["lines"]
    if any(not valid_confidences(line) or len(line["text"]) != 44 for line in lines):
        return None
    first, second = (line["text"] for line in lines)
    if not re.fullmatch(r"PNRUS[A-Z0-9<]{39}", first):
        return None
    # Position 28 is '<' for the indefinite expiry date, not a numeric checksum.
    if not re.fullmatch(r"[0-9]{10}RUS[0-9]{7}[MF]<{7}[0-9]{13}<[0-9]{2}", second):
        return None
    if check_checksums and not all((checksum(second[:9]) == second[9],
                checksum(second[13:19]) == second[19],
                checksum(second[28:42]) == second[42],
                checksum(second[:10] + second[13:20] + second[21:43]) == second[43])):
        return None
    return first, second


def numeric_fields(second, visual):
    """Извлечь числовые поля, уточнив век по визуальной дате или однозначности вариантов."""
    issued_raw = second[29:35]
    birth_raw = second[13:19]
    candidates = []
    for century in (1900, 2000):
        try:
            birth = date(century + int(birth_raw[:2]), int(birth_raw[2:4]), int(birth_raw[4:]))
        except ValueError:
            continue
        for issue_century in (1900, 2000):
            try:
                issued = date(issue_century + int(issued_raw[:2]), int(issued_raw[2:4]), int(issued_raw[4:]))
            except ValueError:
                continue
            # Do not eliminate expired candidates here: that could turn an
            # expired 2000-born passport into a valid 1900-born passport.
            if (date(1900, 1, 1) <= birth <= AS_OF
                    and max(date(1997, 10, 1), anniversary(birth, 14)) <= issued <= AS_OF):
                candidates.append((birth, issued))
    # A complete, valid visual date supplies the century that MRZ omits.
    for index, key in enumerate(("Birth_date", "Issue_date")):
        try:
            observed = parse_date(visual[key])
        except ValueError:
            continue
        same = [pair for pair in candidates if pair[index] == observed]
        if same:
            candidates = same
    if len(candidates) != 1:
        return None
    birth, issued = candidates[0]
    if not valid_dates(birth, issued):
        return None
    number = second[:3] + second[28] + second[3:9]
    return {
        "Licence_number": f"{number[:2]} {number[2:4]} {number[4:]}",
        "Birth_date": birth.strftime("%d.%m.%Y"),
        "Issue_date": issued.strftime("%d.%m.%Y"),
        "Sex_ru": "МУЖ." if second[20] == "M" else "ЖЕН.",
        "Issue_organisation_code": second[35:38] + "-" + second[38:41],
    }


class Postprocessor:
    def __init__(self, reference=REFERENCE, policy=None):
        self.policy = policy or Policy()
        self.reference = defaultdict(set)
        with open(reference, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self.reference[row["code"]].add(norm(row["name"]))
        self.words = {}
        administrative_words = set((
            "ОТДЕЛ ОТДЕЛОМ ОТДЕЛЕНИЕ ОТДЕЛЕНИЕМ ТЕРРИТОРИАЛЬНЫЙ ПУНКТ "
            "УПРАВЛЕНИЕ УПРАВЛЕНИЕМ ПО ВОПРОСАМ МИГРАЦИИ ОМВД МВД УФМС ОУФМС "
            "ОФМС РОССИИ РЕСПУБЛИКЕ РЕСПУБЛИКИ РЕСПУБЛИКА ОБЛАСТИ ОБЛАСТЬ "
            "РАЙОНЕ РАЙОНА РАЙОН ГОРОДЕ ГОРОДА ГОРОД"
        ).split())
        for code, names in self.reference.items():
            self.words[code] = administrative_words | set(re.findall(r"[А-ЯЁ]+", " ".join(names)))

    def organization(self, field, code, source):
        """Исправить только однозначные замены букв с низкой уверенностью.

        Сохраняются исходные формулировка и пунктуация: справочник содержит
        исторические названия, поэтому код не определяет единственное
        актуальное название органа выдачи.
        """
        text, scores = scored_text(field)
        threshold = self.policy.text_threshold(source)
        if confidence_score(scores, self.policy.score) >= threshold:
            return text
        if self.policy.organization_mode == "off":
            return None
        if text in self.reference.get(code, ()):
            return text
        words = self.words.get(code, set())
        resolved = set()
        for match in re.finditer(r"[А-ЯЁ]+", text):
            token = match.group()
            # Word-level repair still needs a weak character, regardless of how
            # the full-field score is aggregated.
            if len(token) < 4 or min(scores[match.start():match.end()]) >= low_threshold(source):
                continue
            candidates = []
            for word in words:
                if len(word) != len(token):
                    continue
                diff = [i for i, (a, b) in enumerate(zip(token, word)) if a != b]
                if len(diff) == 1 and scores[match.start() + diff[0]] < low_threshold(source):
                    candidates.append((word, diff[0]))
            if len(candidates) == 1 and token not in words:
                word, i = candidates[0]
                if self.policy.organization_mode == "safe":
                    alternatives = {w for w in words if one_edit_away(token, w)}
                    # A unique substitution is unsafe if deletion/insertion or
                    # another substitution provides an equally close word.
                    if alternatives != {word}:
                        return None
                text = text[:match.start()] + word + text[match.end():]
                resolved.add(match.start() + i)
        if self.policy.organization_mode == "legacy":
            # Reproduce the previous implementation for a measured baseline.
            remaining = [1.0 if i in resolved else c for i, c in enumerate(scores)]
        else:
            # Resolved characters are checked by the dictionary rule. Do not
            # invent a confidence of 1.0 for a correction.
            remaining = [c for i, c in enumerate(scores) if i not in resolved]
        return text if len(resolved) <= 2 and confidence_score(remaining, self.policy.score) >= threshold else None

    def process(self, document):
        """Вернуть ответ и причину решения для анализа отказов при проверке качества."""
        rejected = {"doc_id": document["doc_id"], "decision": "reject"}
        source = document.get("source")
        if source not in {"web", "mobile", "mobile_v2"}:
            return rejected, "unknown_source"
        viz = document.get("viz") or {}
        if not all(key in viz for key in FIELDS):
            return rejected, "missing_schema"
        for key, field in viz.items():
            if field is not None and not complete_field(field, self.policy.check_geometry):
                return rejected, "incomplete_field"
        fields = {key: read_field(viz[key]) for key in FIELDS}
        for key in ("Birth_place_ru", "Issue_organization_ru"):
            if not fields[key]:
                return rejected, "missing_free_text"
            if key == "Birth_place_ru" and confidence_score(scored_text(viz[key])[1], self.policy.score) < self.policy.text_threshold(source):
                return rejected, "uncertain_" + key
        # Keep the repair path identical in the checksum ablation: only relax
        # the final rejection if strict validation and repair both fail.
        mrz = read_mrz(document.get("mrz"))
        if self.policy.repair_zones and mrz is None and document.get("mrz") and len(document["mrz"].get("lines", [])) == 2:
            # Repair a noisy MRZ only against a fully present, confident VIZ.
            lines = document["mrz"]["lines"]
            numeric_keys = FIELDS[3:5] + FIELDS[6:9]
            if (all(valid_confidences(line) and len(line["text"]) == 44 for line in lines)
                    and all(minimum_confidence(viz[key]) >= corroborated_threshold(source) for key in numeric_keys)):
                try:
                    expected = numeric_line(fields)
                except ValueError:
                    expected = ""
                diffs = [i for i, (a, b) in enumerate(zip(expected, lines[1]["text"])) if a != b]
                if expected and 1 <= len(diffs) <= 2 and all(lines[1]["char_conf"][i] < low_threshold(source) for i in diffs):
                    repaired = {"lines": [lines[0], {"text": expected, "char_conf": lines[1]["char_conf"]}]}
                    mrz = read_mrz(repaired)
        if mrz is None and not self.policy.check_checksums:
            mrz = read_mrz(document.get("mrz"), check_checksums=False)
        if mrz is None:
            return rejected, "invalid_mrz"
        first, second = mrz
        if not fields["Last_name_ru"] or not fields["First_name_ru"]:
            return rejected, "missing_name"
        if any(not re.fullmatch(r"[А-ЯЁ]+(?:[ -][А-ЯЁ]+)*", fields[key])
               for key in NAME_FIELDS if fields[key]):
            return rejected, "name_format"
        reconciled = (reconcile_names(first, document["mrz"]["lines"][0]["char_conf"], fields, viz, source)
                      if self.policy.repair_zones else fields if name_line(fields) == first else None)
        if reconciled is None:
            return rejected, "name_disagreement"
        fields = reconciled
        # A truncated MRZ does not corroborate the invisible name suffix.
        unpadded_length = len("PNRUS" + encode_name(fields[NAME_FIELDS[0]]) + "<<"
                              + encode_name(fields[NAME_FIELDS[1]]) + "<"
                              + encode_name(fields[NAME_FIELDS[2]]))
        if unpadded_length > 44 and any(minimum_confidence(viz[key]) < (1.0 if source == "mobile_v2" else 0.95)
                                         for key in NAME_FIELDS if fields[key]):
            return rejected, "uncertain_truncated_name"
        # Missing patronymic is accepted only if it is explicitly empty in MRZ.
        if viz["Middle_name_ru"] is None and not first.endswith("<"):
            return rejected, "ambiguous_missing_patronymic"
        numeric = numeric_fields(second, fields)
        if numeric is None:
            return rejected, "invalid_or_ambiguous_dates"
        for key, value in numeric.items():
            if fields[key] and fields[key] != value:
                # Full checksum validation plus a confident MRZ may repair a
                # low-confidence visual substitution, never a confident conflict.
                observed, scores = scored_text(viz[key])
                diffs = [i for i, (a, b) in enumerate(zip(observed, value)) if a != b]
                if (not self.policy.repair_zones or len(observed) != len(value) or not 1 <= len(diffs) <= 2
                        or any(scores[i] >= low_threshold(source) for i in diffs)
                        or min(document["mrz"]["lines"][1]["char_conf"]) < corroborated_threshold(source)):
                    return rejected, "numeric_disagreement"
            fields[key] = value
        if fields["Issue_organisation_code"] not in self.reference:
            return rejected, "unknown_subdivision"
        organization = self.organization(viz["Issue_organization_ru"], fields["Issue_organisation_code"], source)
        if organization is None:
            return rejected, "uncertain_Issue_organization_ru"
        fields["Issue_organization_ru"] = organization
        return {"doc_id": document["doc_id"], "decision": "accept", "fields": fields}, "accept"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--score", choices=("min", "mean", "product"), default=Policy.score)
    parser.add_argument("--threshold", type=float, default=Policy.threshold)
    parser.add_argument("--mobile-threshold", type=float, default=Policy.mobile_threshold)
    parser.add_argument("--dictionary-mode", choices=("off", "safe"), default=Policy.organization_mode)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    try:
        policy = Policy(score=args.score, threshold=args.threshold, mobile_threshold=args.mobile_threshold,
                        organization_mode=args.dictionary_mode)
    except ValueError as error:
        parser.error(str(error))
    processor = Postprocessor(policy=policy)
    counts = Counter()
    ids = set()
    # Write atomically so an invalid JSON line cannot leave a partial submission.
    temporary = args.output.with_name(args.output.name + ".tmp")
    try:
        with args.input.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
            for number, line in enumerate(source, 1):
                document = json.loads(line)
                doc_id = document["doc_id"]
                if not isinstance(doc_id, str) or not doc_id or doc_id in ids:
                    raise ValueError(f"line {number}: invalid or repeated doc_id")
                ids.add(doc_id)
                answer, reason = processor.process(document)
                counts[reason] += 1
                target.write(json.dumps(answer, ensure_ascii=False) + "\n")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Documents: {len(ids)}, accepted: {counts['accept']}, rejected: {len(ids) - counts['accept']}", file=sys.stderr)


if __name__ == "__main__":
    main()
