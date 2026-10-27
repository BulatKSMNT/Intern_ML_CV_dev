"""Воспроизводимые эксперименты с оценками уверенности и порогами принятия.

Выполняются на CPU, используется только стандартная библиотека Python.

python experiments.py --output experiments

Прежняя отложенная выборка уже изучалась. Эти результаты относятся к разработке
и не являются новой независимой проверкой. Разметка test не предоставлена;
документы test обрабатываются только после выбора настроек на train.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from postprocess import FIELDS, Policy, Postprocessor, norm, parse_date, read_field, valid_dates

ROOT = Path(__file__).resolve().parent


def fold_id(doc_id):
    return int(hashlib.sha256(doc_id.encode()).hexdigest(), 16) % 5


def wilson(errors, accepted):
    """Рассчитать двусторонний 95%-й интервал; после выбора настроек это не гарантия риска."""
    if not accepted:
        return 0.0, 1.0
    z = 1.959963984540054
    p = errors / accepted
    denominator = 1 + z * z / accepted
    center = (p + z * z / (2 * accepted)) / denominator
    delta = z * math.sqrt(p * (1 - p) / accepted + z * z / (4 * accepted**2)) / denominator
    return max(0.0, center - delta), min(1.0, center + delta)


def policies():
    # The grid and the ranking criterion are fixed before inspecting its results.
    base = Policy(score="min", threshold=0.95, mobile_threshold=1.0, organization_mode="safe")
    result = {
        "legacy": replace(base, organization_mode="legacy"),
        "safe_min_0.95_v2_1": base,
        "no_dictionary": replace(base, organization_mode="off"),
        "no_repairs": replace(base, organization_mode="off", repair_zones=False),
        "no_geometry": replace(base, check_geometry=False),
        "no_checksums": replace(base, check_checksums=False),
        "same_threshold_for_sources": replace(base, mobile_threshold=0.95),
    }
    candidates = []
    grids = {
        "min": (0.85, 0.90, 0.95, 0.98, 0.99),
        "mean": (0.95, 0.98, 0.99, 0.995, 0.999),
        "product": (0.50, 0.70, 0.80, 0.90, 0.95),
    }
    for score, thresholds in grids.items():
        for threshold in thresholds:
            for mobile in (0.98, 0.995, 1.0):
                for mode, organization, zone_repair in (("safe", "safe", True),
                                                         ("no_dict", "off", True),
                                                         ("no_repairs", "off", False)):
                    name = f"{mode}_{score}_{threshold:g}_v2_{mobile:g}"
                    result[name] = replace(base, score=score, threshold=threshold, mobile_threshold=mobile,
                                           organization_mode=organization, repair_zones=zone_repair)
                    candidates.append(name)
    return result, candidates


def metrics(rows):
    total = len(rows)
    accepted = sum(row["accepted"] for row in rows)
    errors = sum(row["error"] for row in rows)
    lower, upper = wilson(errors, accepted)
    return {
        "total": total, "accepted": accepted, "correct": accepted - errors, "errors": errors,
        "invalid": sum(row["invalid"] for row in rows),
        "coverage": accepted / total if total else 0.0,
        "error_rate": errors / accepted if accepted else None,
        "risk_lower_95": lower, "risk_upper_95": upper,
        "corrected_documents": sum(row["changed"] for row in rows),
        "fixed_fields": sum(row["fixed_fields"] for row in rows),
        "wrong_changed_fields": sum(row["wrong_changed_fields"] for row in rows),
        "damaged_correct_fields": sum(row["damaged_correct_fields"] for row in rows),
    }


def evaluate(processor, documents, labels):
    rows = []
    for doc in documents:
        answer, reason = processor.process(doc)
        gt = labels[doc["doc_id"]]
        accepted = answer["decision"] == "accept"
        fields = answer.get("fields", {})
        wrong = [key for key in FIELDS if accepted and norm(fields[key]) != norm(gt[key])]
        changed = [key for key in FIELDS if accepted and fields[key] != read_field(doc["viz"].get(key))]
        invalid = accepted and not valid_dates(parse_date(gt["Birth_date"]), parse_date(gt["Issue_date"]))
        rows.append({
            "doc_id": doc["doc_id"], "source": doc["source"], "fold": fold_id(doc["doc_id"]),
            "accepted": accepted, "error": bool(wrong or invalid), "invalid": invalid,
            "changed": bool(changed), "wrong_fields": wrong, "reason": reason,
            "fixed_fields": sum(key not in wrong for key in changed),
            "wrong_changed_fields": sum(key in wrong for key in changed),
            "damaged_correct_fields": sum(key in wrong and read_field(doc["viz"].get(key)) == norm(gt[key]) for key in changed),
        })
    return rows


def select(runs, candidates, excluded_fold):
    """Выбрать минимальную верхнюю оценку риска, при равенстве — покрытие и имя режима.

    Критерий учитывает объём проверки, чтобы не предпочитать крошечную выборку
    без ошибок. Он не гарантирует нулевую ошибку и не подтверждает статистически
    надёжность лучшего варианта после перебора настроек.
    """
    def key(name):
        m = metrics([row for row in runs[name] if row["fold"] != excluded_fold])
        return m["risk_upper_95"], -m["accepted"], name
    return min(candidates, key=key)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/train_predictions.jsonl").open(encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle]
    with (ROOT / "data/train_labels.jsonl").open(encoding="utf-8") as handle:
        labels = {doc["doc_id"]: doc["fields"] for doc in map(json.loads, handle)}
    configurations, candidates = policies()
    runs, summaries, csv_rows = {}, {}, []
    for i, (name, policy) in enumerate(configurations.items(), 1):
        rows = evaluate(Postprocessor(policy=policy), documents, labels)
        runs[name] = rows
        splits = {"all": rows, "dev": [row for row in rows if row["fold"] != 0],
                  "reviewed_holdout": [row for row in rows if row["fold"] == 0]}
        summary = {"policy": asdict(policy), "splits": {split: metrics(data) for split, data in splits.items()}}
        summary["sources"] = {source: metrics([row for row in rows if row["source"] == source])
                              for source in ("web", "mobile", "mobile_v2")}
        summary["reasons"] = dict(Counter(row["reason"] for row in rows))
        summary["error_documents"] = [{k: row[k] for k in ("doc_id", "source", "wrong_fields", "invalid")}
                                      for row in rows if row["error"]]
        summaries[name] = summary
        for split, m in summary["splits"].items():
            csv_rows.append({"name": name, "split": split, **asdict(policy), **m})
        if i % 10 == 0 or i == len(configurations):
            print(f"Evaluated {i}/{len(configurations)} policies", flush=True)

    selected = select(runs, candidates, excluded_fold=0)
    selected_policy = configurations[selected]
    selections, out_of_fold = [], []
    for fold in range(5):
        choice = select(runs, candidates, excluded_fold=fold)
        validation = [row for row in runs[choice] if row["fold"] == fold]
        selections.append({"fold": fold, "selected": choice, "validation": metrics(validation)})
        out_of_fold.extend(validation)

    # Do not use test coverage to choose thresholds.
    with (ROOT / "data/test_predictions.jsonl").open(encoding="utf-8") as handle:
        test = [json.loads(line) for line in handle]
    test_counts = {}
    for name in ("legacy", selected):
        processor = Postprocessor(policy=configurations[name])
        accepted_sources, reasons = Counter(), Counter()
        for doc in test:
            answer, reason = processor.process(doc)
            reasons[reason] += 1
            if answer["decision"] == "accept":
                accepted_sources[doc["source"]] += 1
        test_counts[name] = {"total": len(test), "accepted": sum(accepted_sources.values()),
                             "accepted_sources": dict(accepted_sources), "reasons": dict(reasons)}
    source_metrics = summaries[selected]["sources"]
    forecast = sum(count * source_metrics[source]["error_rate"]
                   for source, count in test_counts[selected]["accepted_sources"].items())

    details = {
        "protocol": {
            "fold": "int(sha256(doc_id), 16) % 5; original reviewed holdout = fold 0",
            "selection": "minimum two-sided Wilson 95% upper error-rate estimate on folds 1-4, then maximum accepted",
            "limitation": "All train labels, including the original holdout, were previously inspected. Fold selection is a stability check, not fresh independent validation.",
            "confidence_methods": {"min": "min(c)", "mean": "sum(c)/len(c)", "product": "product(c), heuristic, not a calibrated probability"},
            "selection_candidates": candidates,
            "excluded_from_selection": "legacy, geometry/checksum/source-threshold diagnostic ablations; dictionary and cross-zone repair modes are INCLUDED",
        },
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (
            ROOT / "postprocess.py", ROOT / "experiments.py", ROOT / "data/train_predictions.jsonl",
            ROOT / "data/train_labels.jsonl", ROOT / "data/test_predictions.jsonl", ROOT / "reference/subdivisions.csv")},
        "selected": selected, "configurations": summaries,
        "fold_selection": selections, "pooled_fold_selection": metrics(out_of_fold),
        "test": test_counts, "forecast_test_errors": forecast,
    }
    with (args.output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.output / "results.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "selected_policy.json").write_text(json.dumps(asdict(selected_policy), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "policy": asdict(selected_policy),
                      "metrics": summaries[selected]["splits"], "test": test_counts[selected],
                      "forecast_errors": forecast}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
