"""Воспроизводимая оценка качества с проверкой действительности паспорта по датам.

Исходная отложенная выборка уже изучалась и использовалась для диагностики ошибок.
Её нельзя представлять как новую независимую проверку доработанных правил.

python validate.py --split dev
python validate.py --split holdout
python validate.py --split all --answers train_answers.jsonl
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from postprocess import FIELDS, Postprocessor, norm, parse_date, read_field, valid_dates

ROOT = Path(__file__).resolve().parent


def is_holdout(doc_id):
    return int(hashlib.sha256(doc_id.encode()).hexdigest(), 16) % 5 == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="holdout")
    parser.add_argument("--answers", type=Path)
    args = parser.parse_args()
    processor = Postprocessor()
    labels = {}
    with (ROOT / "data/train_labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if args.split == "all" or is_holdout(row["doc_id"]) == (args.split == "holdout"):
                labels[row["doc_id"]] = row["fields"]
    results, reasons, sources = [], Counter(), {}
    errors, field_errors = [], Counter()
    baseline_exact = baseline_valid = accepted = invalid = total_invalid = 0
    with (ROOT / "data/train_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            document = json.loads(line)
            if document["doc_id"] not in labels:
                continue
            gt = labels[document["doc_id"]]
            is_valid = valid_dates(parse_date(gt["Birth_date"]), parse_date(gt["Issue_date"]))
            total_invalid += not is_valid
            baseline_ok = all(read_field(document["viz"].get(f)) == norm(gt[f]) for f in FIELDS)
            baseline_exact += baseline_ok
            baseline_valid += baseline_ok and is_valid
            answer, reason = processor.process(document)
            results.append(answer)
            reasons[reason] += 1
            source = sources.setdefault(document["source"], Counter())
            source["total"] += 1
            if answer["decision"] != "accept":
                continue
            accepted += 1
            source["accepted"] += 1
            wrong = [f for f in FIELDS if norm(answer["fields"][f]) != norm(gt[f])]
            invalid += not is_valid
            if wrong or not is_valid:
                source["errors"] += 1
                errors.append({"doc_id": document["doc_id"], "fields": wrong, "invalid": not is_valid})
                field_errors.update(wrong)
    if args.answers:
        args.answers.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    print(json.dumps({
        "evaluation_note": "Train and original holdout have already been inspected; these are development results.",
        "split": args.split, "total": len(results), "baseline_exact": baseline_exact,
        "baseline_exact_and_valid": baseline_valid, "invalid_in_labels": total_invalid,
        "accepted": accepted, "accepted_correct_and_valid": accepted - len(errors),
        "accepted_errors": len(errors), "accepted_invalid": invalid,
        "field_errors": field_errors, "sources": sources,
        "reasons": reasons, "error_documents": errors,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
