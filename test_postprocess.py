"""Проверки граничных случаев и защита от повторения ошибок: python -m unittest -v."""

import copy
import json
import unittest
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from postprocess import (
    FIELDS, Policy, Postprocessor, anniversary, checksum, complete_field, confidence_score, name_line,
    numeric_fields, numeric_line, one_edit_away, read_mrz, scored_text, valid_dates,
)


def field(text, confidence=0.999, y=0):
    return {"box": [10, y, 500, 30], "conf": confidence,
            "lines": [{"text": text, "box": [10, y, 480, 30],
                       "char_conf": [confidence] * len(text)}]}


class PostprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = Postprocessor(policy=Policy(organization_mode="safe"))
        # One fixed development fixture. No holdout records are used for tests.
        root = Path(__file__).resolve().parent
        with (root / "data/train_predictions.jsonl").open(encoding="utf-8") as handle:
            cls.document = json.loads(next(handle))
        with (root / "data/train_labels.jsonl").open(encoding="utf-8") as handle:
            cls.fields = json.loads(next(handle))["fields"]

    def clean_document(self):
        doc = copy.deepcopy(self.document)
        doc["source"] = "web"
        doc["viz"] = {key: field(value, y=i * 40) for i, (key, value) in enumerate(self.fields.items())}
        return doc

    def test_known_mrz_checksum_and_offsets(self):
        text = "6122839687RUS1006102M<<<<<<<4250911620004<28"
        self.assertEqual(checksum("612283968"), "7")
        self.assertEqual(numeric_line(self.fields), text)
        self.assertEqual(numeric_fields(text, self.fields)["Licence_number"], "61 24 283968")
        self.assertEqual(numeric_fields(text, self.fields)["Issue_organisation_code"], "620-004")

    def test_each_checksum_is_checked(self):
        for index in (9, 19, 42, 43):
            with self.subTest(index=index):
                mrz = copy.deepcopy(self.document["mrz"])
                text = mrz["lines"][1]["text"]
                mrz["lines"][1]["text"] = text[:index] + str((int(text[index]) + 1) % 10) + text[index + 1:]
                self.assertIsNone(read_mrz(mrz))

    def test_clean_accept(self):
        answer, reason = self.processor.process(self.clean_document())
        self.assertEqual(reason, "accept")
        self.assertEqual(answer["fields"], self.fields)
        self.assertEqual(set(answer["fields"]), set(FIELDS))

    def test_high_confidence_name_conflict_rejects(self):
        doc = self.clean_document()
        doc["viz"]["First_name_ru"] = field("ЗАКЕРИХ")
        self.assertEqual(self.processor.process(doc)[0]["decision"], "reject")

    def test_low_confidence_name_substitution_is_corrected(self):
        doc = self.clean_document()
        doc["viz"]["First_name_ru"] = field("ЗАКЕРИХ")
        doc["viz"]["First_name_ru"]["lines"][0]["char_conf"][-1] = 0.5
        self.assertEqual(self.processor.process(doc)[0]["fields"]["First_name_ru"], "ЗАКЕРИЙ")

    def test_missing_patronymic_is_not_assumed_empty(self):
        doc = self.clean_document()
        doc["viz"]["Middle_name_ru"] = None
        self.assertEqual(self.processor.process(doc)[0]["decision"], "reject")

    def test_explicitly_absent_patronymic(self):
        doc = self.clean_document()
        fields = dict(self.fields, Middle_name_ru="")
        doc["viz"]["Middle_name_ru"] = None
        doc["mrz"]["lines"][0]["text"] = name_line(fields)
        self.assertEqual(self.processor.process(doc)[0]["fields"]["Middle_name_ru"], "")

    def test_missing_date_with_ambiguous_century_rejects(self):
        fields = dict(self.fields, Birth_date="08.05.2000", Issue_date="01.03.2015")
        text = numeric_line(fields)
        fields["Birth_date"] = ""
        self.assertIsNone(numeric_fields(text, fields))

    def test_expired_visual_date_never_changes_century(self):
        fields = dict(self.fields, Birth_date="08.05.2000", Issue_date="01.03.2015")
        self.assertIsNone(numeric_fields(numeric_line(fields), fields))

    def test_missing_issue_date_can_be_recovered(self):
        doc = self.clean_document()
        doc["viz"]["Issue_date"] = None
        self.assertEqual(self.processor.process(doc)[0]["fields"]["Issue_date"], self.fields["Issue_date"])

    def test_dropped_top_bottom_and_middle_lines(self):
        value = {"box": [10, 0, 500, 90], "lines": [field("СТРОКА", y=y)["lines"][0] for y in (0, 30, 60)]}
        self.assertTrue(complete_field(value))
        for index in range(3):
            incomplete = copy.deepcopy(value)
            incomplete["lines"].pop(index)
            self.assertFalse(complete_field(incomplete))

    def test_sort_lines_by_coordinates(self):
        value = {"box": [10, 0, 500, 60], "lines": [field("ВТОРАЯ", y=30)["lines"][0], field("ПЕРВАЯ", y=0)["lines"][0]]}
        self.assertTrue(complete_field(value))
        self.assertEqual(scored_text(value)[0], "ПЕРВАЯ ВТОРАЯ")

    def test_misaligned_confidence_rejects(self):
        value = field("ТЕКСТ")
        value["lines"][0]["char_conf"].pop()
        self.assertFalse(complete_field(value))

    def test_no_reference_only_completion(self):
        doc = self.clean_document()
        doc["viz"]["Issue_organization_ru"] = None
        self.assertEqual(self.processor.process(doc)[0]["decision"], "reject")

    def test_mobile_v2_needs_stricter_free_text_confidence(self):
        doc = self.clean_document()
        doc["source"] = "mobile_v2"
        self.assertEqual(self.processor.process(doc)[0]["decision"], "reject")

    def test_20_and_45_year_90_day_boundaries(self):
        birth = date(1981, 6, 1)
        for age, issued in ((20, date(1998, 1, 1)), (45, date(2010, 1, 1))):
            boundary = anniversary(birth, age) + timedelta(days=90)
            self.assertTrue(valid_dates(birth, issued, boundary))
            self.assertFalse(valid_dates(birth, issued, boundary + timedelta(days=1)))
            self.assertTrue(valid_dates(birth, anniversary(birth, age), boundary + timedelta(days=1)))

    def test_issue_before_14_or_in_future_rejects(self):
        self.assertFalse(valid_dates(date(2010, 1, 1), date(2023, 12, 31)))
        self.assertFalse(valid_dates(date(1990, 1, 1), date(2026, 10, 2)))

    def test_leap_day_and_after_45(self):
        self.assertEqual(anniversary(date(2004, 2, 29), 14), date(2018, 3, 1))
        self.assertTrue(valid_dates(date(1960, 1, 1), date(2005, 1, 1)))

    def test_ambiguous_dictionary_substitution_rejects(self):
        value = field("ГУ ПМВД РОССИИ ПО НОВОСИБИРСКОЙ ОБЛ")
        value["lines"][0]["char_conf"][3] = 0.45
        original = copy.deepcopy(value)
        self.assertIsNone(self.processor.organization(value, "540-038", "mobile"))
        self.assertEqual(value, original)
        legacy = Postprocessor(policy=Policy(organization_mode="legacy"))
        self.assertEqual(legacy.organization(value, "540-038", "mobile"), "ГУ ОМВД РОССИИ ПО НОВОСИБИРСКОЙ ОБЛ")

    def test_unique_dictionary_substitution_remains_available(self):
        answer, reason = self.processor.process(copy.deepcopy(self.document))
        self.assertEqual(reason, "accept")
        self.assertEqual(answer["fields"]["Issue_organization_ru"], self.fields["Issue_organization_ru"])

    def test_competing_edit_types(self):
        for a, b in (("ПМВД", "МВД"), ("ПМВД", "ОМВД"), ("МВД", "ОМВД"),
                     ("АБВВ", "АБВ"), ("ОТДЕЛ", "ОТДЕЛА")):
            with self.subTest(a=a, b=b):
                self.assertTrue(one_edit_away(a, b))
                self.assertTrue(one_edit_away(b, a))
        self.assertFalse(one_edit_away("МВД", "МВД"))
        self.assertFalse(one_edit_away("МВД", "УФМС"))

    def test_confidence_aggregation_does_not_hide_its_definition(self):
        scores = [0.99, 0.99, 0.5]
        self.assertEqual(confidence_score(scores, "min"), 0.5)
        self.assertAlmostEqual(confidence_score(scores, "mean"), 2.48 / 3)
        self.assertAlmostEqual(confidence_score(scores, "product"), 0.49005)
        self.assertEqual(confidence_score([], "product"), 0)

    def test_invalid_thresholds_reject(self):
        for value in (0, -1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Policy(threshold=value)

    def test_no_repairs_mode_preserves_conflicts(self):
        doc = self.clean_document()
        doc["viz"]["First_name_ru"] = field("ЗАКЕРИХ")
        doc["viz"]["First_name_ru"]["lines"][0]["char_conf"][-1] = 0.5
        processor = Postprocessor(policy=Policy(repair_zones=False))
        self.assertEqual(processor.process(doc)[0]["decision"], "reject")

    def test_geometry_ablation_keeps_schema_validation(self):
        value = field("ТЕКСТ")
        value["box"][3] = 90
        self.assertFalse(complete_field(value))
        self.assertTrue(complete_field(value, check_geometry=False))
        value["lines"][0]["char_conf"].pop()
        self.assertFalse(complete_field(value, check_geometry=False))

    def test_default_policy_matches_experiment_selection(self):
        path = Path(__file__).resolve().parent / "experiments/selected_policy.json"
        self.assertEqual(asdict(Policy()), json.loads(path.read_text()))
        self.assertEqual(Postprocessor().process(self.clean_document())[0]["decision"], "accept")

    def test_selection_does_not_use_excluded_fold(self):
        from experiments import select
        def row(fold, error):
            return dict(fold=fold, accepted=True, error=error, invalid=False, changed=False,
                        fixed_fields=0, wrong_changed_fields=0, damaged_correct_fields=0)
        runs = {
            "a": [row(1, False) for _ in range(100)] + [row(0, True) for _ in range(100)],
            "b": [row(1, True) for _ in range(100)] + [row(0, False) for _ in range(100)],
        }
        self.assertEqual(select(runs, list(runs), excluded_fold=0), "a")
        self.assertEqual(select(runs, list(runs), excluded_fold=1), "b")


if __name__ == "__main__":
    unittest.main()
