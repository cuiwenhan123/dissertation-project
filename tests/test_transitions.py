import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import transitions


def transition_payload() -> dict:
    combinations = []
    for model in sorted(transitions.MODELS):
        for degradation in sorted(transitions.DEGRADATIONS):
            combinations.append({
                "model": model,
                "degradation": degradation,
                "severityCounts": [{"severity": value, "counts": {}} for value in range(6)],
                "steps": [
                    {
                        "fromSeverity": value,
                        "toSeverity": value + 1,
                        "transitions": [
                            {"from": source, "to": target, "count": 0, "rate": 0.0}
                            for source in sorted(transitions.STATUSES)
                            for target in sorted(transitions.STATUSES)
                        ],
                    }
                    for value in range(5)
                ],
            })
    return {
        "schemaVersion": 1,
        "studyId": "study-test",
        "generatedAt": "2026-08-21T00:00:00+00:00",
        "method": {"matching": "test"},
        "objectCount": 10,
        "combinations": combinations,
    }


class TransitionAnalysisTests(unittest.TestCase):
    def test_loads_complete_transition_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transitions.json"
            path.write_text(json.dumps(transition_payload()), encoding="utf-8")
            payload = transitions.load_transition_analysis(path)
        self.assertEqual(payload["objectCount"], 10)
        self.assertEqual(len(payload["combinations"]), 6)

    def test_selects_one_model_degradation_combination(self):
        payload = transition_payload()
        with patch.object(transitions, "load_transition_analysis", return_value=payload):
            result = transitions.transition_analysis("transformer", "jpeg")
        self.assertEqual(result["selection"], {"model": "transformer", "degradation": "jpeg"})
        self.assertEqual(result["analysis"]["degradation"], "jpeg")

    def test_rejects_incomplete_archive(self):
        payload = transition_payload()
        payload["combinations"].pop()
        with self.assertRaisesRegex(ValueError, "six model-degradation"):
            transitions._validate(payload)


if __name__ == "__main__":
    unittest.main()
