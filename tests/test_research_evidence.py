from __future__ import annotations

import unittest

from scripts.verify_research_evidence import EVIDENCE, load_json, verify_appendix_values, verify_checksums


class ResearchEvidenceTests(unittest.TestCase):
    def test_public_evidence_checksums(self):
        verify_checksums()

    def test_appendix_c_values_match_frozen_results(self):
        results = load_json(EVIDENCE / "results.json")
        transitions = load_json(EVIDENCE / "analysis" / "object_failure_transitions.json")
        verify_appendix_values(results, transitions)


if __name__ == "__main__":
    unittest.main()
