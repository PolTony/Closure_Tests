"""Smoke tests for the reference configuration. Run: python3 -m unittest tests/test_registered_design.py"""
import hashlib, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from model_config import reset_config
import domain_cases as dc

EXPECTED_GT_HASH = "731f496441c8b9807f2e726034f31fb5a2e9128e664dc76ad157a3b864ec6272"


def gt_hash() -> str:
    h = hashlib.sha256()
    for spec in dc.FULL_EXTENDED_CORPUS:
        for p in dc.build_multi_doc_packets_for_task(spec):
            h.update(f"{p.packet_id}|{p.complexity_tier}|{int(p.is_silent_collision)}|{p.gt_diagnosis}|{p.gt_inference}|{p.gt_treatment}".encode())
    return h.hexdigest()


class RegisteredDesign(unittest.TestCase):
    def setUp(self):
        reset_config()

    def test_ground_truth_unchanged(self):
        self.assertEqual(gt_hash(), EXPECTED_GT_HASH)

    def test_compiler_v1_perfect_on_tiers_0_and_1(self):
        from compiler_agent import synthesize_compiled_tool_from_handbook, load_compiled_callable
        import tempfile
        spec = dc.FULL_EXTENDED_CORPUS[0]
        pk = dc.build_multi_doc_packets_for_task(spec)
        with tempfile.TemporaryDirectory() as d:
            fn = load_compiled_callable(synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, pk[0].doc_handbook, d, "v1"))
            for p in pk:
                if p.complexity_tier <= 1:
                    out = fn(p.doc_primary_intake, p.doc_secondary_schedule, p.doc_tertiary_annex)
                    self.assertEqual(out["inference"], p.gt_inference)
                    self.assertAlmostEqual(out["treatment"], p.gt_treatment, places=2)

    def test_incumbent_solo_inference_exact(self):
        from negotiated_order_lab import execute_incumbent_react_agent
        spec = dc.FULL_EXTENDED_CORPUS[0]
        for p in dc.build_multi_doc_packets_for_task(spec):
            out = execute_incumbent_react_agent(spec, p, "SOLO")
            self.assertEqual(out["inference"], p.gt_inference)


if __name__ == "__main__":
    unittest.main()
