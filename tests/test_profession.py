"""Tests for the learning profession (profession.py) and the incumbent's knowledge mask.

  * the full mask reproduces the registered incumbent exactly;
  * a professional without the convention reads a convention case as the handbook does and nothing else changes;
  * a professional without the registry cannot resolve an overt precedent memo and nothing else changes;
  * a reviewer without the convention defers to a program that applied an explicit handler (model-card switch);
  * the knowledge module conserves headcounts in step with the labour block's flows, promotes the most experienced
    juniors first, and, at the pre-compilation pace, transmits both components exactly at promotion.
"""
import os, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from model_config import CFG, reset_config, set_config
import domain_cases as dc
from negotiated_order_lab import FULL_KNOWLEDGE, execute_incumbent_react_agent, parse_doc_fields
from profession import OccupationKnowledge, pretool_exposure_per_head


def _handbook_answer(spec, p):
    f = parse_doc_fields(p.doc_primary_intake, p.doc_secondary_schedule, p.doc_tertiary_annex)
    tier = 1 if f["EXPLICIT_TIER1_OVERRIDE"] else 0
    sev = f["ANNEX_SEVERITY"] if tier == 1 else 0.0
    return dc.evaluate_rulebook(spec, f["V1"], f["V2"], f["V3"], f["V4"], sev, tier)


class KnowledgeMask(unittest.TestCase):
    def setUp(self):
        reset_config()

    def test_full_mask_is_the_registered_incumbent(self):
        for spec in dc.FULL_EXTENDED_CORPUS[:6]:
            for p in dc.build_multi_doc_packets_for_task(spec):
                for mode in ("SOLO", "COLLAB_AUDIT"):
                    a = execute_incumbent_react_agent(spec, p, mode, challenger_out={"diagnosis": "x", "inference": "x", "treatment": 0.0})
                    b = execute_incumbent_react_agent(spec, p, mode, challenger_out={"diagnosis": "x", "inference": "x", "treatment": 0.0}, knowledge=dict(FULL_KNOWLEDGE))
                    self.assertEqual(a, b)

    def test_without_the_convention_a_convention_case_reads_as_routine(self):
        spec = next(s for s in dc.FULL_EXTENDED_CORPUS if s.ring == "Middle_Buffer")
        mask = {"convention": False, "registry": True}
        n2 = 0
        for p in dc.build_multi_doc_packets_for_task(spec):
            full = execute_incumbent_react_agent(spec, p, "SOLO")
            masked = execute_incumbent_react_agent(spec, p, "SOLO", knowledge=mask)
            if p.complexity_tier == 2:
                n2 += 1
                d, i, t = _handbook_answer(spec, p)
                self.assertEqual(masked["hop"], "HOP2_CONVENTION_UNKNOWN")
                self.assertEqual(masked["inference"], i)
                self.assertNotEqual(masked["inference"], p.gt_inference)
            else:
                self.assertEqual(full, masked)
        self.assertGreater(n2, 0)

    def test_without_the_registry_an_overt_memo_cannot_be_resolved(self):
        spec = next(s for s in dc.FULL_EXTENDED_CORPUS if s.ring == "Inner_Core")
        mask = {"convention": True, "registry": False}
        n3 = 0
        for p in dc.build_multi_doc_packets_for_task(spec):
            full = execute_incumbent_react_agent(spec, p, "SOLO")
            masked = execute_incumbent_react_agent(spec, p, "SOLO", knowledge=mask)
            if p.complexity_tier == 3:
                n3 += 1
                self.assertIn(masked["hop"], ("HOP3_REGISTRY_UNKNOWN", "HOP2_ROUTINE"))
                self.assertNotEqual(masked["inference"], p.gt_inference)
            else:
                self.assertEqual(full, masked)
        self.assertGreater(n3, 0)

    def test_deference_to_an_explicit_handler(self):
        spec = next(s for s in dc.FULL_EXTENDED_CORPUS if s.ring == "Middle_Buffer")
        p = next(x for x in dc.build_multi_doc_packets_for_task(spec) if x.complexity_tier == 2)
        tool = {"diagnosis": p.gt_diagnosis, "inference": p.gt_inference, "treatment": p.gt_treatment, "fired": "H1", "unhandled_fields": []}
        out = execute_incumbent_react_agent(spec, p, "COLLAB_AUDIT", challenger_out=tool, knowledge={"convention": False, "registry": True})
        self.assertTrue(out["mode"].startswith("COLLAB_DEFER_UNKNOWING"))
        self.assertEqual(out["inference"], p.gt_inference)
        set_config(learn_defer_when_unknowing=False)
        out2 = execute_incumbent_react_agent(spec, p, "COLLAB_AUDIT", challenger_out=tool, knowledge={"convention": False, "registry": True})
        self.assertTrue(out2["mode"].startswith("COLLAB_OVERRIDE"))
        self.assertNotEqual(out2["inference"], p.gt_inference)


class KnowledgeModule(unittest.TestCase):
    def setUp(self):
        reset_config()

    def _profession(self):
        pr = OccupationKnowledge("Test")
        n = int(CFG.labor_apprenticeship_periods)
        pre = {"convention": 10.0, "registry": 20.0}
        pr.initialise(16.0, [[2.0, 80.0 * k / n] for k in range(n)], 80.0, pre, 1)
        return pr, pre

    def test_initial_state_and_calibration(self):
        pr, pre = self._profession()
        self.assertEqual(pr.share("seniors", "convention"), 1.0)
        self.assertEqual(pr.share("seniors", "registry"), 1.0)
        self.assertAlmostEqual(pr.share("juniors", "convention"), 1.0 / CFG.labor_apprenticeship_periods)   # the cohort due for promotion already holds it
        self.assertAlmostEqual(pr.kappa["convention"], CFG.labor_apprenticeship_periods * pre["convention"])

    def test_pretool_pace_transmits_both_components_at_promotion(self):
        pr, pre = self._profession()
        A = pr.headcount("juniors")
        pr.credit("juniors", "convention", pre["convention"] * A)
        pr.credit("juniors", "registry", pre["registry"] * A)
        rec = pr.end_period(1, {"I_stock": 16.0, "A_stock": A, "promotions": 0.8, "hires": 0.8})
        self.assertAlmostEqual(rec["promoted_hold_convention"], 1.0)
        self.assertAlmostEqual(rec["promoted_hold_registry"], 1.0)
        self.assertAlmostEqual(pr.share("seniors", "convention"), 1.0)
        self.assertAlmostEqual(pr.headcount("seniors"), 16.0 * (1 - CFG.labor_delta) + 0.8)
        self.assertAlmostEqual(pr.headcount("juniors"), A - 0.8 + 0.8)

    def test_no_exposure_promotes_without_knowledge_and_seniors_decay(self):
        pr, pre = self._profession()
        A = pr.headcount("juniors")
        for t in range(1, 6):
            rec = pr.end_period(t, {"I_stock": pr.headcount("seniors"), "A_stock": A, "promotions": 0.8, "hires": 0.8})
        self.assertEqual(rec["promoted_hold_convention"], 0.0)
        self.assertLess(pr.share("seniors", "convention"), 1.0)
        self.assertLess(pr.share("seniors", "registry"), 1.0)

    def test_promotions_take_the_most_experienced_first(self):
        pr, pre = self._profession()
        A = pr.headcount("juniors")
        pr.credit("juniors", "convention", pre["convention"] * A)
        rec = pr.end_period(1, {"I_stock": 16.0, "A_stock": A, "promotions": 1.0, "hires": 1.0})
        top = max(pr.juniors, key=lambda c: c.exposure["convention"])
        self.assertAlmostEqual(top.n, 1.0)                 # half of the (2.0-head) cohort that held was promoted, nothing from any other cohort
        self.assertAlmostEqual(rec["promoted_hold_convention"], 1.0)
        self.assertTrue(all(abs(c.n - 2.0) < 1e-9 for c in pr.juniors if c is not top and c.entered < 1))

    def test_pretool_exposure_per_head(self):
        pre = pretool_exposure_per_head({"a": 100.0, "b": 50.0}, {"a": {2: 0.2, 3: 0.1}, "b": {2: 0.4, 3: 0.0}}, 10.0)
        self.assertAlmostEqual(pre["convention"], (100 * 0.2 + 50 * 0.4) * CFG.learn_w_gate_junior / 10.0)
        self.assertAlmostEqual(pre["registry"], (100 * 0.1) * CFG.learn_w_gate_junior / 10.0)


if __name__ == "__main__":
    unittest.main()
