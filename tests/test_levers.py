"""Tests for the two levers added in the final build phase: the fee-setting incumbent's review-fee override and the
labour block's junior-operator switch."""
import os, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from model_config import CFG, reset_config, set_config
import domain_cases as dc
from negotiated_order_lab import (execute_hybrid_tool_plus_critic, execute_incumbent_react_agent, negotiate_workplace_governance,
                                  registered_review_fraction)
from labor_market import OccupationLabor


class FeeOverride(unittest.TestCase):
    def setUp(self):
        reset_config()

    def test_default_fee_is_the_registered_one_and_a_high_fee_moves_the_client(self):
        from compiler_agent import synthesize_compiled_tool_from_handbook, load_compiled_callable
        import tempfile
        spec = next(s for s in dc.FULL_EXTENDED_CORPUS if s.ring == "Inner_Core" and s.statutory_gate == 1)
        pk = dc.build_multi_doc_packets_for_task(spec)
        with tempfile.TemporaryDirectory() as d:
            fn = load_compiled_callable(synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, pk[0].doc_handbook, d, "v1"))
            moved = 0
            for p in pk:
                hyb = execute_hybrid_tool_plus_critic(spec, fn, p.doc_primary_intake, p.doc_secondary_schedule, p.doc_tertiary_annex)
                solo = execute_incumbent_react_agent(spec, p, "SOLO")
                col = execute_incumbent_react_agent(spec, p, "COLLAB_AUDIT", challenger_out=fn(p.doc_primary_intake, p.doc_secondary_schedule, p.doc_tertiary_annex))
                a = negotiate_workplace_governance(spec, p, hyb, col, solo)
                b = negotiate_workplace_governance(spec, p, hyb, col, solo, review_fraction=registered_review_fraction(spec))
                self.assertEqual((a.selected_governance, a.labor_cost_units), (b.selected_governance, b.labor_cost_units))
                c = negotiate_workplace_governance(spec, p, hyb, col, solo, review_fraction=5.0)   # a prohibitive fee
                self.assertNotEqual(c.selected_governance, "ACCOUNTABILITY_SETTLEMENT")
                moved += int(a.selected_governance == "ACCOUNTABILITY_SETTLEMENT")
            self.assertGreater(moved, 0)


class JuniorOperatorSwitch(unittest.TestCase):
    def setUp(self):
        reset_config()

    def _step(self):
        lab = OccupationLabor("X", 2.0, 0.3)
        tasks = [{"weight": 1.0, "poach": 0.4, "settle": 0.4, "gate": 0.2, "hard_gate": 0.1, "hard_settle": 0.2, "unit_cost": 1.5}]
        return lab.step(1, tasks)

    def test_junior_demand_follows_the_switch(self):
        on = self._step()
        set_config(labor_junior_operates_tool=False)
        off = self._step()
        # registered: juniors on settled and poached work (0.8 of volume); replaced: juniors assist on gated work only (0.2)
        self.assertAlmostEqual(on["J_demand"] / off["J_demand"], 0.8 / 0.2, places=6)
        # learning per apprentice: registered counts settled hard cases at lambda; replaced counts gated hard cases only
        self.assertGreater(on["learning_per_apprentice"] * on["A_stock"], off["learning_per_apprentice"] * off["A_stock"])


if __name__ == "__main__":
    unittest.main()
