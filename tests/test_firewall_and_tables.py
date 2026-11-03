"""Firewall, information-set and table-pinning tests. Run: python3 -m unittest discover tests

These tests pin what the submission claims about the laboratory:
  * the challenger side (template compiler, compiled tools, model-agent module) never imports the rule interpreter,
    the exact solver or the precedent registry; the model-agent module imports the registry only to render the
    incumbent's rulebook (its declared information set);
  * the rule-based incumbent reads nothing but the three case documents (its output is unchanged when the packet's
    hidden answer, parameters and tier labels are scrambled);
  * the registered headline tables (Table 1 and the ring table) reproduce to the printed decimal;
  * the statistics helpers agree with independently computed reference values.
"""
import os, re, sys, tempfile, unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from model_config import reset_config
import domain_cases as dc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source(name: str) -> str:
    import juri_paths
    with open(juri_paths.module_path(name), encoding="utf-8") as f:
        return f.read()


class ChallengerFirewall(unittest.TestCase):
    def setUp(self):
        reset_config()

    def test_template_compiler_imports_no_ground_truth_module(self):
        src = _source("compiler_agent.py")
        self.assertNotRegex(src, r"^\s*(from|import)\s+domain_cases", "the compiler must not import the generator/interpreter module")
        self.assertNotRegex(src, r"^\s*(from|import)\s+negotiated_order_lab")
        for name in ("evaluate_rulebook", "solve_non_isomorphic_domain_gt", "PRECEDENT_REGISTRY", "RULEBOOKS"):
            self.assertNotIn(name, src)

    def test_compiled_tools_are_standalone(self):
        from compiler_agent import synthesize_compiled_tool_from_handbook
        spec = dc.FULL_EXTENDED_CORPUS[0]
        pk = dc.build_multi_doc_packets_for_task(spec)
        archive = dc.build_public_sandbox_archive(spec)
        with tempfile.TemporaryDirectory() as d:
            a1 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, pk[0].doc_handbook, d, "v1")
            a2 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, pk[0].doc_handbook, d, "v2", sandbox=archive, previous=a1)
            a3 = synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, pk[0].doc_handbook, d, "v3", sandbox=archive, previous=a2)
            for art in (a1, a2, a3):
                code = art.source_code
                self.assertNotRegex(code, r"^\s*(from|import)\s+(domain_cases|negotiated_order_lab|model_config|labor_market)", re.M)
                for name in ("evaluate_rulebook", "PRECEDENT_REGISTRY", "gt_treatment", "gt_inference"):
                    self.assertNotIn(name, code)

    def test_model_agent_module_uses_registry_only_for_the_incumbent_rulebook(self):
        src = _source("llm_agents.py")
        self.assertNotIn("evaluate_rulebook", src)
        self.assertNotIn("solve_non_isomorphic_domain_gt", src)
        # the registry may appear only inside render_full_rulebook (the incumbent's declared information set)
        body = re.split(r"\ndef render_full_rulebook", src)
        self.assertEqual(len(body), 2)
        after = body[1].split("\ndef ", 1)[1] if "\ndef " in body[1] else ""
        before = body[0].split("from domain_cases import (", 1)[1].split(")", 1)[1]
        self.assertNotIn("PRECEDENT_REGISTRY", before)
        self.assertNotIn("PRECEDENT_REGISTRY", after)


class IncumbentInformationSet(unittest.TestCase):
    def setUp(self):
        reset_config()

    def test_incumbent_output_depends_only_on_the_documents(self):
        from negotiated_order_lab import execute_incumbent_react_agent
        for spec in dc.FULL_EXTENDED_CORPUS[:6]:
            for p in dc.build_multi_doc_packets_for_task(spec):
                scrambled = replace(p, gt_diagnosis="SCRAMBLED", gt_inference="SCRAMBLED", gt_treatment=-999.0,
                                    params={}, complexity_tier=-1, is_collision_case=False, is_silent_collision=False)
                for mode in ("SOLO", "COLLAB_AUDIT"):
                    a = execute_incumbent_react_agent(spec, p, mode, challenger_out={"diagnosis": "x", "inference": "x", "treatment": 0.0})
                    b = execute_incumbent_react_agent(spec, scrambled, mode, challenger_out={"diagnosis": "x", "inference": "x", "treatment": 0.0})
                    self.assertEqual(a, b)

    def test_tool_failure_is_escalated_not_trusted(self):
        from negotiated_order_lab import execute_hybrid_tool_plus_critic
        from model_config import CFG
        spec = dc.FULL_EXTENDED_CORPUS[0]
        crashed = lambda *a: {"diagnosis": "RUNTIME_ERROR", "inference": "RUNTIME_ERROR", "treatment": 0.0, "fired": "ERROR", "unhandled_fields": []}
        out = execute_hybrid_tool_plus_critic(spec, crashed, "", "", "")
        self.assertEqual(out["epistemic_uncertainty"], CFG.u_p_unparsed)
        self.assertEqual(out["critic_status"], "TOOL_FAILURE_ESCALATED")


class RegisteredTables(unittest.TestCase):
    def setUp(self):
        reset_config()

    def test_table1_and_rings_reproduce(self):
        from run_live_benchmark import run_full
        with tempfile.TemporaryDirectory() as d:   # never share the committed compiled_tools/ with a concurrent benchmark run
            s = run_full(tools_dir=d, out_dir=None, quiet=True)
        e2e = s["table1"]["3. All Multi-Doc Packets E2E"]
        self.assertEqual((e2e["A_Base"], e2e["A_Tool_v1"], e2e["A_Tool_v2"], e2e["A_Tool_v3"], e2e["A_Inc_Solo"], e2e["A_Collab"], e2e["N"]),
                         (33.4, 66.0, 74.4, 81.0, 90.6, 99.4, 1080))
        t2 = s["table1"]["2c. Tier 2: 1st-Order Piecewise Collision"]
        self.assertEqual((t2["A_Tool_v2"], t2["A_Tool_v3"], t2["N"]), (50.8, 91.5, 177))
        t3 = s["table1"]["2d. Tier 3: 2nd-Order Precedent Collision"]
        self.assertEqual((t3["A_Tool_v3"], t3["A_Inc_Solo"], t3["A_Collab"], t3["N"]), (0.0, 92.6, 96.3, 190))
        core = s["rings"]["Inner_Core"]
        self.assertEqual((core["Direct_Poaching_Pct"], core["Accountability_Settlement_Pct"], core["Incumbent_Gatekeeping_Pct"]), (21.7, 44.7, 33.6))
        self.assertEqual((core["SQ_Error_Rate_Pct"], core["CF1_Dereg_Error_Pct"], core["CF2_NoLiab_Error_Pct"]), (6.4, 13.6, 46.1))
        buf = s["rings"]["Middle_Buffer"]
        self.assertEqual((buf["Direct_Poaching_Pct"], buf["Accountability_Settlement_Pct"], buf["Incumbent_Gatekeeping_Pct"], buf["SQ_Error_Rate_Pct"]), (37.2, 55.3, 7.5, 1.1))
        self.assertAlmostEqual(s["table4_shield_on_gate_r2"], 0.94, places=2)


class Statistics(unittest.TestCase):
    """Reference values computed independently with numpy (betas, HC1 and CR1 standard errors) and from t tables."""
    X = [[0.2, 3], [0.5, 2], [0.9, 5], [1.4, 4], [2.0, 6], [2.6, 5]]
    y = [1.0, 2.1, 2.9, 4.2, 5.1, 6.3]

    def test_ols_hc1(self):
        from run_live_benchmark import ols_hc1
        b, se, p, r2 = ols_hc1(self.X, self.y)
        for got, ref in zip(b, (0.978998, 2.175514, -0.032316)):
            self.assertAlmostEqual(got, ref, places=5)
        for got, ref in zip(se, (0.291969, 0.199160, 0.078207)):
            self.assertAlmostEqual(got, ref, places=5)
        self.assertAlmostEqual(r2, 0.9871, places=4)

    def test_ols_cluster_robust(self):
        from run_live_benchmark import ols_cluster_robust
        b, se, p, G = ols_cluster_robust(self.X, self.y, ["a", "a", "b", "b", "c", "c"])
        self.assertEqual(G, 3)
        for got, ref in zip(se, (0.080510, 0.213247, 0.064538)):
            self.assertAlmostEqual(got, ref, places=5)

    def test_t_two_sided_p(self):
        from run_live_benchmark import t_two_sided_p
        self.assertAlmostEqual(t_two_sided_p(2.228, 10), 0.05, places=3)     # t table: t_{0.975, 10} = 2.228
        self.assertAlmostEqual(t_two_sided_p(2.0, 10), 0.0734, places=3)
        self.assertAlmostEqual(t_two_sided_p(1.96, 1_000_000), 0.05, places=3)
        self.assertEqual(t_two_sided_p(2.0, 0), 1.0)

    def test_block_permutation_respects_clusters(self):
        from run_live_benchmark import permutation_p
        X = [[i % 6 / 5.0] for i in range(18)]
        y = [float(i // 3) for i in range(18)]              # six clusters of three identical values
        cl = [str(i // 3) for i in range(18)]
        p = permutation_p(X, y, 1, n_perm=200, cluster_ids=cl)
        self.assertTrue(0.0 < p <= 1.0)
        # three clusters with x = 0 and three with x = 1, y rising by cluster: under block permutation the observed
        # separation recurs in 2 * 3! * 3! of 6! assignments, so the exact block p is 0.10; row permutation would call it rare
        Xb = [[0.0]] * 9 + [[1.0]] * 9
        pb = permutation_p(Xb, y, 1, n_perm=2000, cluster_ids=cl)
        pr = permutation_p(Xb, y, 1, n_perm=2000)
        self.assertAlmostEqual(pb, 0.10, delta=0.03)
        self.assertLess(pr, pb)


if __name__ == "__main__":
    unittest.main()
