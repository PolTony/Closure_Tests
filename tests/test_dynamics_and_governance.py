"""Unit tests for artifact source loading, fee pricing, forecast streams, archive labels, and client beliefs."""
import os, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from model_config import reset_config


class TestExactSourceLoading(unittest.TestCase):
    def _rewrite_same_size_same_mtime(self, path, text, stat):
        Path(path).write_text(text, encoding="utf-8")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(os.stat(path).st_size, stat.st_size)

    def test_equal_size_same_mtime_rewrite_is_not_served_from_bytecode(self):
        from compiler_agent import CompiledToolArtifact, load_compiled_callable
        src1 = "def execute_compiled_routine(a, b, c):\n    return {'v': 1}\n"
        src2 = src1.replace("1", "2")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tool_X_v3.py")
            Path(path).write_text(src1, encoding="utf-8")
            st = os.stat(path)
            # prime any bytecode cache the way the old importlib loader would have
            import importlib.util
            spec = importlib.util.spec_from_file_location("probe_prime", path)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            self._rewrite_same_size_same_mtime(path, src2, st)
            art = CompiledToolArtifact("X", "v3", path, src2, 0, 0, 0.0, 0)
            self.assertEqual(load_compiled_callable(art)("", "", "")["v"], 2)
            art_disk = CompiledToolArtifact("X", "v3", path, "", 0, 0, 0.0, 0)
            self.assertEqual(load_compiled_callable(art_disk)("", "", "")["v"], 2)

    def test_artifact_executes_its_own_source_even_if_file_reused(self):
        from compiler_agent import CompiledToolArtifact, load_compiled_callable
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.py")
            Path(path).write_text("def execute_compiled_routine(a,b,c):\n    return {'v': 9}\n", encoding="utf-8")
            art = CompiledToolArtifact("X", "v3", path, "def execute_compiled_routine(a,b,c):\n    return {'v': 3}\n", 0, 0, 0.0, 0)
            self.assertEqual(load_compiled_callable(art)("", "", "")["v"], 3)

    def test_sandbox_runner_reads_exact_source(self):
        import llm_agents
        self.assertNotIn("spec_from_file_location", llm_agents._SANDBOX_RUNNER)
        self.assertIn('"-B"', Path(llm_agents.__file__).read_text(encoding="utf-8"))


def _gated_core_spec():
    import domain_cases as dc
    return next(s for s in dc.FULL_EXTENDED_CORPUS if s.ring == "Inner_Core" and s.statutory_gate == 1)


def _agents_on(spec, pk, d):
    from compiler_agent import synthesize_compiled_tool_from_handbook, load_compiled_callable
    from negotiated_order_lab import execute_hybrid_tool_plus_critic, execute_incumbent_react_agent
    fn = load_compiled_callable(synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, pk[0].doc_handbook, d, "v1"))
    for p in pk:
        docs = (p.doc_primary_intake, p.doc_secondary_schedule, p.doc_tertiary_annex)
        yield (p, execute_hybrid_tool_plus_critic(spec, fn, *docs), execute_incumbent_react_agent(spec, p, "COLLAB_AUDIT", challenger_out=fn(*docs)),
               execute_incumbent_react_agent(spec, p, "SOLO"))


class TestFeeIsAPriceNotReviewTime(unittest.TestCase):
    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_review_time_is_fixed_whatever_fee_is_charged(self):
        from negotiated_order_lab import registered_review_fraction, review_time_share
        from model_config import set_config
        spec = _gated_core_spec()
        reg = registered_review_fraction(spec)
        for fee in (0.15, 0.25, 0.5, None):
            self.assertAlmostEqual(review_time_share(spec, fee), reg)
        set_config(review_time_mode="fee")   # the legacy confound, diagnostic only
        self.assertAlmostEqual(review_time_share(spec, 0.5), 0.5)

    def test_labour_hours_read_review_time_not_fee(self):
        from labor_market import OccupationLabor
        base = {"weight": 1.0, "poach": 0.0, "settle": 1.0, "gate": 0.0, "hard_gate": 0.0, "hard_settle": 0.5, "unit_cost": 1.5, "review_time": 0.25}
        a = OccupationLabor("X", 2.0, 0.25).step(1, [dict(base, review_fraction=0.25)])
        b = OccupationLabor("X", 2.0, 0.25).step(1, [dict(base, review_fraction=0.50)])   # a stray fee key cannot move hours
        self.assertAlmostEqual(a["I_demand"], b["I_demand"])
        c = OccupationLabor("X", 2.0, 0.25).step(1, [dict(base, review_time=0.50)])
        self.assertAlmostEqual(c["I_demand"], 2 * a["I_demand"])

    def test_excess_review_charge_is_relative_to_registered_fee(self):
        import run_dynamics as rd
        from negotiated_order_lab import registered_review_fraction
        spec = _gated_core_spec()
        rows = [(None, SimpleNamespace(selected_governance="ACCOUNTABILITY_SETTLEMENT"), 1)] * 4
        reg = registered_review_fraction(spec)
        self.assertAlmostEqual(rd.excess_review_charge(spec, rows, reg), 0.0)
        self.assertAlmostEqual(rd.excess_review_charge(spec, rows, reg + 0.1), 4 * 0.1 * spec.wage_ratio_inc_chall)


class TestForecastIndependence(unittest.TestCase):
    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_forecast_sample_is_independent_and_caller_salt_preserved(self):
        import run_dynamics as rd
        from model_config import CFG, set_config
        spec = _gated_core_spec()
        real = rd.period_packets(spec, 2, 6)
        fc = rd.period_packets(spec, 2, 6, stream="forecast")
        self.assertNotEqual([p.params for p in real], [p.params for p in fc])
        set_config(seed_salt="_UNIVERSE_A")
        a = rd.period_packets(spec, 2, 6)
        self.assertEqual(CFG.seed_salt, "_UNIVERSE_A")
        self.assertFalse(CFG.random_tier_allocation)
        self.assertNotEqual([p.params for p in a], [p.params for p in real])
        set_config(seed_salt="_UNIVERSE_B")
        b = rd.period_packets(spec, 2, 6)
        self.assertNotEqual([p.params for p in a], [p.params for p in b])
        reset_config(); set_config(dyn_universe="U7")
        self.assertNotEqual([p.params for p in rd.period_packets(spec, 2, 6)], [p.params for p in real])

    def _trace(self, mode):
        import run_dynamics as rd
        calls, pubs = [], []
        orig_pp, orig_pub = rd.period_packets, rd.publish
        def pp(spec, t, n, stream="realised"):
            calls.append((t, stream)); return orig_pp(spec, t, n, stream)
        def pub(rows, probs, t, stream="realised"):
            pubs.append((t, stream)); return orig_pub(rows, probs, t, stream)
        rd.period_packets, rd.publish = pp, pub
        try:
            spec = _gated_core_spec()
            recs, _, _ = rd.run_regime("strategic", [spec], 2, 6, 10, overrides={"dyn_forecast_mode": mode})
        finally:
            rd.period_packets, rd.publish = orig_pp, orig_pub
        return calls, pubs, recs

    def test_independent_lookahead_never_draws_realised_future(self):
        calls, pubs, recs = self._trace("independent")
        self.assertEqual(calls, [(1, "realised"), (2, "forecast"), (2, "realised"), (3, "forecast")])
        # candidate evaluation uses forecast publication draws; the archive receives one realised draw per period
        self.assertEqual([p for p in pubs if p[1] == "realised"], [(1, "realised"), (2, "realised")])
        self.assertTrue(all(r["forecast_mode"] == "independent" for r in recs))

    def test_perfect_foresight_benchmark_is_labelled(self):
        calls, _, recs = self._trace("perfect_foresight")
        self.assertIn((2, "realised"), calls[:2])
        self.assertTrue(all(r["forecast_mode"] == "perfect_foresight" for r in recs))


class TestArchiveRecords(unittest.TestCase):
    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_recorded_decision_is_not_silently_corrected(self):
        import domain_cases as dc
        import run_dynamics as rd
        from model_config import set_config
        spec = _gated_core_spec()
        pkt = dc.build_multi_doc_packets_for_task(spec, 2, split="eval")[0]
        wrong = SimpleNamespace(final_output={"diagnosis": "WRONG", "inference": "WRONG", "treatment": pkt.gt_treatment + 5}, final_output_accurate=0)
        rec, avail = rd.archive_record(pkt, wrong, 3)
        self.assertEqual((rec.label_status, rec.gt_diagnosis, avail), ("adjudicated", pkt.gt_diagnosis, 4))
        set_config(archive_label_mode="recorded")
        rec, avail = rd.archive_record(pkt, wrong, 3)
        self.assertEqual((rec.label_status, rec.label_source, rec.gt_diagnosis), ("recorded_wrong", "recorded_decision", "WRONG"))
        self.assertEqual(pkt.gt_diagnosis != "WRONG", True)   # the generated packet itself is untouched
        set_config(archive_label_mode="delayed", archive_adjudication_delay=2)
        rec, avail = rd.archive_record(pkt, wrong, 3)
        self.assertEqual((rec.label_status, avail), ("adjudicated", 6))
        set_config(archive_label_mode="noisy", archive_adjudication_rate=0.0)
        self.assertEqual(rd.archive_record(pkt, wrong, 3)[0].label_status, "recorded_wrong")
        set_config(archive_adjudication_rate=1.0)
        self.assertEqual(rd.archive_record(pkt, wrong, 3)[0].label_status, "adjudicated")


class TestClientBelief(unittest.TestCase):
    def setUp(self):
        reset_config()

    def test_observed_incumbent_error_can_reroute_work(self):
        import domain_cases as dc
        from negotiated_order_lab import negotiate_workplace_governance
        # an unlicensed core task: without a statutory penalty the client can move signed work to the challenger
        spec = next(s for s in dc.FULL_EXTENDED_CORPUS if s.ring == "Inner_Core" and s.statutory_gate == 0)
        pk = dc.build_multi_doc_packets_for_task(spec)
        moved = signed = 0
        with tempfile.TemporaryDirectory() as d:
            for p, hyb, col, solo in _agents_on(spec, pk, d):
                ref = negotiate_workplace_governance(spec, p, hyb, col, solo)
                same = negotiate_workplace_governance(spec, p, hyb, col, solo, incumbent_error_belief=0.008)
                self.assertEqual(ref.selected_governance, same.selected_governance)   # reference belief = fixed model card
                hi = negotiate_workplace_governance(spec, p, hyb, col, solo, incumbent_error_belief=0.5)
                if ref.selected_governance != "DIRECT_CHALLENGER_POACHING":
                    signed += 1
                    moved += int(hi.selected_governance != ref.selected_governance)
        self.assertGreater(signed, 0)
        self.assertGreater(moved, 0)


if __name__ == "__main__":
    unittest.main()
