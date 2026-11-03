"""Unit tests for JSON validation, transcript cache identity, provenance, task-level harm, and manifest integrity."""
import csv, json, math, os, sys, unittest
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "data_outputs")


class TestRobustJSON(unittest.TestCase):
    def setUp(self):
        import llm_agents
        self.m = llm_agents

    def test_non_objects_and_nonfinite_are_errors(self):
        p = self.m._parse_json
        self.assertEqual(p("[1, 2]")["error"], "schema")
        self.assertEqual(p("null")["error"], "schema")
        self.assertEqual(p("3.5")["error"], "schema")
        self.assertEqual(p('{"treatment": NaN}')["error"], "nonfinite")
        self.assertEqual(p('{"treatment": Infinity}')["error"], "nonfinite")
        self.assertEqual(p("no json here")["error"], "parse")
        self.assertEqual(p(None)["error"], "parse")
        self.assertEqual(p('text {"a": 1} tail'), {"a": 1})

    def test_answer_objects_never_coerced(self):
        a = self.m._answer_dict
        good = a({"diagnosis": "x", "inference": "y", "treatment": 2.0}, "z")
        self.assertEqual(good["error_class"], "")
        for bad in ({"diagnosis": "x", "inference": "y"}, {"diagnosis": "x", "inference": "y", "treatment": "abc"},
                    {"diagnosis": 3, "inference": "y", "treatment": 1}, {"diagnosis": "x", "inference": "y", "treatment": None},
                    {"error": "nonfinite"}, [1, 2]):
            r = a(bad, "z")
            self.assertNotEqual(r["error_class"], "", bad)
            self.assertTrue(math.isnan(r["treatment"]))
            self.assertTrue(r["diagnosis"].startswith("ERROR_"))


class TestCompleteConfigCacheIdentity(unittest.TestCase):
    def test_any_request_setting_changes_the_key(self):
        import llm_agents as m
        base = m.LLMConfig()
        k0 = m._key_v2("novice", "s", "u", None, m.request_fingerprint(base, "novice", None))
        for change in (dict(model="other-model"), dict(effort="low"), dict(max_tokens=123)):
            cfg = replace(base, **change)
            k = m._key_v2("novice", "s", "u", None, m.request_fingerprint(cfg, "novice", None))
            self.assertNotEqual(k0, k, change)
        for change in (dict(compiler_effort="high"), dict(compiler_max_tokens=1000), dict(compiler_task_budget=0)):
            c0 = m._key_v2("compiler_v1", "s", "u", None, m.request_fingerprint(base, "compiler_v1", None))
            c1 = m._key_v2("compiler_v1", "s", "u", None, m.request_fingerprint(replace(base, **change), "compiler_v1", None))
            self.assertNotEqual(c0, c1, change)
        fp = m.request_fingerprint(base, "novice", None)
        self.assertIn("fallbacks", fp)
        self.assertEqual(fp["key_version"], m.KEY_VERSION)


class TestSelectedRunProvenance(unittest.TestCase):
    def test_selected_run_counts(self):
        path = os.path.join(OUT, "llm_provenance_summary.csv")
        if not os.path.exists(path):
            self.skipTest("run reproduce.py --jobs llm_provenance first")
        with open(path) as f:
            rows = {r["scope"]: r for r in csv.DictReader(f)}
        s = rows["selected_reported_run"]
        self.assertEqual((int(s["n_requests"]), int(s["served_primary"]), int(s["served_other"]),
                          int(s["explicit_fallback_events"])), (774, 761, 13, 12))
        self.assertEqual(int(rows["whole_transcript_file"]["n_requests"]), 794)
        self.assertEqual(int(s["n_requests"]) + int(rows["file_records_not_requested"]["n_requests"]), 794)


class TestTaskLevelHarm(unittest.TestCase):
    def test_harm_uses_each_task_own_liability(self):
        import run_closure_formation as rc
        specs = rc.occupation_specs()
        occ = "Lawyers"
        tasks = [s for s in rc.FULL_EXTENDED_CORPUS if s.occ_title == occ and s.ring != "Outer_Periphery"]
        self.assertGreaterEqual(len(tasks), 2)
        rows = []
        # one wrong visible case on the first task only
        for i, s in enumerate(tasks):
            rows.append(dict(occ_title=occ, wave=rc.WAVES[0], ring=s.ring, task_id=s.task_id, weight="1.0",
                             accurate="0" if i == 0 else "1", silent="0", inc_hours="0", sub_hours="0",
                             labor_cost="0", arrangement="POACH"))
        got = rc.per_occupation(rows, specs)[(occ, rc.WAVES[0])]["visible_harm"]
        self.assertAlmostEqual(got, rc.task_liability(tasks[0].task_id) / len(tasks), places=9)


class TestPathResolution(unittest.TestCase):
    def test_env_home_overrides_checkout(self):
        import importlib, juri_paths
        old = os.environ.get("JURI_LAB_HOME")
        try:
            os.environ["JURI_LAB_HOME"] = "/tmp/juri_test_home"
            self.assertEqual(os.path.abspath(importlib.reload(juri_paths).ROOT), "/tmp/juri_test_home")
        finally:
            if old is None:
                os.environ.pop("JURI_LAB_HOME", None)
            else:
                os.environ["JURI_LAB_HOME"] = old
            importlib.reload(juri_paths)
        self.assertTrue(os.path.exists(juri_paths.module_path("run_live_benchmark.py")))


class TestRingSelection(unittest.TestCase):
    def test_merge_dedups_by_full_identifier_and_rings_unique(self):
        import propose_corpus as pc
        lab, add = pc.load("onet31_tasks_lab18.csv"), pc.load("onet31_tasks_proposed_additions.csv")
        merged = pc.merge_tasks(lab, add)
        keys = [(t["O*NET-SOC Code"].strip(), t["Task ID"].strip()) for t in merged]
        self.assertEqual(len(keys), len(set(keys)))
        rows = pc.propose(merged)
        pc.check_unique_rings(rows)
        self.assertEqual(len(rows), 171)
        self.assertEqual(len({r["soc2018"] for r in rows}), 57)


class TestManifestIntegrity(unittest.TestCase):
    def test_jobs_reference_real_scripts_and_dependencies(self):
        with open(os.path.join(ROOT, "reproduce", "manifest.json")) as f:
            m = json.load(f)
        ids = [j["id"] for j in m["jobs"]]
        self.assertEqual(len(ids), len(set(ids)))
        for j in m["jobs"]:
            self.assertTrue(os.path.exists(os.path.join(ROOT, j["args"][0])), j["id"])
            for d in j.get("deps", []):
                self.assertIn(d, ids, j["id"])
            # Every job has an output inventory; an empty inventory means its printed log is the artifact.
            self.assertIsInstance(j["outputs"], list, j["id"])
            self.assertTrue(j["supports"].strip(), j["id"])


class TestLadderIsNotAValidationCount(unittest.TestCase):
    def test_ladder_columns(self):
        path = os.path.join(OUT, "validation_ladder.csv")
        if not os.path.exists(path):
            self.skipTest("run reproduce.py --jobs validation_ladder first")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for col in ("evidence_class", "construct", "unit_contrast", "failure_criterion"):
            self.assertIn(col, rows[0])
        srcs = " ".join(r["source"] for r in rows)
        self.assertIn("Redbird", srcs)
        self.assertIn("Kleiner", srcs)
        self.assertFalse(any("Redbird" in r["source"] and "Kleiner" in r["source"] for r in rows))


if __name__ == "__main__":
    unittest.main()
