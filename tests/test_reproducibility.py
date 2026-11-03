"""Unit tests for validation splits, reproduction dependency ordering, provenance checks, and packaging."""
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import compiler_agent as ca
import reproduce as rp
from domain_cases import PRIMARY_CORPUS, build_multi_doc_packets_for_task, build_public_sandbox_archive
from model_config import reset_config, set_config


class TestReproducibility(unittest.TestCase):
    def tearDown(self):
        reset_config()

    def test_validation_packets_never_fit_in_either_generation(self):
        spec = next(s for s in PRIMARY_CORPUS if s.task_id == 'LAW-B2')
        archive = build_public_sandbox_archive(spec)
        held_values = {ca._case_features(ca.parse_handbook_spec(p.doc_handbook), p.doc_primary_intake,
                       p.doc_secondary_schedule, p.doc_tertiary_annex)[0]['V1']
                       for i, p in enumerate(archive) if i % 4 == 3}
        set_config(selection_validation_fraction=0.25)
        fit = ca._fit_coefs
        seen = []
        def checked(cases, *args):
            seen.extend(c[0]['V1'] for c in cases)
            self.assertTrue(held_values.isdisjoint(c[0]['V1'] for c in cases))
            return fit(cases, *args)
        with tempfile.TemporaryDirectory() as d, patch.object(ca, '_fit_coefs', side_effect=checked):
            a = ca.synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, archive[0].doc_handbook, d)
            for v in ('v2', 'v3'):
                a = ca.synthesize_compiled_tool_from_handbook(spec.task_id, spec.occ_title, spec.ring, archive[0].doc_handbook,
                                                            d, v, sandbox=archive, previous=a)
        self.assertTrue(seen)

    def test_rerun_waits_for_previously_successful_dependency(self):
        jobs = [dict(id='a', args=['a.py'], deps=[], outputs=[], supports='printed'),
                dict(id='b', args=['b.py'], deps=['a'], outputs=[], supports='printed')]
        finished = threading.Event()
        order = []
        def run(j):
            order.append(j['id'])
            if j['id'] == 'a':
                finished.set()
            else:
                self.assertTrue(finished.is_set())
            return dict(ok=True, seconds=0, outputs={})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'run.json')
            Path(path).write_text(json.dumps({'jobs': {'a': {'ok': True}, 'b': {'ok': True}}}))
            with patch.object(rp, 'RUN_MANIFEST', path), patch.object(rp, 'OUT', d), patch.object(rp, 'load_manifest', return_value={'jobs': jobs}), \
                 patch.object(rp, 'code_hashes', return_value={}), patch.object(rp, 'run_job', side_effect=run), \
                 patch.object(sys, 'argv', ['reproduce.py', '--all', '--parallel', '2']), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as e:
                    rp.main()
                self.assertEqual(e.exception.code, 0)
        self.assertEqual(order, ['a', 'b'])

    def test_changed_or_added_source_invalidates_record(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'log.txt'; log.write_text('table')
            job = dict(id='x', args=['x.py'], outputs=[], deps=[])
            rec = dict(ok=True, args=['x.py'], outputs={}, inputs={'x.py': 'old'}, log=str(log), log_sha256=rp.sha256(log))
            with patch.object(rp, 'input_hashes', return_value={'x.py': 'new', 'added.py': 'hash'}):
                self.assertIn('source, inputs or dependency artifacts changed', rp.record_errors(job, rec))

    def test_missing_declared_figure_invalidates_record(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'log.txt'; log.write_text('figure')
            job = dict(id='figures', args=['make_figures.py'], outputs=['missing.png'], output_root='project', deps=[])
            rec = dict(ok=True, args=job['args'], output_root='project', outputs={'missing.png': 'hash'}, inputs={},
                       log=str(log), log_sha256=rp.sha256(log))
            with patch.object(rp, 'BASE', d), patch.object(rp, 'input_hashes', return_value={}):
                self.assertTrue(any('missing.png' in x for x in rp.record_errors(job, rec)))

    def test_replaced_juniors_get_no_settlement_registry_credit(self):
        from types import SimpleNamespace
        from run_dynamics import credit_exposure
        from profession import OccupationKnowledge
        pkt = SimpleNamespace(complexity_tier=3)
        st = SimpleNamespace(selected_governance='ACCOUNTABILITY_SETTLEMENT',
                             notes={'collab_mode': 'COLLAB_OVERRIDE_DEEP_AUDIT', 'review_mask': {'registry': True}, 'esc_rev': False})
        pr = OccupationKnowledge('test')
        set_config(labor_junior_operates_tool=False)
        credit_exposure(pr, pkt, st, 100)
        self.assertEqual(pr.credits['juniors']['registry'], 0)
        set_config(labor_junior_operates_tool=True)
        credit_exposure(pr, pkt, st, 100)
        self.assertGreater(pr.credits['juniors']['registry'], 0)

    def test_all_reproduction_entry_points_are_packaged(self):
        import re
        root = Path(__file__).resolve().parents[1]
        block = re.search(r'py-modules = \[([\s\S]*?)\]', (root / 'pyproject.toml').read_text())[1]
        modules = re.findall(r'"([^"]+)"', block)
        for job in json.loads((root / 'reproduce/manifest.json').read_text())['jobs']:
            if job['args'][0].startswith('src/'):
                self.assertIn(Path(job['args'][0]).stem, modules)


if __name__ == '__main__':
    unittest.main()
