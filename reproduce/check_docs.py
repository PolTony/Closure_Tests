"""Validate the Word-only submission without replacing manually edited content."""
import json
from dataclasses import fields
from pathlib import Path
import re
import sys
import unittest
from zipfile import ZipFile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model_config import ModelConfig


def main():
    errors = []
    n_fields = len(fields(ModelConfig))
    n_jobs = len(json.loads((ROOT / 'reproduce/manifest.json').read_text())['jobs'])
    n_tests = unittest.defaultTestLoader.discover(str(ROOT / 'tests')).countTestCases()
    namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    words = 0
    for name in ('PROPOSAL_STAGE1.docx', 'TECHNICAL_APPENDIX.docx'):
        p = ROOT / 'submission' / name
        try:
            with ZipFile(p) as z:
                document = ET.fromstring(z.read('word/document.xml'))
            paragraphs = [''.join(t.text or '' for t in q.iter(namespace + 't')) for q in document.iter(namespace + 'p')]
            text = ' '.join(paragraphs)
            if name == 'PROPOSAL_STAGE1.docx':
                references = [i for i, t in enumerate(paragraphs) if t.strip().lower() == 'references']
                words = len(' '.join(paragraphs[:references[-1]] if references else paragraphs).split())
            if 'editor_options:' in text or re.search(r'\\(?:beta|frac|text|times)\b', text):
                errors.append(f'{name}: editor metadata or raw LaTeX in Word')
        except (OSError, ValueError, ET.ParseError) as e:
            errors.append(f'{name}: {e}')
    for p in (ROOT / 'README.md', ROOT / 'docs/VALIDATION_PROTOCOL.md'):
        s = p.read_text()
        for pattern, actual in [(r'(\d+) (?:configuration|ModelConfig) fields', n_fields), (r'(\d+)-job', n_jobs), (r'(?<![\d.])(\d+) (?:unit )?tests\b', n_tests)]:
            for match in re.finditer(pattern, s):
                if int(match[1]) != actual:
                    errors.append(f'{p.name}: stale count {match[1]} (actual {actual})')
        for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', s):
            if not target.startswith(('http:', 'https:', '#')) and not (p.parent / target.split('#')[0]).exists():
                errors.append(f'{p.name}: missing link target {target}')
    print(f'Proposal: {words} words before references; configuration: {n_fields} fields; reproduction: {n_jobs} jobs; tests: {n_tests}.')
    if not 1500 <= words <= 2000:
        print('WORD-LIMIT NOTE: the retained proposal is outside the 1500–2000-word range; no user text was changed.')
    for error in errors:
        print('FAIL', error)
    print('DOCUMENT CHECK OK' if not errors else f'DOCUMENT CHECK FAILED ({len(errors)})')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
