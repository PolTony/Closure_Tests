#!/usr/bin/env python3
"""
Module 2: Specification-Driven Task Compiler with Closed-Loop Inductive Repair (compiler_agent.py)
JURI Lab — executed architecture

What the compiler can see: the PUBLIC handbook text of a task and the PUBLIC resolved-case archive
(labelled Tier 0-2 packets). It never imports the exact solver and never sees the private registry.

Generation v1 (Turn 1, specification-driven compilation): a small compiler for the handbook's rule
  language. It parses DOCUMENT_FIELDS, DIAGNOSTIC_CUTOFFS, LET, DIAGNOSE, RULE and CHECK statements,
  builds expression ASTs with a recursive-descent parser, and emits a standalone Python module exposing
  `execute_compiled_routine(doc_intake, doc_sched, doc_annex)`.
Generation v2 (Turn 2, unit-test repair, linear hypothesis class): runs v1 on the archive, collects the
  assertion failures, finds the annex key that is present in every failure and absent from every pass,
  and searches a grammar of four exception-handler templates (TEMPLATES below; B a handbook-defined
  quantity or V1, S the discriminating annex feature, X,Y raw document fields, coefficients on a
  CFG.coef_decimals grid) by exact k-case solves scored by consensus over all failures. Ties are broken
  by the number of free coefficients (Occam), then by template order. The winning handler is inserted
  ahead of the routine rules and the winning template is recorded for reporting.
Generation v3 (Turn 3, piecewise repair): runs v2 on the archive, takes the residual failures of the
  learned handler, searches a one-split decision stump over {V1, V2, V3, V4, S} that separates residuals
  from passes, refits the handler on the residual side, and emits a bracketed piecewise handler.

Every generation writes `tool_<task>_<version>.py` and reports AST node count, McCabe cyclomatic
complexity, Halstead volume and maximum branch depth of the emitted source.
"""

import ast
import copy
import itertools
import math
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from model_config import CFG


@dataclass
class CompiledToolArtifact:
    task_id: str
    version: str               # "v1", "v2", or "v3"
    filepath: str
    source_code: str
    ast_node_count: int
    cyclomatic_complexity: int
    halstead_volume: float
    max_branch_depth: int
    ir: Dict[str, object] = field(default_factory=dict, repr=False)
    repair_log: List[str] = field(default_factory=list)
    # the same metrics on the program with the handbook's validation-flag scaffold removed. The scaffold
    # (CHECK ... FLAG lines) never changes diagnosis, inference or treatment, and on registered draws it never fires.
    ast_node_count_active: int = 0
    halstead_volume_active: float = 0.0
    n_validation_checks: int = 0


# ---------------------------------------------------------------------------
# 1. Expression language front-end: tokenizer, recursive-descent parser, emitter, evaluator
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\s*(?:(\d+\.\d+|\d+)|([A-Za-z_][A-Za-z0-9_.]*)|(\*\*|>=|<=|==|[-+*/(),<>]))")
_KEYWORDS = {"IF", "ELSE"}
_NUM_RE = r"([0-9]+(?:\.[0-9]+)?)"
_DOC_MAP = {"DOC1": "doc_intake", "DOC2": "doc_sched", "DOC3": "doc_annex"}


def tokenize(text: str) -> List[Tuple[str, str]]:
    toks: List[Tuple[str, str]] = []
    text = text.strip()
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise SyntaxError(f"cannot tokenize {text[pos:pos + 20]!r}")
        if m.group(1) is not None:
            toks.append(("num", m.group(1)))
        elif m.group(2) is not None:
            w = m.group(2)
            toks.append(("kw", w) if w in _KEYWORDS else ("id", w))
        else:
            toks.append(("op", m.group(3)))
        pos = m.end()
    return toks


class _Parser:
    """Precedence-climbing parser. AST nodes are tuples:
    ("num", text) | ("id", name) | ("neg", x) | ("bin", op, l, r) | ("cmp", op, l, r)
    | ("if", body, test, orelse) | ("call", fname, [args])"""

    def __init__(self, text: str):
        self.toks = tokenize(text)
        self.i = 0

    def peek(self) -> Tuple[Optional[str], Optional[str]]:
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self, kind: Optional[str] = None, val: Optional[str] = None) -> Tuple[str, str]:
        t = self.peek()
        if t[0] is None or (kind and t[0] != kind) or (val and t[1] != val):
            raise SyntaxError(f"unexpected token {t}, expected {kind} {val}")
        self.i += 1
        return t  # type: ignore[return-value]

    def parse(self):
        node = self.ternary()
        if self.i != len(self.toks):
            raise SyntaxError(f"trailing tokens: {self.toks[self.i:]}")
        return node

    def ternary(self):
        body = self.comparison()
        if self.peek() == ("kw", "IF"):
            self.take()
            test = self.comparison()
            self.take("kw", "ELSE")
            orelse = self.ternary()
            return ("if", body, test, orelse)
        return body

    def comparison(self):
        left = self.additive()
        t = self.peek()
        if t[0] == "op" and t[1] in (">=", "<=", ">", "<", "=="):
            op = self.take()[1]
            return ("cmp", op, left, self.additive())
        return left

    def additive(self):
        node = self.term()
        while self.peek()[0] == "op" and self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            node = ("bin", op, node, self.term())
        return node

    def term(self):
        node = self.power()
        while self.peek()[0] == "op" and self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            node = ("bin", op, node, self.power())
        return node

    def power(self):
        base = self.unary()
        if self.peek() == ("op", "**"):
            self.take()
            return ("bin", "**", base, self.power())
        return base

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return ("neg", self.unary())
        return self.primary()

    def primary(self):
        t = self.peek()
        if t[0] == "num":
            self.take()
            return ("num", t[1])
        if t[0] == "id":
            self.take()
            if self.peek() == ("op", "("):
                self.take()
                args = []
                if self.peek() != ("op", ")"):
                    args.append(self.ternary())
                    while self.peek() == ("op", ","):
                        self.take()
                        args.append(self.ternary())
                self.take("op", ")")
                return ("call", t[1], args)
            return ("id", t[1])
        if t == ("op", "("):
            self.take()
            node = self.ternary()
            self.take("op", ")")
            return node
        raise SyntaxError(f"unexpected token {t}")


def parse_expr(text: str):
    return _Parser(text).parse()


def emit_py(node, names: Dict[str, str]) -> str:
    """Emits fully parenthesised Python for an expression AST (preserves evaluation order exactly)."""
    k = node[0]
    if k == "num":
        return node[1]
    if k == "id":
        if node[1] not in names:
            raise KeyError(f"unbound identifier in handbook expression: {node[1]}")
        return names[node[1]]
    if k == "neg":
        return f"(-{emit_py(node[1], names)})"
    if k in ("bin", "cmp"):
        return f"({emit_py(node[2], names)} {node[1]} {emit_py(node[3], names)})"
    if k == "if":
        return f"({emit_py(node[1], names)} if {emit_py(node[2], names)} else {emit_py(node[3], names)})"
    if k == "call":
        fn = {"MIN": "min", "MAX": "max"}[node[1]]
        return f"{fn}({', '.join(emit_py(a, names) for a in node[2])})"
    raise ValueError(k)


def eval_node(node, env: Dict[str, float]) -> float:
    """Evaluates an expression AST (used by the repair loop to compute handbook quantities per case)."""
    k = node[0]
    if k == "num":
        return float(node[1])
    if k == "id":
        return float(env[node[1]])
    if k == "neg":
        return -eval_node(node[1], env)
    if k == "bin":
        l, r, op = eval_node(node[2], env), eval_node(node[3], env), node[1]
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            return l / r
        return l ** r
    if k == "cmp":
        l, r, op = eval_node(node[2], env), eval_node(node[3], env), node[1]
        return float({">=": l >= r, "<=": l <= r, ">": l > r, "<": l < r, "==": l == r}[op])
    if k == "if":
        return eval_node(node[1], env) if eval_node(node[2], env) else eval_node(node[3], env)
    if k == "call":
        f = min if node[1] == "MIN" else max
        return float(f(*[eval_node(a, env) for a in node[2]]))
    raise ValueError(k)


# ---------------------------------------------------------------------------
# 2. Handbook front-end -> intermediate representation (IR)
# ---------------------------------------------------------------------------
def parse_handbook_spec(doc_handbook: str) -> Dict[str, object]:
    """Parses the raw text of a public handbook into the compiler IR."""
    ir: Dict[str, object] = {"family": "UNKNOWN", "job_zone": 4, "overlap_j": 0.5, "fields": {}, "cutoffs": {},
                             "lets": [], "diags": [], "rules": [], "checks": [], "learned": []}
    for raw in doc_handbook.splitlines():
        line = raw.strip()
        if line.startswith("OCCUPATION:"):
            m = re.search(r"JOB_ZONE:\s*(\d+)", line)
            if m:
                ir["job_zone"] = int(m.group(1))
        elif line.startswith("DOMAIN_FAMILY:"):
            m = re.search(r"DOMAIN_FAMILY:\s*([A-Z_]+)", line)
            if m:
                ir["family"] = m.group(1)
            m = re.search(r"OVERLAP_J:\s*([0-9.]+)", line)
            if m:
                ir["overlap_j"] = float(m.group(1))
        elif line.startswith("DIAGNOSTIC_CUTOFFS:"):
            for k, v in re.findall(r"([A-Z_]+)=([0-9.]+)", line):
                ir["cutoffs"][k] = float(v)
        elif line.startswith("DOCUMENT_FIELDS:"):
            for var, doc, key in re.findall(r"([A-Z][A-Z0-9_]*)\s*:=\s*(DOC\d)\.([A-Za-z0-9_]+)", line):
                ir["fields"][var] = (_DOC_MAP[doc], key)
        elif line.startswith("LET "):
            m = re.match(r"LET\s+([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*(.+)$", line)
            ir["lets"].append((m.group(1), parse_expr(m.group(2))))
        elif line.startswith("DIAGNOSE "):
            m = re.match(r"DIAGNOSE\s+([A-Z_]+)\s+IF\s+(.+?)\s+ELSE\s+([A-Z_]+)$", line)
            ir["diags"].append((m.group(1), parse_expr(m.group(2)), m.group(3)))
        elif line.startswith("RULE "):
            m = re.match(r"RULE\s+(\w+)\s+(?:WHEN\s+(.+?)|(OTHERWISE))\s*->\s*INFER\s+([A-Z][A-Z0-9_]*)\s*;\s*TREAT\s+(.+)$", line)
            name, cond, other, label, treat = m.groups()
            rule: Dict[str, object] = {"name": name, "label": label, "treat": parse_expr(treat)}
            if other:
                rule["kind"] = "otherwise"
            else:
                fm = re.match(r"(DOC\d)\.([A-Za-z0-9_]+)\s+IS\s+TRUE$", cond)
                if fm:
                    rule["kind"] = "flag"
                    rule["doc"] = _DOC_MAP[fm.group(1)]
                    rule["flag"] = fm.group(2)
                else:
                    rule["kind"] = "expr"
                    rule["cond"] = parse_expr(cond)
            ir["rules"].append(rule)
        elif line.startswith("CHECK "):
            m = re.match(r"CHECK\s+(\w+)\s+FLAG\s+([A-Z][A-Z0-9_]*)\s+WHEN\s+(.+)$", line)
            ir["checks"].append((m.group(2), parse_expr(m.group(3))))
    return ir


# ---------------------------------------------------------------------------
# 3. Back-end: IR -> standalone Python source
# ---------------------------------------------------------------------------
# Handler template grammar. Each template is T = offset + sum_j c_j * phi_j; `k` free coefficients.
# `render` returns Python source using c0..c{k-1}; `sv` is the local holding the annex feature S.
TEMPLATES: Dict[str, Dict[str, object]] = {
    "G3_mult_severity_one_field": {
        "k": 2, "needs_y": False,
        "offset": lambda e, B, S, X, Y: e[B],
        "phi": lambda e, B, S, X, Y: [-e[B] * S, e[X]],
        "render": lambda nm, B, sv, X, Y: f"{nm[B]} * (1 - c0 * {sv}) + c1 * {nm[X]}",
    },
    "G1_mult_severity": {
        "k": 3, "needs_y": True,
        "offset": lambda e, B, S, X, Y: e[B],
        "phi": lambda e, B, S, X, Y: [-e[B] * S, e[X], e[Y]],
        "render": lambda nm, B, sv, X, Y: f"{nm[B]} * (1 - c0 * {sv}) + c1 * {nm[X]} + c2 * {nm[Y]}",
    },
    "G2_add_severity": {
        "k": 3, "needs_y": True,
        "offset": lambda e, B, S, X, Y: e[B],
        "phi": lambda e, B, S, X, Y: [S, e[X], e[Y]],
        "render": lambda nm, B, sv, X, Y: f"{nm[B]} + c0 * {sv} + c1 * {nm[X]} + c2 * {nm[Y]}",
    },
    "G5_mult_severity_interaction": {
        "k": 3, "needs_y": False, "extended_only": True,
        "offset": lambda e, B, S, X, Y: e[B],
        "phi": lambda e, B, S, X, Y: [-e[B] * S, -e[B] * S * e["V2"], e[X]],
        "render": lambda nm, B, sv, X, Y: f"{nm[B]} * (1 - c0 * {sv} - c1 * {sv} * v2) + c2 * {nm[X]}",
    },
    "G4_free_base": {
        "k": 4, "needs_y": True,
        "offset": lambda e, B, S, X, Y: 0.0,
        "phi": lambda e, B, S, X, Y: [e[B], e[B] * S, e[X], e[Y]],
        "render": lambda nm, B, sv, X, Y: f"c0 * {nm[B]} + c1 * {nm[B]} * {sv} + c2 * {nm[X]} + c3 * {nm[Y]}",
    },
}


def _name_map(ir: Dict[str, object]) -> Dict[str, str]:
    names = {"V1": "v1", "V2": "v2", "V3": "v3", "V4": "v4", "ANNEX_SEVERITY": "annex_severity"}
    for k, v in ir["cutoffs"].items():
        names[k] = repr(float(v))
    for n, _ in ir["lets"]:
        names[n] = n
    return names


def emit_source(ir: Dict[str, object], task_id: str, occ_title: str, ring: str, version: str) -> str:
    names = _name_map(ir)
    n_learned = len(ir["learned"])
    L: List[str] = []
    L.append("#!/usr/bin/env python3")
    L.append('"""')
    L.append(f"Auto-Compiled Occupational Task Routine: {task_id} ({occ_title} - {ring})")
    L.append(f"Domain Engine: {ir['family']} | Compiler Generation: {version}")
    prov = (f"Provenance: public handbook ({len(ir['lets'])} LET, {len(ir['diags'])} DIAGNOSE, "
            f"{len(ir['rules'])} RULE, {len(ir['checks'])} CHECK statements)")
    if n_learned:
        prov += f" + {n_learned} exception handler(s) induced from the public resolved-case archive"
    L.append(prov)
    L.append('"""')
    L.append("import re")
    L.append("from typing import Dict")
    L.append("")
    L.append('_NUM = r"([0-9]+(?:\\.[0-9]+)?)"')
    L.append('_KV = r"([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*[0-9]+(?:\\.[0-9]+)?"')
    L.append("")
    L.append("")
    L.append("def _field(doc: str, key: str):")
    L.append('    m = re.search(re.escape(key) + r"\\s*[:=]\\s*" + _NUM, doc)')
    L.append("    return float(m.group(1)) if m else None")
    L.append("")
    L.append("")
    L.append("def _out(diagnosis, inference, treatment, flags, fired, annex_fields, handled):")
    L.append('    return {"diagnosis": diagnosis, "inference": inference, "treatment": round(treatment, 2),')
    L.append('            "validation_flags": flags, "fired": fired, "unhandled_fields": sorted(set(annex_fields) - set(handled))}')
    L.append("")
    L.append("")
    L.append("def execute_compiled_routine(doc_intake: str, doc_sched: str, doc_annex: str) -> Dict[str, object]:")
    req = [v for v in ("V1", "V2", "V3", "V4") if v in ir["fields"]]
    for var in req:
        doc, key = ir["fields"][var]
        L.append(f'    {names[var]} = _field({doc}, "{key}")')
    L.append("    if " + " or ".join(f"{names[v]} is None" for v in req) + ":")
    L.append('        return {"diagnosis": "PARSE_ERROR", "inference": "PARSE_ERROR", "treatment": 0.0, "validation_flags": [], "fired": "NONE", "unhandled_fields": []}')
    L.append("    validation_flags = []")
    L.append("    annex_fields = re.findall(_KV, doc_annex)")
    L.append("    handled_fields = []")
    for n, node in ir["lets"]:
        L.append(f"    {n} = {emit_py(node, names)}")
    for flag, cond in ir["checks"]:
        L.append(f"    if {emit_py(cond, names)}:")
        L.append(f'        validation_flags.append("{flag}")')
    for i, (tl, cond, fl) in enumerate(ir["diags"]):
        L.append(f'    diag_{i} = "{tl}" if {emit_py(cond, names)} else "{fl}"')
    L.append('    diagnosis = "__".join([' + ", ".join(f"diag_{i}" for i in range(len(ir["diags"]))) + "])")

    def ret(label: str, expr: str, indent: str, fired: str) -> str:
        return f'{indent}return _out(diagnosis, "{label}", {expr}, validation_flags, "{fired}", annex_fields, handled_fields)'

    # (a) explicit handbook override rules
    for r in ir["rules"]:
        if r["kind"] != "flag":
            continue
        L.append(f'    if re.search(r"{r["flag"]}\\s*=\\s*TRUE", {r["doc"]}):')
        if "ANNEX_SEVERITY" in ir["fields"]:
            doc, key = ir["fields"]["ANNEX_SEVERITY"]
            L.append(f'        annex_severity = _field({doc}, "{key}")')
            L.append(f'        handled_fields.append("{key}")')
        L.append(ret(r["label"], emit_py(r["treat"], names), "        ", r["name"]))
    # (b) exception handlers induced from the archive (v2 linear, v3 piecewise)
    for hi, h in enumerate(ir["learned"]):
        sv = f'{h["key"]}_val'
        tpl = TEMPLATES[h["template"]]
        cvars = ", ".join(f"c{j}" for j in range(tpl["k"]))
        L.append(f'    {sv} = _field(doc_annex, "{h["key"]}")')
        L.append(f"    if {sv} is not None:")
        L.append(f'        handled_fields.append("{h["key"]}")')
        if h.get("split") is None:
            L.append(f'        {cvars} = {tuple(h["coefs"])!r}')
        else:
            sp = h["split"]
            fexpr = sv if sp["feature"] == "S" else names[sp["feature"]]
            L.append(f'        {cvars} = {tuple(sp["ge_coefs"])!r} if {fexpr} >= {sp["thr"]!r} else {tuple(sp["lt_coefs"])!r}')
        expr = tpl["render"](names, h["base"], sv, h["x"], h["y"])
        L.append(ret(h["label"], expr, "        ", f"H{hi + 1}"))
    # (c) routine rules
    for r in ir["rules"]:
        if r["kind"] == "expr":
            L.append(f'    if {emit_py(r["cond"], names)}:')
            L.append(ret(r["label"], emit_py(r["treat"], names), "        ", r["name"]))
        elif r["kind"] == "otherwise":
            L.append(ret(r["label"], emit_py(r["treat"], names), "    ", r["name"]))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 4. Software-engineering complexity metrics on the emitted source
# ---------------------------------------------------------------------------
def compute_ast_and_halstead_metrics(source_code: str) -> Tuple[int, int, float, int]:
    """AST node count, McCabe cyclomatic complexity, Halstead volume V = (N1+N2) log2(eta1+eta2), max branch depth."""
    tree = ast.parse(source_code)
    node_count = sum(1 for _ in ast.walk(tree))
    branch_types = (ast.If, ast.For, ast.While, ast.IfExp, ast.ExceptHandler, ast.With)
    op_types = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.And, ast.Or, ast.Not, ast.USub, ast.Assign, ast.AugAssign, ast.Return, ast.Call
    )
    complexity = 1
    operators: List[str] = []
    operands: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, branch_types):
            complexity += 1
            operators.append(type(node).__name__)
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
            operators.append(type(node.op).__name__)
        elif isinstance(node, op_types):
            operators.append(type(node).__name__)
        elif isinstance(node, ast.Name):
            operands.append(node.id)
        elif isinstance(node, ast.Constant):
            operands.append(repr(node.value))
    n1, n2 = len(operators), len(operands)
    vocab = max(2, len(set(operators)) + len(set(operands)))
    halstead_v = round((n1 + n2) * math.log2(vocab), 1)

    def _depth(n: ast.AST, current: int = 0) -> int:
        next_d = current + (1 if isinstance(n, branch_types) else 0)
        child_depths = [_depth(c, next_d) for c in ast.iter_child_nodes(n)]
        return max(child_depths) if child_depths else next_d

    return node_count, complexity, halstead_v, _depth(tree, 0)


# ---------------------------------------------------------------------------
# 5. Closed-loop inductive repair (v2 linear handler, v3 piecewise refinement)
# ---------------------------------------------------------------------------
def _case_features(ir: Dict[str, object], doc_intake: str, doc_sched: str, doc_annex: str):
    """Document fields, handbook LET quantities and annex key=value pairs for one archived case."""
    docs = {"doc_intake": doc_intake, "doc_sched": doc_sched, "doc_annex": doc_annex}
    env: Dict[str, float] = {}
    for var in ("V1", "V2", "V3", "V4"):
        doc, key = ir["fields"][var]
        m = re.search(re.escape(key) + r"\s*[:=]\s*" + _NUM_RE, docs[doc])
        if not m:
            return None, {}
        env[var] = float(m.group(1))
    for k, v in ir["cutoffs"].items():
        env[k] = float(v)
    for n, node in ir["lets"]:
        env[n] = eval_node(node, env)
    kv = {k: float(v) for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*" + _NUM_RE, doc_annex)}
    return env, kv


def _solve_k(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Gaussian elimination with partial pivoting for a k x k exact solve; None if singular."""
    k = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-9:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, k + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][k] / M[i][i] for i in range(k)]


def _predict(tpl: Dict[str, object], env: Dict[str, float], s: float, B: str, X: str, Y: str, coefs) -> float:
    phi = tpl["phi"](env, B, s, X, Y)
    return round(tpl["offset"](env, B, s, X, Y) + sum(c * f for c, f in zip(coefs, phi)), 2)


def _fit_coefs(cases, tpl: Dict[str, object], B: str, X: str, Y: str) -> Tuple[int, Optional[Tuple[float, ...]]]:
    """Exact k-case solves of T = offset + sum c_j phi_j on the coefficient grid, scored by consensus."""
    k = tpl["k"]
    best: Tuple[int, Optional[Tuple[float, ...]]] = (0, None)
    sample = cases[:CFG.triple_sample]
    if len(sample) < k:
        return best
    for combo in itertools.combinations(range(len(sample)), k):
        A = [tpl["phi"](sample[i][0], B, sample[i][1], X, Y) for i in combo]
        rhs = [sample[i][2] - tpl["offset"](sample[i][0], B, sample[i][1], X, Y) for i in combo]
        sol = _solve_k(A, rhs)
        if sol is None:
            continue
        coefs = tuple(round(v, CFG.coef_decimals) for v in sol)
        cons = sum(1 for env, sv, t in cases if abs(_predict(tpl, env, sv, B, X, Y, coefs) - t) <= 0.01)
        if cons > best[0]:
            best = (cons, coefs)
        if cons == len(cases):
            break
    return best


def required_consensus(k: int) -> int:
    """Additional-case consensus requirement: a hypothesis with k free coefficients must reproduce at least
    max(min_consensus, k + 1) archived outcomes, i.e. at least one case beyond the k it was solved on. This
    rules out exact interpolation of the fitting sample. It is not a validation split: candidates are solved on
    and scored against the same archive. A true split is available with selection_validation_fraction > 0;
    the fresh evaluation packets remain disjoint from the archive in both cases."""
    return max(CFG.min_consensus, k + 1)


# Templates algebraically able to represent the conventions' true G1 form B(1 - aS) + bX + cY. G4 contains it
# (c0 = 1, c1 = -a, c2 = b, c3 = c), so removing G1 alone is a change of parameterization, not of family.
G1_EQUIVALENT_TEMPLATES = ("G1_mult_severity", "G4_free_base")


def grammar_templates(grammar: str) -> List[str]:
    """Template names searched under a grammar setting, ordered by number of free coefficients.
    "full": G1-G4. "full+G5": adds the interaction template. "G1": G1 only. "no-G1-reparam" (legacy name
    "no-G1"): removes G1 but keeps G4, which nests G1. "no-G1-family": removes every template that can represent
    G1 (G1 and G4), a true functional-family ablation."""
    base = [t for t in TEMPLATES if not TEMPLATES[t].get("extended_only")]
    if grammar == "G1":
        names = ["G1_mult_severity"]
    elif grammar in ("no-G1", "no-G1-reparam"):
        names = [t for t in base if t != "G1_mult_severity"]
    elif grammar == "no-G1-family":
        names = [t for t in base if t not in G1_EQUIVALENT_TEMPLATES]
    elif grammar == "full+G5":
        names = list(TEMPLATES)
    elif grammar == "full":
        names = base
    else:
        raise ValueError(f"unknown grammar {grammar}")
    return sorted(names, key=lambda t: TEMPLATES[t]["k"])


def _search_handler_split(ir: Dict[str, object], train, bases, val):
    """Training/validation selection on the fixed archive partition. Coefficients are solved and the additional-case consensus is checked on the training cases only;
    candidates are ranked by validation agreement, then training consensus, then fewer coefficients. A candidate
    must reproduce at least half of the validation cases. Fresh evaluation packets remain the final test set."""
    if not val or not train:
        return None
    best, best_key = None, None
    for tname in grammar_templates(CFG.grammar):
        tpl = TEMPLATES[tname]
        if len(train) < required_consensus(tpl["k"]):
            continue
        pairs = ([(x, None) for x in ("V1", "V2", "V3", "V4")] if not tpl["needs_y"]
                 else list(itertools.combinations(("V1", "V2", "V3", "V4"), 2)))
        for B in bases:
            for X, Y in pairs:
                cons, coefs = _fit_coefs(train, tpl, B, X, Y)
                if coefs is None or cons < required_consensus(tpl["k"]):
                    continue
                vcons = sum(1 for env, sv, t in val if abs(_predict(tpl, env, sv, B, X, Y, coefs) - t) <= 0.01)
                if 2 * vcons < len(val):
                    continue
                key = (vcons, cons, -tpl["k"])
                if best_key is None or key > best_key:
                    best_key, best = key, (cons + vcons, tname, B, X, Y, coefs)
    return best


def _search_handler(ir: Dict[str, object], cases, validation_cases=None):
    """Searches template x base x field pairs; returns (consensus, template, B, X, Y, coefs) or None.
    Candidates below their template's required consensus are discarded. Ties are broken by fewer free
    coefficients (Occam), then by template order."""
    bases = [n for n, _ in ir["lets"]] + ["V1"]
    if CFG.selection_validation_fraction > 0:
        return _search_handler_split(ir, cases, bases, validation_cases or [])
    names = grammar_templates(CFG.grammar)
    best = None
    for tname in names:
        tpl = TEMPLATES[tname]
        if best is not None and best[0] == len(cases) and tpl["k"] > TEMPLATES[best[1]]["k"]:
            continue  # a simpler template already reproduces every case
        if len(cases) < required_consensus(tpl["k"]):
            continue  # too few archived cases to validate a template of this size
        pairs = ([(x, None) for x in ("V1", "V2", "V3", "V4")] if not tpl["needs_y"]
                 else list(itertools.combinations(("V1", "V2", "V3", "V4"), 2)))
        for B in bases:
            for X, Y in pairs:
                cons, coefs = _fit_coefs(cases, tpl, B, X, Y)
                if coefs is None or cons < required_consensus(tpl["k"]):
                    continue
                cand = (cons, tname, B, X, Y, coefs)
                if best is None or cons > best[0] or (cons == best[0] and tpl["k"] < TEMPLATES[best[1]]["k"]):
                    best = cand
    return best


def _evaluate_on_archive(fn: Callable, sandbox) -> List[Tuple[object, Dict[str, object], bool]]:
    res = []
    for pkt in sandbox:
        out = fn(pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex)
        ok = (out["diagnosis"] == pkt.gt_diagnosis and out["inference"] == pkt.gt_inference
              and abs(float(out["treatment"]) - pkt.gt_treatment) <= 0.01)
        res.append((pkt, out, ok))
    return res


def _write_artifact(ir, task_id, occ_title, ring, version, output_dir, log) -> CompiledToolArtifact:
    os.makedirs(output_dir, exist_ok=True)
    ir = copy.deepcopy(ir)
    ir["version"] = version
    src = emit_source(ir, task_id, occ_title, ring, version)
    path = os.path.join(output_dir, f"tool_{task_id.replace('-', '_')}_{version}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    n, c, h, d = compute_ast_and_halstead_metrics(src)
    ir_active = dict(ir, checks=[])
    na, _, ha, _ = compute_ast_and_halstead_metrics(emit_source(ir_active, task_id, occ_title, ring, version))
    return CompiledToolArtifact(task_id, version, path, src, n, c, h, d, ir=ir, repair_log=list(log),
                                ast_node_count_active=na, halstead_volume_active=ha, n_validation_checks=len(ir.get("checks", [])))


def repair_generation(previous: CompiledToolArtifact, sandbox, version: str, task_id: str, occ_title: str,
                      ring: str, output_dir: str) -> CompiledToolArtifact:
    ir = copy.deepcopy(previous.ir)
    log = list(previous.repair_log)
    fn = load_compiled_callable(previous)
    # Fix the partition before using any outcome to select keys, labels, coefficients or a split.
    # The same packets stay in validation in both v2 and v3.
    validation = []
    if CFG.selection_validation_fraction > 0:
        m = max(2, round(1.0 / CFG.selection_validation_fraction))
        validation = [p for i, p in enumerate(sandbox) if i % m == m - 1]
        sandbox = [p for i, p in enumerate(sandbox) if i % m != m - 1]
    results = _evaluate_on_archive(fn, sandbox)
    val_features = [(p, *_case_features(ir, p.doc_primary_intake, p.doc_secondary_schedule, p.doc_tertiary_annex))
                    for p in validation]
    # keyed by object identity: archives may hold packets from several universes that share an id string
    feats = {id(pkt): _case_features(ir, pkt.doc_primary_intake, pkt.doc_secondary_schedule, pkt.doc_tertiary_annex)
             for pkt, _, _ in results}
    fails = [(p, o) for p, o, ok in results if not ok]
    n_fail, n_arch = len(fails), len(sandbox)

    if n_fail == 0:
        log.append(f"{version}: 0/{n_arch} assertion failures on the public archive; no repair needed")
        return _write_artifact(ir, task_id, occ_title, ring, version, output_dir, log)

    if version == "v2":
        pass_keys = set()
        for p, _, ok in results:
            if ok:
                pass_keys |= set(feats[id(p)][1].keys())
        key_counts: Dict[str, int] = {}
        for p, _ in fails:
            for kk in feats[id(p)][1].keys():
                if kk not in pass_keys:
                    key_counts[kk] = key_counts.get(kk, 0) + 1
        cands = [kk for kk, c in sorted(key_counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= 2]
        if not cands:
            log.append(f"v2: {n_fail}/{n_arch} failures but no annex key discriminates failures from passes; no repair")
            return _write_artifact(ir, task_id, occ_title, ring, version, output_dir, log)
        for key in cands:
            sub = [(p, o) for p, o in fails if key in feats[id(p)][1]]
            label_counts: Dict[str, int] = {}
            for p, _ in sub:
                label_counts[p.gt_inference] = label_counts.get(p.gt_inference, 0) + 1
            label = max(label_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            cases = [(feats[id(p)][0], feats[id(p)][1][key], p.gt_treatment)
                     for p, _ in sub if p.gt_inference == label and feats[id(p)][0] is not None]
            val_cases = [(env, kv[key], p.gt_treatment) for p, env, kv in val_features
                         if env is not None and key in kv and p.gt_inference == label]
            best = _search_handler(ir, cases, val_cases) if len(cases) >= 2 else None
            if best is None:
                log.append(f"v2: key='{key}': {len(cases)} labelled failures; no template reached its required consensus; no handler")
                continue
            cons, tname, B, X, Y, coefs = best
            ir["learned"].append({"key": key, "label": label, "template": tname, "base": B, "x": X, "y": Y,
                                  "coefs": list(coefs), "split": None})
            log.append(f"v2: {n_fail}/{n_arch} failures; annex key='{key}'; template {tname} with B={B}, X={X}, Y={Y}, "
                       f"coefs={coefs}, consensus {cons}/{len(cases)}")
    elif version == "v3":
        if not ir["learned"]:
            log.append(f"v3: {n_fail}/{n_arch} residual failures but no learned handler to refine; no repair")
            return _write_artifact(ir, task_id, occ_title, ring, version, output_dir, log)
        for h in ir["learned"]:
            key = h["key"]
            kp = [(p, feats[id(p)][0], feats[id(p)][1][key], ok) for p, _, ok in results
                  if key in feats[id(p)][1] and p.gt_inference == h["label"] and feats[id(p)][0] is not None]
            residual = [c for c in kp if not c[3]]
            if not residual:
                log.append(f"v3: handler on '{key}' has no residual failures on the archive; no refinement")
                continue

            def val(c, f):
                return c[2] if f == "S" else c[1][f]

            best_split = None
            for f in ("V1", "V2", "V3", "V4", "S"):
                vals = sorted(set(val(c, f) for c in kp))
                for i in range(len(vals) - 1):
                    thr = round((vals[i] + vals[i + 1]) / 2.0, 4)
                    for ge_is_fail in (True, False):
                        correct = sum(1 for c in kp if ((val(c, f) >= thr) == ge_is_fail) == (not c[3]))
                        if best_split is None or correct > best_split[0]:
                            best_split = (correct, f, thr, ge_is_fail)
            correct, f, thr, ge_is_fail = best_split
            side = [c for c in kp if (val(c, f) >= thr) == ge_is_fail]
            tpl = TEMPLATES[h["template"]]
            if len(side) >= tpl["k"]:
                cons, coefs = _fit_coefs([(c[1], c[2], c[0].gt_treatment) for c in side], tpl, h["base"], h["x"], h["y"])
            else:
                cons, coefs = 0, None
            if coefs is None or cons < required_consensus(tpl["k"]):
                log.append(f"v3: residual {len(residual)}/{len(kp)} on '{key}'; best stump {f} {'>=' if ge_is_fail else '<'} {thr} "
                           f"separates {correct}/{len(kp)}; residual-side refit consensus {cons} < required {required_consensus(tpl['k'])}; no refinement")
                continue
            old = list(h["coefs"])
            h["split"] = {"feature": f, "thr": thr,
                          "ge_coefs": list(coefs) if ge_is_fail else old,
                          "lt_coefs": old if ge_is_fail else list(coefs)}
            if CFG.selection_validation_fraction > 0:
                held = [p for p, env, kv in val_features if env is not None and key in kv and p.gt_inference == h["label"]]
                candidate_fn = load_source_exactly(emit_source(ir, task_id, occ_title, ring, version), "<validation-candidate>").execute_compiled_routine
                score = sum(ok for _, _, ok in _evaluate_on_archive(candidate_fn, held))
                previous_score = sum(ok for _, _, ok in _evaluate_on_archive(fn, held))
                if not held or 2 * score < len(held) or score < previous_score:
                    h["split"] = None
                    log.append(f"v3: training-only stump on '{key}' rejected by validation ({score}/{len(held)}; previous {previous_score})")
                    continue
            log.append(f"v3: residual {len(residual)}/{len(kp)} on '{key}'; stump {f} {'>=' if ge_is_fail else '<'} {thr} "
                       f"separates {correct}/{len(kp)}; residual-side coefs={coefs}, consensus {cons}/{len(side)}")
    else:
        raise ValueError(version)
    return _write_artifact(ir, task_id, occ_title, ring, version, output_dir, log)


def synthesize_compiled_tool_from_handbook(
    task_id: str, occ_title: str, ring: str, doc_handbook: str, output_dir: str, version: str = "v1",
    sandbox=None, previous: Optional[CompiledToolArtifact] = None,
) -> CompiledToolArtifact:
    """v1: compile the public handbook. v2/v3: repair `previous` against the public resolved-case archive."""
    if version == "v1":
        ir = parse_handbook_spec(doc_handbook)
        log = [f"v1: compiled from public handbook: {len(ir['lets'])} LET, {len(ir['diags'])} DIAGNOSE, "
               f"{len(ir['rules'])} RULE, {len(ir['checks'])} CHECK statements; fields={sorted(ir['fields'])}"]
        return _write_artifact(ir, task_id, occ_title, ring, "v1", output_dir, log)
    if previous is None or sandbox is None:
        raise ValueError("v2/v3 require the previous generation and the public sandbox archive")
    return repair_generation(previous, sandbox, version, task_id, occ_title, ring, output_dir)


def load_source_exactly(source: str, filename: str, module_name: str = "generated_tool"):
    """Compiles and executes the exact `source` text in a fresh module namespace.

    No importlib source loader is used, so no __pycache__ bytecode is read or written: a file rewritten with the
    same byte length inside one timestamp tick cannot execute a stale program."""
    import types
    module = types.ModuleType(module_name)
    module.__file__ = filename
    code = compile(source, filename, "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def load_compiled_callable(artifact: CompiledToolArtifact) -> Callable[[str, str, str], Dict[str, object]]:
    """Returns `execute_compiled_routine` compiled from the artifact's exact emitted source.

    The in-memory `source_code` is authoritative; when the file on disk has since been overwritten by another
    candidate (paths are reused across periods), the artifact still executes the program it records."""
    source = artifact.source_code
    if not source:
        with open(artifact.filepath, encoding="utf-8") as f:
            source = f.read()
    name = f"mod_{artifact.task_id.replace('-', '_')}_{artifact.version}"
    return load_source_exactly(source, artifact.filepath, name).execute_compiled_routine
