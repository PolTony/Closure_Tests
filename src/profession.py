#!/usr/bin/env python3
"""
The learning profession (profession.py) — JURI Lab

The reference configuration's incumbent knows the whole rulebook. This module replaces that oracle, in the dynamics, with
a PROFESSION whose knowledge is reproduced through practice: cohorts of seniors and juniors who hold, or have not yet
learned, the two components of the domain's knowledge that are not in the handbook,

    convention   the Tier 2 rule that is never written down and is learned by handling convention cases,
    registry     the Tier 3 precedent constants, held privately and transmitted only by working beside someone who holds them.

Nothing here changes the ground truth or the firewall. A professional still applies the rulebook through the shared
interpreter (negotiated_order_lab._incumbent_three_hops), but only the parts the knowledge mask says they have learned:
without the convention they read a convention case as the handbook does; without the registry they cannot resolve an
overt precedent memo and cannot apply the silent-collision gate. The mask is drawn per case from the share of the
relevant pool that holds the component, so the profession's accuracy is a state variable rather than a constant.

Knowledge is acquired by EXPOSURE, in case volumes per head, with weights on the model card (learn_w_*):
    a junior working a hard case alongside a senior who holds the component   learn_w_gate_junior     (gated work)
    a junior running a settled hard case through a program that made the      learn_w_settle_junior   (settlement)
      rule explicit (its handler fired)
    a senior deep-reviewing a program answer that applied an explicit handler  learn_w_review_deep     (settlement)
    a senior shallow-reviewing such an answer                                  learn_w_review_shallow  (settlement)
    a senior escalating an overt precedent memo to a colleague who holds       learn_w_escalate_senior (gated work)
      the registry
A poached case teaches nobody in the profession. A case that nobody present could resolve teaches nobody either: the
convention and the registry are transmitted, not rediscovered. A cohort holds a component once its per-head exposure
reaches the threshold kappa. The thresholds are CALIBRATED, not typed: kappa is the exposure a junior would
accumulate over `labor_apprenticeship_periods` periods at the pre-compilation pace, in which every hard case of the
occupation is worked alongside a senior who holds the knowledge; `learn_kappa_scale` scales it for sensitivity runs.
So in the registered calibration a pre-compilation apprenticeship transmits both components exactly, and the question
the dynamics answer is how the compiled tool's arrival changes what the apprenticeship transmits.

Flows are synchronised with the labour block (labor_market.OccupationLabor): the same retirements, promotions and
hires, applied oldest cohort first. Promoted juniors carry their exposures into the senior pool. New hires know the
handbook only. At period 1 every senior holds both components (the profession as the reference configuration assumes it),
and the apprentice cohorts, which were trained at the pre-compilation pace up to the tool's arrival, hold exposures
in proportion to their progress: cohort k of n holds (k + 1)/n of kappa, so the cohort due for promotion already
holds both components and the first promotion is not an artefact of the seeding.

Two thresholds govern the pipeline and they are related, which matters for reading the results. The labour block
promotes a cohort once its pooled hard-case experience reaches K = labor_apprenticeship_periods x (period-1 learning
per apprentice), where period-1 learning counts gated hard cases at 1 and settled ones at labor_lambda_settle, Tiers
2 and 3 pooled. This module's kappa, per component, is labor_apprenticeship_periods x (pre-compilation exposure per
head), which counts every hard case of that component at the gated weight. Eligibility (K) and competence (kappa)
therefore coincide in a world without the tool, and come apart through two wedges: K is set at the realised period-1
pace while kappa is set at the pre-compilation pace, and K pools the components while kappa counts them separately.
A promoted cohort can hold the registry and not the convention, or neither, only through those wedges, so hollowing is
a statement about how the tool's arrival moves the pace of each component relative to the pace of promotion.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from domain_cases import _det_float
from model_config import CFG

COMPONENTS = ("convention", "registry")


@dataclass
class KnowledgeCohort:
    n: float                      # headcount
    entered: int                  # period of entry into the pool (ordering for retirements and promotions)
    exposure: Dict[str, float] = field(default_factory=lambda: {"convention": 0.0, "registry": 0.0})


@dataclass
class OccupationKnowledge:
    occ_title: str
    kappa: Dict[str, float] = field(default_factory=dict)   # exposure per head at which a cohort holds the component
    seniors: List[KnowledgeCohort] = field(default_factory=list)
    juniors: List[KnowledgeCohort] = field(default_factory=list)
    credits: Dict[str, Dict[str, float]] = field(default_factory=lambda: {"seniors": {"convention": 0.0, "registry": 0.0},
                                                                          "juniors": {"convention": 0.0, "registry": 0.0}})
    initialised: bool = False
    history: List[Dict[str, float]] = field(default_factory=list)

    # ---- state -------------------------------------------------------------------------------------------------
    def pool(self, name: str) -> List[KnowledgeCohort]:
        return self.seniors if name == "seniors" else self.juniors

    def headcount(self, name: str) -> float:
        return sum(c.n for c in self.pool(name))

    def holds(self, cohort: KnowledgeCohort, comp: str) -> bool:
        return cohort.exposure[comp] >= self.kappa[comp] * (1.0 - 1e-6)

    def share(self, name: str, comp: str) -> float:
        """Headcount share of the pool that holds the component. Before initialisation (period 1, before the labour block
        has reported its stocks) every senior is the registered oracle (1.0) and the junior pool is not yet defined (nan)."""
        if not self.initialised:
            return 1.0 if name == "seniors" else float("nan")
        n = self.headcount(name)
        if n <= 1e-12:
            return 0.0
        return sum(c.n for c in self.pool(name) if self.holds(c, comp)) / n

    # ---- initialisation from the labour block at the end of period 1 ---------------------------------------------
    def initialise(self, I_stock: float, labor_cohorts: List[List[float]], K: float, pretool_exposure: Dict[str, float], t: int,
                   period1_exposure: Optional[Dict[str, float]] = None) -> None:
        """`labor_cohorts` are the labour block's apprentice cohorts [headcount, experience] as they stood at the start
        of period 1 (before promotions); `K` its eligibility threshold; `pretool_exposure` the per-head exposure per
        period a junior would gain if every hard case of the occupation were worked alongside a knowing senior;
        `period1_exposure` the per-head exposure juniors actually gained in period 1 (the labour block's basis for K),
        used when `learn_kappa_basis == "period1"`."""
        basis = pretool_exposure if (CFG.learn_kappa_basis == "pretool" or period1_exposure is None) else period1_exposure
        self.kappa = {comp: CFG.learn_kappa_scale * CFG.labor_apprenticeship_periods * max(basis[comp], 1e-9) for comp in COMPONENTS}
        self.seniors = [KnowledgeCohort(I_stock, 0, {comp: self.kappa[comp] for comp in COMPONENTS})]
        self.juniors = []
        n_c = max(1, len(labor_cohorts))
        for k, (n, exp) in enumerate(labor_cohorts):
            # cohorts in the pipeline at the tool's arrival were trained at the pre-compilation pace: cohort k of n_c
            # holds (k + 1) / n_c of kappa, so the cohort due for promotion already holds both components
            f = min(1.0, (k + 1) / n_c)
            self.juniors.append(KnowledgeCohort(n, -(k + 1), {comp: f * self.kappa[comp] for comp in COMPONENTS}))   # the most experienced cohort is the oldest
        self.initialised = True

    # ---- per-case masks ----------------------------------------------------------------------------------------
    def draw_mask(self, name: str, seed: str, overt_precedent: bool) -> Tuple[Dict[str, bool], bool]:
        """The knowledge of the professional who handles (or reviews) one case, drawn from the pool's shares. An overt
        precedent memo is routed to a senior who holds the registry while one exists (escalation; model-card switch);
        a convention case cannot be routed because nothing on its face says it is one."""
        s_conv, s_reg = self.share(name, "convention"), self.share(name, "registry")
        conv = _det_float(seed + "_conv", 0.0, 1.0, 6) < s_conv
        escalated = False
        if overt_precedent and CFG.learn_escalate_overt and s_reg > 1e-12:
            reg, escalated = True, _det_float(seed + "_reg", 0.0, 1.0, 6) >= s_reg
        else:
            reg = _det_float(seed + "_reg", 0.0, 1.0, 6) < s_reg
        return {"convention": conv, "registry": reg}, escalated

    # ---- exposure credits ---------------------------------------------------------------------------------------
    def credit(self, name: str, comp: str, amount: float) -> None:
        self.credits[name][comp] += amount

    # ---- end of period: exposure, retirements, promotions, hires (in step with the labour block) -----------------
    def end_period(self, t: int, labor_rec: Dict[str, float]) -> Dict[str, float]:
        I0, A0 = max(labor_rec["I_stock"], 1e-9), max(labor_rec["A_stock"], 1e-9)
        per_head = {name: {comp: self.credits[name][comp] / (I0 if name == "seniors" else A0) for comp in COMPONENTS} for name in ("seniors", "juniors")}
        for name in ("seniors", "juniors"):
            for c in self.pool(name):
                for comp in COMPONENTS:
                    c.exposure[comp] += per_head[name][comp]
        before = {f"{name}_{comp}": self.share(name, comp) for name in ("seniors", "juniors") for comp in COMPONENTS}
        # retirements: delta of the senior stock, oldest cohorts first
        self._remove(self.seniors, CFG.labor_delta * labor_rec["I_stock"], key=lambda c: c.entered)
        # promotions: the labour block's count, most experienced juniors first (its rule); they carry their exposures
        promoted = self._remove(self.juniors, labor_rec["promotions"], key=lambda c: -(c.exposure["convention"] + c.exposure["registry"]))
        promoted_n = sum(n for n, _ in promoted)
        promoted_hold = {comp: (sum(n for n, e in promoted if e[comp] >= self.kappa[comp] * (1.0 - 1e-6)) / promoted_n) if promoted_n > 1e-12 else float("nan") for comp in COMPONENTS}
        for n, exposure in promoted:
            if n > 1e-12:
                self.seniors.append(KnowledgeCohort(n, t, dict(exposure)))
        # hires: a new cohort that knows the handbook only
        if labor_rec["hires"] > 1e-12:
            self.juniors.append(KnowledgeCohort(labor_rec["hires"], t))
        rec = {"period": t, "seniors": self.headcount("seniors"), "juniors": self.headcount("juniors"),
               "credit_seniors_convention": self.credits["seniors"]["convention"], "credit_seniors_registry": self.credits["seniors"]["registry"],
               "credit_juniors_convention": self.credits["juniors"]["convention"], "credit_juniors_registry": self.credits["juniors"]["registry"],
               "per_head_juniors_convention": per_head["juniors"]["convention"], "per_head_juniors_registry": per_head["juniors"]["registry"],
               "kappa_convention": self.kappa["convention"], "kappa_registry": self.kappa["registry"],
               "promoted": promoted_n, "promoted_hold_convention": promoted_hold["convention"], "promoted_hold_registry": promoted_hold["registry"], **before}
        self.credits = {"seniors": {"convention": 0.0, "registry": 0.0}, "juniors": {"convention": 0.0, "registry": 0.0}}
        self.history.append(rec)
        return rec

    @staticmethod
    def _remove(pool: List[KnowledgeCohort], amount: float, key) -> List[Tuple[float, Dict[str, float]]]:
        """Removes `amount` of headcount from the cohorts in `key` order; returns what was removed with its exposures."""
        removed: List[Tuple[float, Dict[str, float]]] = []
        remaining = amount
        for c in sorted(pool, key=key):
            if remaining <= 1e-12:
                break
            take = min(c.n, remaining)
            c.n -= take
            remaining -= take
            removed.append((take, dict(c.exposure)))
        pool[:] = [c for c in pool if c.n > 1e-12]
        return removed


def pretool_exposure_per_head(task_volumes: Dict[str, float], tier_shares: Dict[str, Dict[int, float]], juniors0: float) -> Dict[str, float]:
    """Per-head exposure per period at the pre-compilation pace: every convention case (Tier 2) counts toward the
    convention and every precedent case (Tier 3) toward the registry, at the gated-work weight."""
    conv = sum(task_volumes[k] * tier_shares[k].get(2, 0.0) for k in task_volumes) * CFG.learn_w_gate_junior
    reg = sum(task_volumes[k] * tier_shares[k].get(3, 0.0) for k in task_volumes) * CFG.learn_w_gate_junior
    j = max(juniors0, 1e-9)
    return {"convention": conv / j, "registry": reg / j}
