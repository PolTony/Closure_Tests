#!/usr/bin/env python3
"""
Labour demand and apprenticeship pipeline (labor_market.py) — JURI Lab

A deterministic accounting block layered on each period's governance outcomes, at the level of the
OCCUPATION (its core, buffer and periphery tasks pooled), because apprentices learn on buffer and periphery
work and replace the incumbents who hold the core. It turns per-packet governance into the quantities the
empirical literature measures (headcount, entry, utilisation, the price of the task) and adds the "becoming"
lens: who gets to accumulate the experience that makes an incumbent.

Per occupation and period, summing over its tasks k with volumes Q_k, governance shares (poach p_k, settle
s_k, gate g_k) and the shares of HARD cases (Tier 2 and 3) that were gated (hg_k) or settled (hs_k):
  incumbent hours        H_I = sum_k Q_k (g_k + s_k r)            r = settlement review fraction
  junior hours           H_J = sum_k Q_k (s_k + p_k)              one tool-equipped junior-hour per settled or poached packet
  learning pool          L   = sum_k Q_k (hg_k + lam * hs_k)      hands-on hard-case equivalents: a gated hard case is
                                                                   worked alongside the incumbent (1), a settled hard case
                                                                   is run through the tool and audited (lam), routine and
                                                                   poached cases teach nothing an incumbent needs to know
Stocks: incumbents I; apprentices as cohorts (headcount, accumulated experience), A = total headcount.
  demanded incumbents    I_d = H_I / cap                            cap = packets per worker-period
  demanded juniors       J_d = H_J / cap
  learning per apprentice x  = L / A                                added to every cohort's experience this period
  apprenticeship length  K   = labor_apprenticeship_periods * x_1   experience needed for promotion, set so that the
                                                                   period-1 pace yields eligibility after that many periods
  eligible               E   = headcount of cohorts with experience >= K
  vacancies              V   = max(0, I_d - I (1 - delta))          retirements delta plus any demand growth
  promotions             P   = min(E, V), most experienced first    promotion requires both experience and a vacancy
  apprenticeship debt    D   = (V - P) / I_d                        vacancies left unfilled for want of eligible apprentices
  eligible backlog       B   = (E - P) / A                          eligible apprentices waiting for a vacancy
  incumbents next        I'  = I (1 - delta) + P
  hires                  N   = max(0, J_d - (A - P))                replenish the junior pool to demand; a new cohort with
                                                                   zero experience
Reported: learning index x / x_1; periods to eligibility K / x; promotions over vacancies; debt; backlog; utilisation
u = H_I / (I cap); employment inside the occupation E_in = I + A; outside E_out = I; scarcity-implied incumbent
rate w_inc (I_d / I)^eta (not fed back into governance in this version); unit labour cost relative to period 1.
Demand: Q_k,t = Q_0 w_k (1 + g_d)^(t-1) (c_t / c_1)^(-eps), w_k the task's importance weight, c the unit cost.
All constants are in model_config.CFG (labor_*). Initial stocks are the period-1 demands (a mature pipeline).
"""

from dataclasses import dataclass, field
from typing import Dict, List

from model_config import CFG


@dataclass
class OccupationLabor:
    """Cohort version. Learning happens on HARD cases (Tier 2 and 3) worked hands-on: a gated hard case is worked
    alongside the incumbent (1), a settled hard case is run through the tool and audited (lam), a poached hard
    case involves no apprentice (0). Apprentices are tracked as cohorts [headcount, accumulated experience]; every
    cohort gains this period's learning per apprentice x = L / A; a cohort is eligible once its experience reaches
    K, the experience that `labor_apprenticeship_periods` periods of period-1 learning would give. Promotions are
    the smaller of the eligible headcount and the vacancies, most experienced first; eligible apprentices who find
    no vacancy wait (the backlog); vacancies that find no eligible apprentice are the apprenticeship debt. New
    hires enter as a cohort with zero experience, so a slower learning pace delays their eligibility one for one."""
    occ_title: str
    w_inc: float
    review_fraction: float
    I: float = 0.0
    K: float = 0.0
    x1: float = 0.0
    c1: float = 1.0
    cohorts: List[List[float]] = field(default_factory=list)   # [headcount, experience], oldest first
    initialised: bool = False
    history: List[Dict[str, float]] = field(default_factory=list)

    def step(self, t: int, tasks: List[Dict[str, float]]) -> Dict[str, float]:
        """tasks: list of {"weight", "poach", "settle", "gate", "hard_gate", "hard_settle", "unit_cost"} for this period."""
        cap, lam, delta, gd, eps, eta = (CFG.labor_capacity, CFG.labor_lambda_settle, CFG.labor_delta,
                                         CFG.labor_demand_growth, CFG.labor_price_elasticity, CFG.labor_scarcity_eta)
        wsum = sum(tk["weight"] for tk in tasks) or 1.0
        unit_cost = sum(tk["unit_cost"] * tk["weight"] for tk in tasks) / wsum
        if not self.initialised:
            self.c1 = unit_cost
        scale = CFG.labor_Q0 * len(tasks) * (1.0 + gd) ** (t - 1) * (unit_cost / self.c1) ** (-eps)
        H_I = H_J = L = 0.0
        for tk in tasks:
            Q = scale * tk["weight"] / wsum
            H_I += Q * (tk["gate"] + tk["settle"] * tk.get("review_time", self.review_fraction))   # physical review time, not the fee
            if CFG.labor_junior_operates_tool:
                H_J += Q * (tk["settle"] + tk["poach"])                 # juniors run the program on settled and poached work
                L += Q * (tk["hard_gate"] + lam * tk["hard_settle"])
            else:
                H_J += Q * tk["gate"] * CFG.labor_junior_assist        # the program replaces junior hours; juniors assist on gated work only
                L += Q * tk["hard_gate"]
        I_d, J_d = H_I / cap, H_J / cap
        if not self.initialised:
            # a mature pipeline: incumbents at demand, apprentices at junior demand spread evenly over the
            # apprenticeship, so that one cohort becomes eligible each period at the period-1 pace
            self.I, self.initialised = I_d, True
            A0 = max(J_d, 1e-9)
            self.x1 = L / A0
            self.K = CFG.labor_apprenticeship_periods * max(self.x1, 1e-9)
            n = max(1, int(round(CFG.labor_apprenticeship_periods)))
            self.cohorts = [[A0 / n, self.K * k / n] for k in range(n)]
        A = sum(c[0] for c in self.cohorts)
        x = L / max(A, 1e-9)
        for c in self.cohorts:
            c[1] += x
        eligible = sum(c[0] for c in self.cohorts if c[1] >= self.K - 1e-12)
        vacancies = max(0.0, I_d - self.I * (1.0 - delta))
        P = min(eligible, vacancies)
        remaining = P
        for c in sorted(self.cohorts, key=lambda c: -c[1]):
            if remaining <= 0:
                break
            take = min(c[0], remaining)
            c[0] -= take
            remaining -= take
        debt = (vacancies - P) / max(I_d, 1e-9)
        backlog = (eligible - P) / max(A, 1e-9)
        I_next = self.I * (1.0 - delta) + P
        A_after = A - P
        N = max(0.0, J_d - A_after)
        self.cohorts = [c for c in self.cohorts if c[0] > 1e-12]
        if N > 0:
            self.cohorts.append([N, 0.0])
        util = H_I / max(self.I * cap, 1e-9)
        w_scarce = self.w_inc * (max(I_d, 1e-9) / max(self.I, 1e-9)) ** eta
        rec = {"period": t, "I_demand": I_d, "I_stock": self.I, "J_demand": J_d, "A_stock": A,
               "learning_per_apprentice": x, "learning_index": x / max(self.x1, 1e-9),
               "periods_to_eligibility": (self.K / x) if x > 0 else float("inf"),
               "eligible": eligible, "vacancies": vacancies, "promotions": P, "hires": N,
               "apprenticeship_debt": debt, "eligible_backlog": backlog, "utilisation": util,
               "emp_inside": self.I + A, "emp_outside": self.I,
               "unit_cost_rel": unit_cost / self.c1, "w_inc_scarcity": w_scarce}
        self.history.append(rec)
        self.I = I_next
        return rec
