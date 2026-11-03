# Subordination settlements: what is established, what failed, what is pending (2026-10-29)

Consolidates the week-1 tests and the settlements build phase. Sources:
- `CONTRIBUTION_MEMO.md`, `CAUSAL_DESIGN.md`;
- test notes: `T5v2_census_notes.md`, `RESULT_T1_v2_ARENAS.md` (this folder), and `research_notes/protection_week1_2026-10/` (T1, T2, T5, T8).

## 1. The idea
When AI competes for professional work, closure institutions do not keep AI out of the work. Licensed and organised professions win new rules. Their own regulatory bodies write rules that subordinate AI to the licensee. Legislatures, not the professions' bodies, are the source of exclusion. The law binds firms that deploy AI more than clients who use it themselves. In the terms of the AJ-revision theory, institutions protect the third component of the professional claim, the right to perform the work, while the codified content of the work moves to AI.

## 2. Established
| claim | evidence | strength |
|---|---|---|
| **Who wins rules.** Licensing and professional organisation predict AI rules, net of exposure; unions and public employment do not | **Test 5 v1:** licensing +.107 per SD (SE .039). **Test 5 v2, on the census:** licensing +.10 (Holm p .062); pre-2022 IRS association count +.067 (Holm p .0004) and 2019–21 revenue +.056 (Holm p .0065); union coverage −.001; public share −.035 | Moderate to strong. The association measures are dated before AI and pass Holm. Licensing is borderline because rules cover about 21 professions. Observational, net of exposure |
| **Who issues what.** Own bodies subordinate AI; legislatures exclude it | **Census of ~470 state bodies, 9 professions:** own bodies issued 80 subordination rows against 9 exclusion rows; 63 of the 80 are soft guidance. 13 of 15 binding exclusion documents are statutes (the exception is New Mexico's nursing rules). In 25 of 28 state-profession cells with an exclusion statute, the board issued nothing | Strong as a description; no longer a convenience sample |
| **Arena.** Legal reservation binds firm deployment more than client use | **Test 1, 17 professions, 36 occupations, AEI v3–v5 (Aug 2025–Feb 2026).** Within occupation, reserved tasks over-index in Claude.ai use (+.118, randomisation p .003). Firm API use reaches them less than consumer use does, relative to open tasks (−.082, p .006). The gap survives controls for client-facing, advice, decision and administrative tasks (−.076, p .009), the strict flag, each release, dropping health (−.119) and dropping any occupation (−.065 to −.103) | Moderate. Robust in v3–v5, but zero in v6 (Apr–May 2026, a different pipeline; reason unknown). Task usage is not use by the occupation's workers |
| **Exclusion's reach.** Bans do not reach clients' self-help | **Test 8:** no fall in Claude.ai emotional-support requests in ban states (−2%, p_RI .68; −4%, p_RI .71); falls larger than 16–22% excluded. The statutes exempt self-help or bind only providers | Moderate. A bounded null with few treated states. The provider side is pending (N3) |

## 3. Failed or null
- **Self-regulatory capacity as a cause (H2a).** Integrated bars and independent boards do not reliably act more or earlier: lawyers +.17 to +.20, randomisation p .13–.18; physicians and nurses null; capacity × national-shock hazard null. The raw curves for lawyers diverge after ABA Opinion 512 (.58 vs .35 by mid-2025), but this is not significant. Capacity may shape the form and issuer of instruments (official opinions vs court rules); that pattern is exploratory.
- **Public employment (Test 2).** No protection of exposed occupations; this line was dropped as uninteresting.
- **Occupation-level licensing × O*NET core (Test 1 v1).** Null. O*NET's Core flag is not the legal core; the reserved-acts flag replaced it.
- **Subordination in job ads (N1).** Supervision-of-AI language is rare: 535 sentences in 6.1 M postings, 427 of them in 2026, with precision .51. AI-skill tags are essentially zero in health professions.
- **Consequences for hiring and job content (C4, Test 9, 2026-10-29).** None of the six pre-specified tests is significant before or after Holm (`T9_consequences_notes.md`; `data/results/phase4/protection_2026-10/test9_consequences/`):
  - **Lawyers' postings.** In 31 states dated to the first own-body instrument, lawyer postings in the balanced firm panel moved by −0.065 log points (did2s; randomisation p .29). The interval excludes falls larger than about 16%, and the pre-trend is flat (placebo p .41).
  - **Lawyers relative to paralegals.** The triple difference is −0.059 (p .34). A raw within-state contrast agrees: −0.08 for lawyers at 12 months (p .10), but −0.02 relative to paralegals. Legal hiring as a whole grew a little less in the treated states, and the licensed profession lost no ground to its rival.
  - **AI content of lawyers' postings.** Neither the AI-skill share (p .63) nor the AI-mention share (p .91) rose. Both nulls are uninformative, because the detectable change is two to five times the pre-period share.
  - **Mental-health bans.** Therapist postings in NV and IL track the 39 control states (exact p .90; relative to social-service occupations p .53). With two treated states, only a near-doubling could be detected.
  - **Other professions.** Physicians, nurses, counsellors, real estate, engineers, and pharmacists show nothing (pooled −0.03, p .49). The title-law RN drop (p .045) has a pre-trend and fails Holm.
  - **Reading.** Subordination left hiring in the licensed profession, and its position relative to its rival, where the trends in other states put them, within what LinkUp can detect. This fits the settlement claim, because the right to perform the work stayed with the licensee. However, the design cannot show that guidance caused or legitimated AI use in practice.
- **Within-task, across-state design (C3-A).** Underpowered with current data: AEI state × task cells are thin for health tasks; the reserved-acts state coding covers six states; health job ads rarely mention AI. Revisit when the scope-change history (N2) is in.

## 4. Pending
- **N2:** dated scope changes for legal services, PA independence, pharmacist prescribing, hygienist direct access, counsellor diagnosis, and AI waivers.
- **N3 (DONE 2026-10-29):** 30 products, 2023-10 to 2026-10 (`data/intermediate/clean/ai_mh_providers_2026-10/`). No terms, FAQ, help-centre or app-store page excludes a US state; the only state block found is Ash in Illinois (in-app, reported in news 10 days after HB 1806). Providers instead relabelled nationally, dropping 'therapy' language: 7 relabel events, 5 of them 11–58 days after a ban or title law was signed (Elomia, ElizaChat, Ash, Earkick, Abby), 3 before any ban. Woebot retired its app on 2025-06-30, the day before Nevada's ban took effect; Youper closed on 2026-09-30; neither cites a state law. Reading: exclusion reached providers' labels (who may call a service therapy) more than their availability, which fits a title-and-service settlement and leaves clients' general-AI self-help untouched (Test 8). Original plan: AI mental-health and companion provider availability by state over time. This is the provider side of the exclusion claim; the frame of 30 products is fixed. Ash and Earkick issued statements on the Illinois and Nevada bans.
- **Consequences (C4): DONE 2026-10-29 (Test 9; see §3).** A bounded null on hiring. AI-content outcomes are too thin in LinkUp to be informative, and post-periods are short (a mean of 20 months for lawyers, 14–15 for the NV and IL bans).

## 5. Threats that remain
1. **Usage of a task is not use by the occupation's workers.** Consumer usage includes clients, students and other occupations.
2. **The v6 null** in the arena test is unexplained.
3. **Reserved-acts coding was done by language model** with two blind passes (κ .909 reserved vs not). Your two check samples are not yet reviewed (`research_notes/protection_week1_2026-10/reserved_acts_check_sample.xlsx`, `reserved_acts_check_sample_v2.xlsx`).
4. **Census completeness.** "Searched, none found" may miss unindexed guidance; 29 bodies could not be reached.
5. **Few professions** carry the rule outcomes (law and health dominate). Causal identification of who wins rules is limited to selection on observables.
