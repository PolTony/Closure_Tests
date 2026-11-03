# Prospective validation protocol (frozen 2026-11-01, before any new independent validation)

Status: FROZEN. I wrote and fixed this protocol before running any of the comparisons below on independent out-of-sample field data. Because no independent held-out field dataset is available at the time of freezing, I have not yet performed out-of-sample validation under this protocol. Every existing comparison in the technical appendix and proposal, including the pre-AI task-allocation and bumping baseline (`data_outputs/bumping_extended.csv`), the consistency inventory in Section 18c (`data_outputs/validation_ladder.csv`), the endogenous closure-formation game in Section 18b (`data_outputs/closure_formation_game.csv`), and the micro-to-macro bridge in Table 4 (`data_inputs/macro_outcomes.csv` and `data_inputs/macro_outcomes_extended_lab54.csv`), serves as an exploratory, calibrated in-sample face-validity check computed under a reference configuration whose structural constants (`model_config.py`) I set iteratively while aware of the occupational literature. Calibrated in-sample face-validity checks verify that the laboratory's structural primitives reproduce known occupational hierarchies and supervision patterns under specified parameters, whereas independent out-of-sample validation requires predicting held-out cases coded without access to laboratory outputs. Any later amendment must be dated, appended below the freeze line, and must leave the frozen sections unaltered.

## 1. Distinction between calibrated in-sample face validity and out-of-sample validation

I distinguish two levels of empirical evaluation across the project:
- **Stage A calibrated in-sample face validity (completed in Stage 1)**: Using the 18 occupations and 54 tasks in the reference configuration, I test whether the laboratory's structural primitives (information sets, error liability weights, and occupational wage ratios) reproduce foundational theoretical patterns from Abbott (1988), Freidson (2001), and Weeden (2002), such as concentric task delegation to subordinate occupations (`run_bumping.py`) and modal subordination settlements in contested work (`run_validation_ladder.py`), and I identify where endogenous statute formation fails without third-party harm (`run_closure_formation.py`). Because these tests rely on calibrated parameters in `model_config.py`, I report them as calibrated face-validity and boundary-condition diagnostics rather than independent out-of-sample validation.
- **Prospective out-of-sample validation (specified below for Stage 2)**: The laboratory claims that the governance of a contested work packet (settled under incumbent sign-off, gated behind solo incumbent execution, or poached directly by a challenger) depends on whether the rule that decides the case can be compiled and verified, as well as on malpractice liability and statutory penalty terms. The prospective out-of-sample claim to test is narrow: occupation-level gate status and settlement shares predicted by the laboratory's governance and closure rules, computed solely from pre-specified external inputs on held-out occupations, agree with independently coded institutional status better than simple baseline predictors do.

## 2. Shared constructs, unit and contrast

Each prospective comparison must state, before held-out data are opened:
- **Construct** on both sides with the same operational definition (for example, a statutory reservation of a named task to a licensed practitioner, rather than occupational licensing in general).
- **Unit**: the source unit at which the independent data are recorded (occupation-by-jurisdiction, task-by-occupation, or regulatory body). I aggregate laboratory values to that source unit and never disaggregate source data to synthetic packets.
- **Contrast**: the comparison carried by the test (for example, gated versus ungated tasks within a jurisdiction), with its directional sign predicted in advance.
Any comparison that cannot meet these three conditions is reported as a descriptive juxtaposition and excluded from the validation benchmark.

## 3. Held-out benchmark against simple alternatives

- **Hold-out set**: occupation-jurisdiction cases from the expanded O*NET corpus (`data_inputs/proposed_additional_occupations.csv`) that are not among the 18 occupations used to develop the reference configuration, coded by a domain coder who has not inspected laboratory outputs.
- **Predictors compared on identical held-out cases**: (a) the laboratory closure and governance rule; (b) licensing-only (predict a gate whenever the occupation's CPS licensing share exceeds 0.50); (c) harm-only (predict a gate whenever expected visible harm exceeds the fixed reference threshold of 0.30); (d) base rate (predict the majority class from the development set); and (e) a logistic regression on licensing share and log employment size fitted exclusively on the 18 development occupations.
- **Metrics**: balanced accuracy, sensitivity, specificity, and Brier score where probabilistic predictions exist, accompanied by 95% bootstrap confidence intervals over source units.
- **Success criterion (fixed now)**: the laboratory rule achieves a balanced accuracy exceeding both the licensing-only predictor (b) and the harm-only predictor (c), with the lower 95% bootstrap bound of the difference strictly above zero. Otherwise, I report the result as a failure to outperform simple alternatives.

## 4. Case-level gate proxy provenance

Every coded statutory gate must carry a statute or administrative rule citation, jurisdiction, effective date, coder identity, and a verbatim statutory excerpt. Indirect proxies (such as legislative bill mentions) are not admissible as gate codes; if examined, I report them separately as legislative salience proxies. A second coder independently codes a random 25% sample of tasks; I report Cohen's kappa and adjudicate disagreements before unsealing model predictions.

## 5. Rival explanations with distinct predictions

For each comparison, I record the predictions of three rival theoretical accounts:
- **Public-interest quality regulation**: statutory gates track expected third-party and client harm prevented, regardless of task codifiability.
- **Entry standardization** (Redbird, 2017): licensing standardizes entry credentials and raises occupational participation without restricting labor supply or generating a net wage premium.
- **Supply restriction and regulatory capture** (Kleiner and Soltas, 2023): statutory gates restrict labor supply, raise service prices and incumbent rents, and track professional lobbying capacity.
Any comparison in which the laboratory and a rival account predict the same directional sign is reported as non-discriminating.

## 6. Harm outcome

I report expected harm prevented alongside residual error (silent and overt errors that still occur under statutory gatekeeping or accountability settlement). Because the synthetic laboratory does not identify unique external health or financial harm effects in field settings, I make no claim from laboratory output alone that a real-world statute causes or prevents field harm.

## 7. Bridge inference (source-unit level)

The micro-to-macro bridge (`data_inputs/macro_outcomes.csv` and `data_inputs/macro_outcomes_extended_lab54.csv`) is descriptive at the source-unit level. Where I report formal inference on held-out bridge data, I use a conditional Freedman-Lane permutation procedure that respects the multi-source design: the outcome is regressed on nuisance covariates (platform or source fixed effects and occupation employment size), residuals are permuted within source strata at the harmonized occupation-unit level, and the institutional shield-gap coefficient is recomputed across 10,000 permutations. Because the laboratory's institutional shield gap correlates strongly with the statutory-gate indicator ($R^2 = 0.94$ across the 18 reference occupations), I always report the gate-controlled specification alongside the baseline bridge and do not claim predictive content beyond statutory licensing unless the gate-controlled coefficient is independently distinguishable from zero.

## 8. Reporting

I report all pre-specified comparisons, including negative findings and null results, and list any deviations with explicit reasons. The declarative run manifest (`reproduce/manifest.json` and `data_outputs/RUN_MANIFEST.json`) and the frozen configuration hash accompany all reported outputs.

---- FREEZE LINE (amendments below, dated) ----

## Editorial cross-reference update (2026-11-01)

The submission appendix has been consolidated. Occupational competition is now in Section 7; licensing formation, the empirical comparisons, and their limits are in Section 9; reproduction is in Section 10. The prospective comparisons and success criteria above are unchanged.
