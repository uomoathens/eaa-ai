# EAA-AI: Explainable Accountability-Aware AI Architecture

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Accountability audit](../../actions/workflows/audit.yml/badge.svg)](../../actions/workflows/audit.yml)

Reference implementation of the Explainable Accountability-Aware AI Architecture introduced in

> Sargiotis, D. (2026). *Empathy in Algorithms: Shaping Ethical AI for a Just and Inclusive Society*. Springer Nature. §8.4 (EAA-AI) and §9.4 (XAI-AAI). https://doi.org/10.1007/978-3-032-32128-2

The repository turns the architecture into runnable, testable software and extends it with four refinements aimed at the high-risk obligations of the EU AI Act (Regulation (EU) 2024/1689), which apply to stand-alone Annex III systems from 2 December 2027 following the Digital Omnibus on AI.

## What it does

Every training run of a public-service decision model (the bundled example is eligibility for a public benefit) produces:

| Output | Content |
|---|---|
| `audit_ledger.jsonl` | Tamper-evident, hash-chained record of every parameter update, compliance evaluation, rollback, deployment decision and human review |
| `eaa_ai_audit_report.xlsx` | Summary, requirement-by-requirement pass/fail, full training history, ledger, global SHAP importance and the top three reasons behind each individual decision |
| `model.json` | Selected parameters, their hash and the ledger head they are bound to |

## Architecture

| Layer | Book | Module |
|---|---|---|
| 1 Data collection and preprocessing | Data Preprocessing and Validation Engine | `data.py` |
| 2 Explainability attribution | SHAP | `explain.py` (exact SHAP for the linear model, locally accurate by test) |
| 3 Traceable parameter optimisation | Parameter Traceability Optimisation Module | `ledger.py`, `trainer.py` |
| 4 Ethical compliance monitoring | Ethical Compliance Monitoring Engine | `compliance.py` |
| 5 Responsible AI reward | Reinforcement Learning | `reward.py` |
| 6 Continuous evaluation and audit | Transparency Evaluation and Audit Module | `report.py`, GitHub Actions |

The learner is a fairness-regularised logistic regression with the governance loss of the book (EGAPF, §1.1):

```math
L_{\text{total}} = L_{\text{task}} + \lambda L_{\text{fair}} + \beta L_{\text{explain}}
```

## Refinements over the published formulation

**1. Attribution moved out of the parameter arithmetic.** The book writes $`\theta^{*} = \theta_{0} + \sum_{t} \Delta_{t} \cdot I_{d_t}`$, in which an indicator would zero out unattributed updates. Here the parameters evolve as $`\theta^{*} = \theta_{0} + \sum_{t} \Delta_{t}`$ and attribution lives in a hash-chained ledger:

```math
\mathcal{L}_{\text{audit}} = \left\{ \left( t,\ h(\Delta_t),\ d_t,\ h(\mathcal{D}_t),\ C_{\text{ethics}}(\theta_t) \right) \right\}_{t=1}^{T}, \quad h_t = H\left(h_{t-1} \,\|\, \text{record}_t\right)
```

Any edited, inserted or deleted record breaks verification (AI Act, Article 12).

**2. Continuous compliance penalty and hard deployment gate.** The binary bonus $`\gamma_2 C_{\text{ethics}}`$ rewards barely crossing a threshold and is blind to the size of a violation. Both rewards are computed and logged; the refined one is

```math
R(s_t,a_t) = U_{\text{task}}(a_t) - \gamma_1 L_{\text{bias}}(a_t) - \gamma_2 \sum_{m} \max\left(0,\ \tau_m - m(\theta_t)\right)
```

and deployment requires $`\min_m (m(\theta_t) - \tau_m) \ge 0`$ on held-out data. If the final iteration fails, the model is rolled back to the last compliant iteration and the rollback is recorded.

**3. Proxy-aware explanation metric.** `sensitive_attribution_share` measures how much of the SHAP mass falls on the protected attribute and on declared proxies (here `rural_region`), so a model that is blind to the attribute but leans on a proxy is still caught.

**4. Role-based accountability.** Records are attributed to AI Act roles (`provider`, `deployer`, `oversight-board`) and organisational units rather than to individual engineers. Human oversight (Article 14) is a separate, signed ledger event.

## Requirements checked on every run

| Metric | Default rule |
|---|---|
| accuracy | ≥ 0.75 |
| abs_statistical_parity_difference | ≤ 0.10 |
| disparate_impact_ratio | ≥ 0.80 (four-fifths rule) |
| equalized_odds_difference | ≤ 0.15 |
| counterfactual_flip_rate | ≤ 0.10 |
| sensitive_attribution_share | ≤ 0.20 |

All thresholds live in `configs/default.yaml`.

## Running it on GitHub

Nothing needs to be run locally.

- **CI** runs the test suite on Python 3.10 to 3.13 on every push and pull request.
- **Accountability audit** runs on every push to `main`, every Monday at 05:17 UTC and on demand (Actions → Accountability audit → Run workflow). It trains the governed model with `--strict`, trains an unconstrained baseline for comparison, verifies the ledger, writes the results table to the run summary and uploads all outputs as the `eaa-ai-audit` artifact.
- **Human review** (Actions → Human review → Run workflow) takes the run ID of an audit, a decision and a justification, appends a `human_review` record to that run's ledger, re-verifies the chain and stores the reviewed ledger for 365 days.

With the bundled configuration the governed model passes every requirement while the unconstrained baseline is blocked on the disparate impact ratio, which shows the gate doing its job.

## Using your own data

Set `data.source` to `file`, point `data.path` to an `.xlsx` or `.csv` table, and name the `target` and `sensitive` columns. List any suspected proxies under `proxy_features` and any attributes the model must not use as inputs under `model.blind_features`.

## Limitations

The bundled data are synthetic. The exact SHAP explainer applies to the linear model; non-linear or generative models need a model-agnostic explainer or the counterfactual regulariser of CER-GAI (§8.4). When historical labels are themselves biased, demographic parity and equalized odds pull in opposite directions; raising `historical_bias` in the configuration makes this trade-off visible, and the gate will then block deployment rather than hide it. Passing the gate supports, but does not by itself constitute, a conformity assessment under the AI Act.

## Licence

MIT. See `LICENSE`.
