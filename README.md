# EAA-AI: Explainable Accountability-Aware AI Architecture

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Accountability audit](../../actions/workflows/audit.yml/badge.svg)](../../actions/workflows/audit.yml)

Reference implementation of the Explainable Accountability-Aware AI Architecture introduced in

> Sargiotis, D. (2026). *Empathy in Algorithms: Shaping Ethical AI for a Just and Inclusive Society*. Springer Nature. §8.4 (EAA-AI) and §9.4 (XAI-AAI). https://doi.org/10.1007/978-3-032-32128-2

The repository turns the architecture into runnable, testable software and extends it with five refinements aimed at the high-risk obligations of the EU AI Act (Regulation (EU) 2024/1689), which apply to stand-alone Annex III systems from 2 December 2027 following the Digital Omnibus on AI.

## What it does

Every training run of a public-service decision model (the bundled example is eligibility for a public benefit) produces:

| Output | Content |
|---|---|
| `audit_ledger.jsonl` | Tamper-evident, hash-chained record of every parameter update, compliance evaluation, rollback, deployment decision and human review |
| `eaa_ai_audit_report.xlsx` | Summary, requirement-by-requirement pass/fail with 95% bootstrap confidence intervals, model selection across fairness weights, diagnostics against a reference model, full training history, ledger, global SHAP importance and the top three reasons behind each individual decision |
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

**5. Fairness only as strong as needed, chosen with statistical confidence.** The book fixes the fairness weight $`\lambda`$. Here a grid of weights is trained and the **smallest** $`\lambda`$ is selected whose validation metrics clear every requirement at the 95% one-sided bootstrap confidence level:

```math
\lambda^{*} = \min \left\{ \lambda \in \Lambda \ :\ \min_{m} \left( Q_{0.05}\big(m_{\lambda}\big) - \tau_m \right) \ge 0 \right\}
```

for requirements of the form $`m \ge \tau_m`$, with the upper quantile used for requirements of the form $`m \le \tau_m`$. If no weight qualifies, the candidate closest to qualifying is selected and the shortfall is recorded. Every candidate, its confidence margins and the reason for the choice are written to the ledger. This guards against two failure modes observed while building the repository: applying a strong constraint where no bias exists, which can move decisions the wrong way, and selecting a weight on the strength of validation noise.

## Requirements checked on every run

| Metric | Default rule |
|---|---|
| accuracy_lift (accuracy minus that of always predicting the majority outcome) | ≥ 0.02 |
| abs_statistical_parity_difference | ≤ 0.10 |
| disparate_impact_ratio | ≥ 0.80 (four-fifths rule) |
| equalized_odds_difference | ≤ 0.15 |
| counterfactual_flip_rate | ≤ 0.10 |
| sensitive_attribution_share | ≤ 0.20 |

All thresholds live in the configuration files. An absolute accuracy threshold was used in the first version and replaced, because its meaning depends on the base rate of each dataset; the lift criterion is

```math
\text{acc} - \max\left(\bar{y},\, 1-\bar{y}\right) \ge 0.02
```

Each report also carries a **Diagnostics** sheet: observed and predicted positive rates for each group, the accuracy of the constant predictor, and an unconstrained scikit-learn logistic regression trained on the same inputs, so that under-training can be told apart from a genuine limit of the data.

## Running it on GitHub

Nothing needs to be run locally.

- **CI** runs the test suite on Python 3.10 to 3.13 on every push and pull request.
- **Accountability audit** runs on every push to `main`, every Monday at 05:17 UTC and on demand (Actions → Accountability audit → Run workflow). It trains the governed model with `--strict`, trains an unconstrained baseline for comparison, repeats both on real ACS data, verifies the ledgers, writes the results table to the run summary and uploads all outputs as the `eaa-ai-audit` artifact.
- **Human review** (Actions → Human review → Run workflow) takes the run ID of an audit, a decision and a justification, appends a `human_review` record to that run's ledger, re-verifies the chain, shows the decision, the accountable role, the justification and the hash of the approved model on the run summary page, adds a **Human review** sheet to the Excel report and brings its ledger and summary sheets up to date, and stores the reviewed outputs for 365 days. A review is refused if the ledger fails verification.

On the synthetic data (60,000 records with a deliberately strong historical bias) the governed model selects $`\lambda = 1`$ and passes every requirement, while the unconstrained baseline is blocked with a disparate impact ratio of 0.766 (95% CI 0.730 to 0.806). The governed model passes on point estimates but with thin margins (disparate impact ratio 0.808, 95% CI 0.767 to 0.851; equalized odds difference 0.1496 against a limit of 0.15), which the confidence intervals make explicit rather than hide. An earlier version with 6,000 records appeared to separate the two models more sharply; at that sample size the difference was within sampling noise.

Each audit run produces four results, all written to the run summary and the artifact:

| Output folder | Data | Fairness constraint | Gates the workflow |
|---|---|---|---|
| `governed` | Synthetic public benefit | Yes | Yes (`--strict`) |
| `baseline` | Synthetic public benefit | No | No |
| `acs_public_coverage` | Real ACS public coverage | Yes | No |
| `acs_baseline` | Real ACS public coverage | No | No |

The real-data runs are reported, not enforced: whether a model trained on real data clears every requirement is a finding to be read, not a build condition.

## Real data: ACS public health coverage

`configs/acs_public_coverage.yaml` applies the architecture to the ACSPublicCoverage task of Folktables: predicting whether a low-income individual under 65 (personal income up to USD 30,000) is covered by public health insurance. The data are the 2018 one-year American Community Survey Public Use Microdata Sample for California, downloaded from the US Census Bureau on the first run and cached between runs.

- **Sensitive attribute:** race (`RAC1P`), binarised as White versus all other groups. It is used only to measure bias and is blind to the model (`blind_features`), in the spirit of Article 10(5) of the AI Act.
- **Declared proxies:** ancestry (`ANC`), citizenship (`CIT`) and nativity (`native_born`), whose share of the SHAP mass is monitored.
- **Population filter and target** follow the published Folktables task definition; categorical variables are one-hot encoded and the state indicator is dropped for single-state runs.
- **Size:** up to 150,000 sampled individuals (`max_rows`), split 60/20/20.
- **Training:** all inputs except the sensitive attribute are standardised, 1,500 full-batch iterations per candidate, fairness weight selected from the grid with 95% confidence as described above; `acs_baseline.yaml` is the same model with $`\lambda = 0`$.

Please cite the task as: Ding, F., Hardt, M., Miller, J. and Schmidt, L. (2021). Retiring Adult: New Datasets for Fair Machine Learning. *Advances in Neural Information Processing Systems 34*. The data contain information about real people; no attempt may be made to re-identify individuals.

## Using your own data

Set `data.source` to `file`, point `data.path` to an `.xlsx` or `.csv` table, and name the `target` and `sensitive` columns. List any suspected proxies under `proxy_features` and any attributes the model must not use as inputs under `model.blind_features`.

## Limitations

The default configuration uses synthetic data; the ACS configuration uses real survey microdata from one US state and year, so its results describe that population and should not be generalised to European public services without validation on European data. The exact SHAP explainer applies to the linear model; non-linear or generative models need a model-agnostic explainer or the counterfactual regulariser of CER-GAI (§8.4). When historical labels are themselves biased, demographic parity and equalized odds pull in opposite directions; raising `historical_bias` in the configuration makes this trade-off visible, and the gate will then block deployment rather than hide it. Passing the gate supports, but does not by itself constitute, a conformity assessment under the AI Act.

## Licence

MIT. See `LICENSE`.
