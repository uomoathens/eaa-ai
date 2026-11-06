from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF")
PASS_FILL = PatternFill("solid", fgColor="E2EFDA")
FAIL_FILL = PatternFill("solid", fgColor="FCE4D6")


def _summary_frame(result, verified):
    rows = [
        ("Selected fairness weight (lambda)", result.selected_fairness_weight),
        ("Fairness weights tried", ", ".join(str(c["fairness_weight"]) for c in result.candidates)),
        ("Selected iteration", result.selected_iteration),
        ("Rolled back to last compliant iteration", "Yes" if result.rolled_back else "No"),
        ("Deployable on test set", "Yes" if result.deployable else "No"),
        ("Violations on test set", ", ".join(result.test_compliance["violations"]) or "No violations"),
        ("Hinge penalty on test set", round(result.test_compliance["hinge_penalty"], 6)),
        ("Ledger records", len(result.ledger.records)),
        ("Ledger integrity", "Verified" if verified else "BROKEN"),
        ("Ledger head hash", result.ledger.head),
        ("Accountable role", result.config["actor"]["role"]),
        ("Accountable unit", result.config["actor"]["unit"]),
        ("Human approval", "Required before deployment"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def _requirements_frame(result):
    rows = []
    for name, spec in result.config["policy"].items():
        value = result.test_metrics[name]
        margin = result.test_compliance["margins"][name]
        rows.append(
            {
                "Metric": name,
                "Rule": f"{'<=' if spec['direction'] == 'max' else '>='} {spec['threshold']}",
                "Test value": round(value, 6),
                "95% CI low": round(result.test_intervals[name][0], 6),
                "95% CI high": round(result.test_intervals[name][1], 6),
                "Margin": round(margin, 6),
                "Status": "PASS" if margin >= 0 else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def _selection_frame(result):
    return pd.DataFrame(result.candidates)


def _diagnostics_frame(result):
    d = result.diagnostics
    rows = [
        ("Test set size", d["test_size"], ""),
        ("Share in protected group", d["protected_group_share"], ""),
        ("Observed positive rate (all)", d["base_rate"], ""),
        ("Observed positive rate (reference group)", d["base_rate_reference_group"], ""),
        ("Observed positive rate (protected group)", d["base_rate_protected_group"], ""),
        ("Constant predictor accuracy", d["constant_predictor_accuracy"], "Accuracy of always predicting the majority outcome"),
    ]
    for label, key in (("Selected model", "selected_model"), ("Reference model (scikit-learn, unconstrained)", "reference_model")):
        block = d[key]
        rows.extend(
            [
                (f"{label}: accuracy", block["accuracy"], ""),
                (f"{label}: accuracy lift over constant predictor", block["accuracy_lift"], ""),
                (f"{label}: predicted positive rate", block["positive_rate"], ""),
                (f"{label}: predicted positive rate (reference group)", block["positive_rate_reference_group"], ""),
                (f"{label}: predicted positive rate (protected group)", block["positive_rate_protected_group"], ""),
            ]
        )
    rows.append(("Accuracy gap to reference model", d["accuracy_gap_to_reference"], "Large positive gap suggests under-training"))
    return pd.DataFrame(rows, columns=["Diagnostic", "Value", "Note"])


def _history_frame(result):
    frame = pd.DataFrame(result.history)
    frame["violations"] = frame["violations"].apply(lambda items: ", ".join(items))
    return frame.drop(columns=["margins"])


def _ledger_row(record):
    payload = record["payload"]
    return {
        "index": record["index"],
        "timestamp": record["timestamp"],
        "event": record["event"],
        "role": record["actor"].get("role"),
        "unit": record["actor"].get("unit"),
        "fairness_weight": payload.get("fairness_weight", payload.get("selected_fairness_weight")),
        "iteration": payload.get("iteration", payload.get("selected_iteration")),
        "flagged": payload.get("flagged"),
        "parameters_hash": payload.get("parameters_hash", payload.get("reviewed_parameters_hash")),
        "previous_hash": record["previous_hash"],
        "hash": record["hash"],
    }


def _ledger_frame(result):
    return pd.DataFrame([_ledger_row(record) for record in result.ledger.records])


def _importance_frame(result):
    return pd.DataFrame(result.explanation.global_importance(), columns=["Feature", "Mean |SHAP|"])


def _cases_frame(result, test, limit):
    probabilities = result.model.predict_proba(test.features)
    rows = []
    for row in range(min(limit, len(test.labels))):
        top = result.explanation.top_k(row, 3)
        entry = {
            "case": row,
            "decision": "Eligible" if probabilities[row] >= result.model.threshold else "Not eligible",
            "probability": round(float(probabilities[row]), 4),
        }
        for rank, (feature, value) in enumerate(top, start=1):
            entry[f"reason_{rank}"] = feature
            entry[f"contribution_{rank}"] = round(value, 4)
        rows.append(entry)
    return pd.DataFrame(rows)


def _format(sheet):
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.freeze_panes = "A2"
    for index, column in enumerate(sheet.columns, start=1):
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
        sheet.column_dimensions[get_column_letter(index)].width = min(max(12, width + 2), 70)


def write_report(result, test, path, verified, case_limit=200):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheets = {
        "Summary": _summary_frame(result, verified),
        "Requirements": _requirements_frame(result),
        "Model selection": _selection_frame(result),
        "Diagnostics": _diagnostics_frame(result),
        "Training history": _history_frame(result),
        "Audit ledger": _ledger_frame(result),
        "Global explanation": _importance_frame(result),
        "Case explanations": _cases_frame(result, test, case_limit),
    }
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        for name in sheets:
            _format(writer.sheets[name])
        requirements = writer.sheets["Requirements"]
        for row in range(2, requirements.max_row + 1):
            status = requirements.cell(row=row, column=7)
            fill = PASS_FILL if status.value == "PASS" else FAIL_FILL
            for column in range(1, 8):
                requirements.cell(row=row, column=column).fill = fill
    return target


def _review_rows(ledger):
    return [
        {
            "timestamp": record["timestamp"],
            "decision": record["payload"]["decision"],
            "role": record["actor"].get("role"),
            "reviewer": record["actor"].get("unit"),
            "note": record["payload"].get("note", ""),
            "reviewed_parameters_hash": record["payload"]["reviewed_parameters_hash"],
            "reviewed_record_hash": record["payload"]["reviewed_record_hash"],
            "review_record_hash": record["hash"],
        }
        for record in ledger.records
        if record["event"] == "human_review"
    ]


def review_status(ledger):
    reviews = _review_rows(ledger)
    if not reviews:
        return "Required before deployment"
    last = reviews[-1]
    verb = "Approved" if last["decision"] == "approve" else "Rejected"
    return f"{verb} by {last['role']} ({last['reviewer']}) on {last['timestamp'][:19].replace('T', ' ')} UTC"


def annotate_report(path, ledger, verified):
    workbook = load_workbook(path)
    ledger_sheet = workbook["Audit ledger"]
    headers = [cell.value for cell in ledger_sheet[1]]
    existing = {ledger_sheet.cell(row=row, column=1).value for row in range(2, ledger_sheet.max_row + 1)}
    for record in ledger.records:
        if record["index"] not in existing:
            row = _ledger_row(record)
            ledger_sheet.append([row.get(header) for header in headers])
    summary = workbook["Summary"]
    updates = {
        "Ledger records": len(ledger.records),
        "Ledger integrity": "Verified" if verified else "BROKEN",
        "Ledger head hash": ledger.head,
        "Human approval": review_status(ledger),
    }
    for row in range(2, summary.max_row + 1):
        label = summary.cell(row=row, column=1).value
        if label in updates:
            summary.cell(row=row, column=2).value = updates[label]
    if "Human review" in workbook.sheetnames:
        del workbook["Human review"]
    review_sheet = workbook.create_sheet("Human review", 1)
    rows = _review_rows(ledger)
    columns = list(rows[0]) if rows else ["timestamp", "decision", "role", "reviewer", "note"]
    review_sheet.append(columns)
    for entry in rows:
        review_sheet.append([entry[column] for column in columns])
    _format(review_sheet)
    workbook.save(path)
    return path
