"""Aggregate runs/<exp>/<seed>/metrics.jsonl across seeds and render the
Phase 0 results table into RESULTS.md, between the
<!-- PHASE0_TABLE_START --> / <!-- PHASE0_TABLE_END --> markers.

Usage: python scripts/make_results_table.py
"""
import glob
import json
import os
import statistics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
RESULTS_MD = os.path.join(REPO_ROOT, "RESULTS.md")

SEEDS = [0, 1, 2]
CONDITIONS = [1, 2]

# (paper_mean_pct, paper_std_pct, tolerance_pp, has_gate)
REFERENCE = {
    ("ltn", 1): (80.54, 23.33, None, False),  # A-NeSI Table 1's own LTN row; no formal Phase 0 gate
    ("ltn", 2): (77.54, 35.55, None, False),
    ("deepsoftlog", 1): (98.4, 0.1, 1.0, True),
    ("deepsoftlog", 2): (96.6, 0.3, 1.0, True),
    ("anesi", 1): (97.66, 0.21, 1.0, True),
    ("anesi", 2): (95.96, 0.38, 1.0, True),
}

METHOD_LABELS = {
    "ltn": "LTN (LTNtorch, standard)",
    "deepsoftlog": "DeepSoftLog (공식 코드)",
    "anesi": "A-NeSI (predict, 공식 코드)",
}


def _read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_ltn(metrics_path):
    for rec in reversed(_read_jsonl(metrics_path)):
        if rec.get("epoch") == "final":
            return rec["test_sum_accuracy"] * 100
    return None


def extract_deepsoftlog(metrics_path):
    for rec in _read_jsonl(metrics_path):
        if rec.get("split") == "test":
            return rec["threshold_accuracy"] * 100
    return None


def extract_anesi(metrics_path):
    last = None
    for rec in _read_jsonl(metrics_path):
        if rec.get("split") == "test":
            last = rec["accuracy"] * 100
    return last


EXTRACTORS = {"ltn": extract_ltn, "deepsoftlog": extract_deepsoftlog, "anesi": extract_anesi}


def collect(method, n):
    values = []
    missing = []
    for seed in SEEDS:
        metrics_path = os.path.join(RUNS_DIR, f"{method}_n{n}" if method != "ltn" else f"ltn_standard_n{n}", str(seed), "metrics.jsonl")
        if not os.path.exists(metrics_path):
            missing.append(seed)
            continue
        v = EXTRACTORS[method](metrics_path)
        if v is None:
            missing.append(seed)
        else:
            values.append(v)
    return values, missing


def fmt_mean_std(values):
    if not values:
        return "—"
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f} ± {std:.2f}"


def main():
    rows = []
    for method in ("ltn", "deepsoftlog", "anesi"):
        for n in CONDITIONS:
            values, missing = collect(method, n)
            paper_mean, paper_std, tol, has_gate = REFERENCE[(method, n)]
            paper_str = f"{paper_mean:.2f} ± {paper_std:.2f}"
            repro_str = fmt_mean_std(values)
            if not values:
                verdict = "미실행" + (f" (seed {missing} 누락)" if missing else "")
            elif has_gate:
                mean = statistics.mean(values)
                verdict = "PASS" if abs(mean - paper_mean) <= tol else "FAIL"
                if missing:
                    verdict += f" (seed {missing} 누락)"
            else:
                verdict = "— (gate 없음, 참고용)"
                if missing:
                    verdict += f" / seed {missing} 누락"
            rows.append((METHOD_LABELS[method], n, paper_str, repro_str, verdict))

    table_lines = [
        "| 방법 | N | 논문 수치 | 재현 수치 | 통과 여부 |",
        "|---|---|---|---|---|",
    ]
    for label, n, paper_str, repro_str, verdict in rows:
        table_lines.append(f"| {label} | {n} | {paper_str} | {repro_str} | {verdict} |")
    table_md = "\n".join(table_lines)

    with open(RESULTS_MD) as f:
        content = f.read()

    start_marker = "<!-- PHASE0_TABLE_START -->"
    end_marker = "<!-- PHASE0_TABLE_END -->"
    if start_marker in content and end_marker in content:
        pre, rest = content.split(start_marker, 1)
        _, post = rest.split(end_marker, 1)
        content = pre + start_marker + "\n" + table_md + "\n" + end_marker + post
    else:
        content += f"\n\n{start_marker}\n{table_md}\n{end_marker}\n"

    with open(RESULTS_MD, "w") as f:
        f.write(content)

    print(table_md)


if __name__ == "__main__":
    main()
