from load_info import extract_experiment_features
from network_structure import model_definition  # kept if used elsewhere
import argparse
import json
import pickle
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# -----------------------------
# Friedman test utilities (Holm kept only for reporting; NOT used for highlighting)
# -----------------------------
from scipy.stats import friedmanchisquare, rankdata, norm

EXPERIMENT_DIRECTION = "early_stopping_preact_KPA_loglik"
CND_TYPE = "PMF"
# early_stopping_preact_KPA_loglik early_stopping_variab_no_early_stopping_preact_fin_no_mil

def friedman_holm_on_ranks_from_pivot(
    pivot: pd.DataFrame,
    baseline_col: str,
    alpha: float = 0.05,
) -> dict:
    """
    Run Friedman test on treatments (columns). Compute average ranks and, optionally,
    Holm-adjusted z-tests vs. baseline for diagnostics only.

    IMPORTANT: The caller should NOT rely on 'best_method' for presentation.
    Highlighting must be decided solely by: (i) Friedman significant, and
    (ii) highest mean accuracy across seeds.
    """
    if baseline_col not in pivot.columns:
        return {"ok": False, "reason": f"Baseline '{baseline_col}' not found."}

    pivot = pivot.dropna(axis=0, how="any")
    n, k = pivot.shape
    if n < 2 or k < 2:
        return {"ok": False, "reason": f"Not enough data (n={n}, k={k})."}

    methods = list(pivot.columns)

    # Friedman on raw columns (treatments)
    args = [pivot[m].to_numpy() for m in methods]
    fr_stat, fr_p = friedmanchisquare(*args)

    # Rank per seed: 1 = best, so rank on negative to get descending by score
    rank_mat = np.vstack([rankdata(-row.values, method="average") for _, row in pivot.iterrows()])
    avg_ranks = rank_mat.mean(axis=0)
    avg_rank_map = {m: r for m, r in zip(methods, avg_ranks)}

    # z-tests on average-rank differences vs baseline (Holm-adjusted) — diagnostics only
    SE = np.sqrt(k * (k + 1) / (6.0 * n))
    tests = []
    for m in methods:
        if m == baseline_col:
            continue
        diff = avg_rank_map[baseline_col] - avg_rank_map[m]  # >0 => method m better rank than baseline
        z = diff / SE
        p = 2 * (1 - norm.cdf(abs(z)))
        tests.append((m, p, diff, avg_rank_map[m]))

    tests.sort(key=lambda t: (np.nan_to_num(t[1], nan=1.0)))
    m_tests = len(tests)
    holm_results = {}
    stop = False
    for i, (mth, p, diff, alt_avg_rank) in enumerate(tests, start=1):
        thr = alpha / (m_tests - i + 1)
        sig = (p <= thr) and not stop and not np.isnan(p)
        if not sig and not stop:
            stop = True
        holm_results[mth] = {
            "method": mth,
            "p_value": p,
            "significant": sig if not stop else False,
            "rank_diff": diff,        # baseline_avg_rank - alt_avg_rank (>0 => alternative better)
            "avg_rank": alt_avg_rank,
        }

    # Pack a friendly table (diagnostics)
    holm_table = pd.DataFrame([holm_results[m] for m in holm_results]).sort_values("p_value", na_position="last")

    return {
        "ok": True,
        "n": n,
        "k": k,
        "methods": methods,
        "avg_ranks": avg_ranks,
        "friedman_stat": fr_stat,
        "friedman_p": fr_p,
        "holm_table": holm_table,
    }


def holm_results_to_latex(group_key, methods, avg_ranks, friedman_stat, friedman_p, holm_table, baseline_col):
    """
    LaTeX table for one (dataset, noise_type, lnr) experiment.
    Uses .format() and doubles braces to avoid f-string brace issues in LaTeX.
    """
    import re as _re
    import numpy as _np
    import pandas as _pd

    gname = " – ".join([str(g) for g in (group_key if isinstance(group_key, tuple) else (group_key,))])
    safe_label = _re.sub(r"[^A-Za-z0-9]+", "_", gname)

    # rows: baseline first, then others (sorted by p)
    rows = [{
        "method": baseline_col,
        "p_value": _np.nan,
        "significant": False,
        "rank_diff": 0.0,
        "avg_rank": {m: r for m, r in zip(methods, avg_ranks)}.get(baseline_col, _np.nan),
    }]
    others = holm_table.sort_values("p_value", na_position="last").to_dict(orient="records")
    rows.extend(others)

    rank_map = {m: r for m, r in zip(methods, avg_ranks)}

    header = (
        "\\begin{{table}}[t]\n"
        "\\centering\n"
        "\\caption{{Friedman + Holm on ranks (baseline={}) for {}}}\n"
        "\\label{{tab:holm_{}}}\n"
        "\\begin{{tabular}}{{lrrrr}}\n"
        "\\hline\n"
        "Metric & Avg. rank $\\downarrow$ & $p$ (vs. baseline) & Holm sig. & Rank diff \\\\\n"
        "\\hline\n"
    ).format(baseline_col, gname, safe_label)

    body_lines = []
    for r in rows:
        m = r["method"]
        p = r.get("p_value", np.nan)
        sig = r.get("significant", False)
        rd = r.get("rank_diff", 0.0)
        ar = rank_map.get(m, np.nan)
        p_str = ("{:.4g}".format(p) if pd.notnull(p) else "--")
        sig_str = "\\checkmark" if sig else "--"
        body_lines.append("{} & {:.3f} & {} & {} & {:.4f} \\\\".format(m, ar, p_str, sig_str, rd))

    footer = (
        "\\hline\n"
        "\\multicolumn{{5}}{{l}}{{Friedman $\\chi^2$={:.3f}, $p$={:.4g}}} \\\\\n"
        "\\hline\n"
        "\\end{{tabular}}\n"
        "\\end{{table}}\n"
    ).format(friedman_stat, friedman_p)

    table_tex = header + "\n".join(body_lines) + "\n" + footer
    table_tex = _re.sub(r"(?<!\\)_", r"\\_", table_tex)
    return table_tex


# -----------------------------
# CLI & I/O helpers
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate early stopping proxies (PC, KPA, CND) over various window sizes and patiences"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("models") / EXPERIMENT_DIRECTION,
        help="Base directory containing experiment subfolders",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[5],
        help="List of window sizes to evaluate",
    )
    parser.add_argument(
        "--patiences",
        nargs="+",
        type=int,
        default=[10],
        help="List of patiences to evaluate",
    )
    parser.add_argument(
        "--cnd-type",
        type=str,
        default=CND_TYPE,
        help="Type of CND proxy to use",
    )
    parser.add_argument(
        "--cnd-percentile",
        default=90.0,
        help="Percentile (0-100) for selecting the CND quantile (e.g., 25 for the first quartile)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device identifier (e.g. 'cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--baseline-metric",
        type=str,
        default="PC",
        help="Metric name to use as the baseline/control for Holm (e.g., 'PC', 'KPA', 'PMF').",
    )
    return parser.parse_args()


def load_experiment_data(base_dir: str) -> dict:
    experiment_data = {}
    experiments = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for experiment in experiments:
        experiment_path = os.path.join(base_dir, experiment)
        args_path = os.path.join(experiment_path, "args.json")
        performances_path = os.path.join(experiment_path, "performances.pkl")
        experiment_data[experiment] = {"args": None, "performances": None}
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                try:
                    experiment_data[experiment]["args"] = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {args_path}")
        if os.path.exists(performances_path):
            with open(performformances_path := performances_path, "rb") as f:
                try:
                    experiment_data[experiment]["performances"] = pickle.load(f)
                except pickle.UnpicklingError:
                    print(f"Error unpickling file {performances_path}")
    return experiment_data


def filter_experiments(data: dict, filters: set) -> dict:
    if not filters:
        return data
    filtered_data = {}
    for experiment, details in data.items():
        if details.get("args"):
            include = all(details["args"].get(key) == value for key, value in filters)
            if include:
                filtered_data[experiment] = details
    return filtered_data


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


# -----------------------------
# Main
# -----------------------------
def main():
    # If the code is run in a subfolder named "program", go one level up
    if "program" in os.getcwd():
        os.chdir("..")

    ensure_results_dir()

    # Parse args
    global args
    args = parse_args()

    # Load data
    data = load_experiment_data(args.base_dir)

    # Example filters (keep or remove as needed)
    filters = {("expand_dataset", True)}
    filtered_data = filter_experiments(data, filters) if filters else data

    device = torch.device(args.device)

    # Collect features from runs
    results_list = []
    for experiment, details in filtered_data.items():
        results_list.extend(
            extract_experiment_features(
                experiment,
                details,
                args.windows,
                args.cnd_type,
                args.patiences,
                device,
                args.cnd_percentile,
            )
        )

    # DataFrame with all results
    df_results = pd.DataFrame(results_list)

    # Add total number of epochs (computed from test_acc length)
    df_results["total_epochs"] = df_results["experiment"].map(
        lambda exp: len(filtered_data[exp]["performances"].test_acc) - 1 if exp in filtered_data else "N/A"
    )

    # Rename columns for brevity
    df_results = df_results.rename(
        columns={
            "best_epoch": "B_E",
            "memorization_epoch": "M_E",
            "best_test_epoch": "BT_E",
            "delta_test_acc": "ΔT_A",
            "delta_best_epoch": "ΔB_E",
            "pearson_corr": "P_C",
            "total_epochs": "T_E",
            "test_acc": "T_A",
            "ea_test_acc": "ea_T_A",
            "delta_ea_test_acc": "Δea_T_A",
        }
    )

    # Keep ΔT_A numeric in %
    df_results["ΔT_A"] = df_results["ΔT_A"] * 100

    # Numeric copy for stats
    df_results["label_noise_ratio_num"] = df_results["label_noise_ratio"].astype(float)

    # -------------------------
    # 1) Excel summary pivots
    # -------------------------
    pivot_table_excel = df_results.pivot_table(
        index=["dataset", "lr", "noise_type", "label_noise_ratio", "experiment", "seed", "metric", "patience"],
        values=["ea_T_A", "Δea_T_A", "ΔB_E", "P_C"],
        aggfunc=np.mean,
    )
    pivot_table_excel.to_excel(
        f"results/{EXPERIMENT_DIRECTION}_all_experiments_window{int(args.windows[0])}_patience{int(args.patiences[0])}_cnd{args.cnd_percentile}.xlsx",
        engine="openpyxl",
    )

    pivot_table_excel = df_results.pivot_table(
        index=["dataset", "noise_type", "label_noise_ratio", "metric", "patience"],
        values=["ea_T_A", "Δea_T_A", "ΔB_E", "P_C"],
        aggfunc=np.mean,
    )
    pivot_table_excel.to_excel(
        f"results/{EXPERIMENT_DIRECTION}_summary_window{int(args.windows[0])}_patience{int(args.patiences[0])}_cnd{args.cnd_percentile}.xlsx",
        engine="openpyxl",
    )

    # -------------------------
    # 2) Scale % columns for presentation
    # -------------------------
    df_results["ea_T_A"] = df_results["ea_T_A"].apply(lambda x: x * 100)
    df_results["Δea_T_A"] = df_results["Δea_T_A"].apply(lambda x: x * 100)

    # -----------------------------------------
    # Friedman PER EXPERIMENT (seeds = blocks)
    # Highlighting rule:
    #   Bold ONLY if Friedman test is significant (p <= 0.05),
    #   and bold the metric(s) with the highest mean ea_T_A across seeds.
    #   No post-hoc/ Holm for highlighting (Holm kept only for diagnostics output).
    # -----------------------------------------
    method_col = "preprocess" if "preprocess" in df_results.columns else "metric"
    score_col = "ea_T_A"  # already scaled to %
    baseline_metric = args.baseline_metric
    dataset_list = df_results["dataset"].unique()

    per_experiment_rows = []
    per_experiment_latex_chunks = []
    experiment_winners = {}

    for dataset in dataset_list:
        df_ds = df_results.query("dataset == @dataset").copy()

        for (nt, lnr) in sorted(
            df_ds[["noise_type", "label_noise_ratio_num"]].drop_duplicates().itertuples(index=False, name=None)
        ):
            df_exp = df_ds.query("noise_type == @nt and label_noise_ratio_num == @lnr").copy()

            # Long tidy: average duplicates within (seed, metric) if any
            df_long = df_exp.groupby(["seed", method_col], as_index=False)[score_col].mean()

            # Require all metrics present for each seed
            presence = df_long.groupby(["seed", method_col]).size().reset_index(name="cnt")
            if presence.empty:
                continue
            wide_presence = presence.pivot_table(index="seed", columns=method_col, values="cnt", fill_value=0, aggfunc="sum")
            complete_seeds = wide_presence.index[(wide_presence > 0).all(axis=1)]
            if len(complete_seeds) < 2:
                continue

            pivot = (
                df_long[df_long["seed"].isin(complete_seeds)]
                .pivot(index="seed", columns=method_col, values=score_col)
            )

            if baseline_metric not in pivot.columns:
                # You can still run Friedman without baseline, but our diagnostic helper expects it.
                # Skip diagnostics; compute Friedman directly here for the highlight rule.
                methods = list(pivot.columns)
                args_fr = [pivot[m].to_numpy() for m in methods]
                fr_stat, fr_p = friedmanchisquare(*args_fr)
                res = {
                    "ok": True,
                    "n": pivot.shape[0],
                    "k": pivot.shape[1],
                    "methods": methods,
                    "avg_ranks": np.vstack([rankdata(-row.values, method="average") for _, row in pivot.iterrows()]).mean(axis=0),
                    "friedman_stat": fr_stat,
                    "friedman_p": fr_p,
                    "holm_table": pd.DataFrame([]),
                }
            else:
                res = friedman_holm_on_ranks_from_pivot(
                    pivot=pivot,
                    baseline_col=baseline_metric,
                    alpha=0.05,
                )
                if not res.get("ok", False):
                    continue

            # Decide highlight winners strictly by rule (no post-hoc)
            alpha_val = 0.05
            fr_pass = (res["friedman_p"] <= alpha_val)

            means_by_metric = (
                df_long[df_long["seed"].isin(complete_seeds)]
                .groupby(method_col, as_index=True)[score_col]
                .mean()
            )

            if fr_pass and not means_by_metric.empty:
                max_mean = means_by_metric.max()
                best_by_acc = set(means_by_metric.index[means_by_metric.eq(max_mean)].tolist())
            else:
                best_by_acc = set()

            experiment_winners[(dataset, nt, lnr)] = best_by_acc

            # Persist per-experiment stats (record what we highlighted)
            row = {
                "dataset": dataset,
                "noise_type": nt,
                "label_noise_ratio_num": lnr,
                "n_seeds": res["n"],
                "k_metrics": res["k"],
                "friedman_stat": res["friedman_stat"],
                "friedman_p": res["friedman_p"],
                "highlighted_metric_highest_acc_if_Friedman": ", ".join(sorted(best_by_acc)) if best_by_acc else "",
                "baseline_metric_available": baseline_metric in pivot.columns,
            }
            for m, r in zip(res["methods"], res["avg_ranks"]):
                row[f"avg_rank_{m}"] = r
            per_experiment_rows.append(row)

            # Optional diagnostics LaTeX (Holm table) — still useful to inspect, but not used for highlighting
            if baseline_metric in pivot.columns:
                group_key = (dataset, nt, f"noise={lnr}")
                per_experiment_latex_chunks.append(
                    holm_results_to_latex(
                        group_key=group_key,
                        methods=res["methods"],
                        avg_ranks=res["avg_ranks"],
                        friedman_stat=res["friedman_stat"],
                        friedman_p=res["friedman_p"],
                        holm_table=res["holm_table"],
                        baseline_col=baseline_metric,
                    )
                )

    # Persist per-experiment stats (Holm diagnostics + highlight record)
    if per_experiment_rows:
        per_exp_df = pd.DataFrame(per_experiment_rows)
        out_xlsx = f"results/{EXPERIMENT_DIRECTION}_friedman_per_experiment_window{int(args.windows[0])}_patience{int(args.patiences[0])}_cnd{args.cnd_percentile}_baseline{args.baseline_metric}.xlsx"
        per_exp_df.to_excel(out_xlsx, index=False)
        out_tex = f"results/{EXPERIMENT_DIRECTION}_friedman_holm_diagnostics_per_experiment_window{int(args.windows[0])}_patience{int(args.patiences[0])}_cnd{args.cnd_percentile}_baseline{args.baseline_metric}.tex"
        if per_experiment_latex_chunks:
            with open(out_tex, "w") as f:
                f.write("\n\n".join(per_experiment_latex_chunks))
            print(f"Diagnostics (Holm) per-experiment LaTeX saved to {out_tex}")
        print(f"Per-experiment results saved to {out_xlsx}")
    else:
        print("No experiments with sufficient complete data for Friedman.")

    # -----------------------------------------
    # Now format label_noise_ratio for presentation (AFTER stats)
    # -----------------------------------------
    df_results["label_noise_ratio"] = df_results["label_noise_ratio"].apply(lambda x: f"{x*100:.4g}\\%")

    # -----------------------------------------
    # LaTeX cleaning helper
    # -----------------------------------------
    def clean_latex_text(latex_table: str) -> str:
        latex_table = latex_table.replace("Δ", r"\Delta ")
        latex_table = re.sub(r"(?<!\\)_", " ", latex_table)
        return latex_table

    # -----------------------------------------
    # Presentation tables (mean ± std) per dataset, bold winners by the new rule
    # -----------------------------------------
    for dataset in dataset_list:
        df_ds = df_results.query("dataset == @dataset")

        pivot_table_pres = df_ds.pivot_table(
            index=["noise_type", "label_noise_ratio", "metric"],
            values=["ea_T_A", "Δea_T_A", "B_E", "ΔB_E"],
            aggfunc=[np.mean, np.std],
        )

        pivot_table_pres[("ΔB_E", "mean ± std")] = pivot_table_pres.apply(
            lambda row: f"{row[('mean','ΔB_E')]:.2f} ± {row[('std','ΔB_E')]:.2f}"
            if ("std", "ΔB_E") in row else f"{row[('mean','ΔB_E')]:.2f} ± N/A",
            axis=1,
        )
        pivot_table_pres[("Δea_T_A", "mean ± std")] = pivot_table_pres.apply(
            lambda row: f"{row[('mean','Δea_T_A')]:.2f}\\% ± {row[('std','Δea_T_A')]:.2f}\\%"
            if ("std", "Δea_T_A") in row else f"{row[('mean','Δea_T_A')]:.2f}\\% ± N/A",
            axis=1,
        )
        pivot_table_pres[("B_E", "mean ± std")] = pivot_table_pres.apply(
            lambda row: f"{row[('mean','B_E')]:.2f} ± {row[('std','B_E')]:.2f}"
            if ("std", "B_E") in row else f"{row[('mean','B_E')]:.2f} ± N/A",
            axis=1,
        )
        pivot_table_pres[("ea_T_A", "mean ± std")] = pivot_table_pres.apply(
            lambda row: f"{row[('mean','ea_T_A')]:.2f}\\% ± {row[('std','ea_T_A')]:.2f}\\%"
            if ("std", "ea_T_A") in row else f"{row[('mean','ea_T_A')]:.2f}\\% ± N/A",
            axis=1,
        )

        pivot_table_pres = pivot_table_pres[
            [("B_E", "mean ± std"), ("ΔB_E", "mean ± std"), ("ea_T_A", "mean ± std"), ("Δea_T_A", "mean ± std")]
        ]

        # Bold the winner metric(s) per (noise_type, label_noise_ratio) experiment
        new_index = []
        for (nt, lnr_str, met) in pivot_table_pres.index:
            try:
                lnr_num = float(lnr_str.replace("\\%", "")) / 100.0
            except Exception:
                lnr_num = None
            bold = False
            if lnr_num is not None:
                winners = experiment_winners.get((dataset, nt, lnr_num), set())
                if met in winners:
                    bold = True
            met_lbl = f"\\textbf{{{met}}}" if bold else met
            new_index.append((nt, lnr_str, met_lbl))
        pivot_table_pres.index = pd.MultiIndex.from_tuples(new_index, names=pivot_table_pres.index.names)
        pivot_table_pres.columns.set_names(["", ""], inplace=True)

        latex_table = pivot_table_pres.to_latex(escape=False, multirow=True)
        latex_table = clean_latex_text(latex_table)

        output_filename = (
            f"results/{EXPERIMENT_DIRECTION}_{dataset}_early_stopping_results_window"
            f"{int(args.windows[0])}_patience{int(args.patiences[0])}_cnd{args.cnd_percentile}_baseline{args.baseline_metric}.tex"
        )
        with open(output_filename, "w") as f:
            f.write(latex_table)

        print(f"LaTeX table successfully generated and saved to {output_filename}!")


if __name__ == "__main__":
    args = parse_args()
    main()
