import numpy as np
import pandas as pd
import re
from scipy.stats import friedmanchisquare, rankdata



def _nemenyi_qalpha(k: int, alpha: float = 0.05) -> float:
    # Critical values for two-sided Nemenyi at alpha=0.05 (studentized range for infinite df)
    table_005 = {2:1.960, 3:2.343, 4:2.569, 5:2.728, 6:2.850, 7:2.949, 8:3.031, 9:3.102, 10:3.164}
    if alpha != 0.05:
        # conservative fallback if other alphas are requested
        return table_005.get(k, table_005[10])
    return table_005.get(k, table_005[10])

def friedman_nemenyi_from_long(df: pd.DataFrame,
                               method_col: str,
                               score_col: str,
                               scenario_cols: list,
                               alpha: float = 0.05) -> dict:
    # Build a complete matrix: rows=scenarios, cols=methods
    pivot = df.pivot_table(index=scenario_cols, columns=method_col, values=score_col, aggfunc='mean')
    pivot = pivot.dropna(axis=0, how='any')  # only scenarios where all methods exist
    n, k = pivot.shape
    if n < 2 or k < 2:
        return {"ok": False, "reason": f"Not enough data (n={n}, k={k})."}

    # Friedman
    args = [pivot.iloc[:, j].to_numpy() for j in range(k)]
    fr_stat, fr_p = friedmanchisquare(*args)

    # Ranks per scenario (higher accuracy = better -> we invert ranks of losses; for accuracy, we rank descending)
    # rankdata gives 1=lowest; we want 1=best -> rank on negative values
    ranks = np.vstack([rankdata(-row.values, method='average') for _, row in pivot.iterrows()])
    avg_ranks = ranks.mean(axis=0)
    methods = list(pivot.columns)

    # Nemenyi CD
    q_alpha = _nemenyi_qalpha(k, alpha)
    CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    # Significant pairs
    sig_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(avg_ranks[i] - avg_ranks[j]) > CD:
                sig_pairs.append((methods[i], methods[j]))

    # Identify best (lowest average rank number = best)
    best_idx = int(np.argmin(avg_ranks))
    best_method = methods[best_idx]
    best_is_strict = all((best_method, m) in sig_pairs or (m, best_method) in sig_pairs
                         for m in methods if m != best_method)

    return {
        "ok": True,
        "n": n, "k": k,
        "friedman_stat": fr_stat, "friedman_p": fr_p,
        "methods": methods, "avg_ranks": avg_ranks,
        "CD": CD, "sig_pairs": sig_pairs,
        "best_method": best_method if best_is_strict else None
    }

def results_to_latex_table(group_key, methods, avg_ranks, CD, sig_pairs, best_method):
    # Build a compact LaTeX table with avg ranks and a marker for pairs beyond CD
    # group_key may be a tuple; stringify safely
    gname = " – ".join([str(g) for g in (group_key if isinstance(group_key, tuple) else (group_key,))])

    header = r"""\begin{table}[t]
    \centering
    \caption{Friedman–Nemenyi over preprocessing methods for %s}
    \label{tab:nemenyi_%s}
    \begin{tabular}{lrr}
    \hline
    Method & Avg. rank $\downarrow$ & Significant vs. \\
    \hline
    """ % (gname, re.sub(r'[^A-Za-z0-9]+', '_', gname))

    rows = []
    for i, m in enumerate(methods):
        rivals = []
        for a, b in sig_pairs:
            if a == m:
                rivals.append(b)
            elif b == m:
                rivals.append(a)
        mark = r"\textbf{%s}" % m if best_method == m else m
        rows.append(f"{mark} & {avg_ranks[i]:.3f} & {', '.join(rivals) if rivals else '--'} \\\\")
    footer = r"""\hline
    \multicolumn{3}{l}{Critical difference (CD): %.3f} \\
    \hline
    \end{tabular}
    \end{table}
    """ % CD
    table = header + "\n".join(rows) + "\n" + footer
    # minimal cleaning to avoid LaTeX issues with underscores outside math
    table = re.sub(r'(?<!\\)_', r'\_', table)
    return table
