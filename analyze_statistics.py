import json
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def a12_fast(x, y):
    """Probability of superiority (Vargha-Delaney A12)."""
    x, y = np.array(x), np.array(y)
    gt = (x[:, None] > y).sum()
    eq = (x[:, None] == y).sum()
    return (gt + 0.5 * eq) / (len(x) * len(y))

def analyze_results(json_path):
    print(f"Leyendo resultados desde: {json_path}\n")
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    
    rows = []
    
    for c in data["contracts"]:
        if "baseline_ssa" not in c: continue
            
        label = c["label"]
        c_runs = c["runs"]
        s_runs = c["baseline_ssa"]["runs"]
        
        # Extraemos las listas de las 30 semillas
        c_oil = [r["representative"]["oil"] for r in c_runs]
        s_oil = [r["representatives"]["knee"]["oil"] for r in s_runs]
        
        c_pvi = [r["representative"]["pvi"] for r in c_runs]
        s_pvi = [r["representatives"]["knee"]["pvi"] for r in s_runs]
        
        c_hv = [r["hv_oil_pvi"] for r in c_runs]
        s_hv = [r["hv_oil_pvi"] for r in s_runs]
        
        c_sp = [r["spacing"]["spacing"] for r in c_runs if r["spacing"]["spacing"] is not None]
        s_sp = [r["spacing"]["spacing"] for r in s_runs if r["spacing"]["spacing"] is not None]

        metrics = [
            ("Producción (L)", c_oil, s_oil),
            ("PVI (t/día)", c_pvi, s_pvi),
            ("HV (aceite, PVI)", c_hv, s_hv),
            ("Spacing", c_sp, s_sp)
        ]

        for name, c_arr, s_arr in metrics:
            if len(c_arr) > 0 and len(s_arr) > 0:
                # Mann-Whitney U para muestras independientes
                stat, p_val = mannwhitneyu(c_arr, s_arr, alternative='two-sided')
                a12 = a12_fast(c_arr, s_arr)
                
                c_mean, c_std = np.mean(c_arr), np.std(c_arr)
                s_mean, s_std = np.mean(s_arr), np.std(s_arr)
                
                rows.append({
                    "contract": label, "metric": name,
                    "c_str": f"{c_mean:,.2f} ± {c_std:,.2f}",
                    "s_str": f"{s_mean:,.2f} ± {s_std:,.2f}",
                    "p_val": p_val, "a12": a12
                })

    # Corrección de Holm para comparaciones múltiples
    p_vals = [r["p_val"] for r in rows]
    reject, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method='holm')

    print("| Contrato | Métrica | CancerGA | SSA | p-valor (Holm) | A12 |")
    print("|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        p_str = f"{p_adj[i]:.4f}" + (" *" if reject[i] else " (n.s.)")
        print(f"| {r['contract']} | {r['metric']} | {r['c_str']} | {r['s_str']} | {p_str} | {r['a12']:.3f} |")

if __name__ == "__main__":
    analyze_results("results/final_paper_30s/results.json")