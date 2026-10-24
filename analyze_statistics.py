import json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

def analyze_results(json_path):
    print(f"Leyendo resultados desde: {json_path}\n")
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    
    print("| Contrato | Métrica | CancerGA (Media ± STD) | SSA (Media ± STD) | p-valor (Wilcoxon) |")
    print("|---|---|---|---|---|")

    for c in data["contracts"]:
        if "baseline_ssa" not in c:
            continue
            
        label = c["label"]
        cancer_runs = c["runs"]
        ssa_runs = c["baseline_ssa"]["runs"]
        
        # Extraer listas de 30 valores para cada método
        def get_vals(runs, method):
            oil, pvi, hv, spacing = [], [], [], []
            for r in runs:
                if method == "cancer":
                    oil.append(r["representative"]["oil"])
                    pvi.append(r["representative"]["pvi"])
                else:
                    # Para SSA usamos el 'knee' que definimos en el arreglo anterior
                    oil.append(r["representatives"]["knee"]["oil"])
                    pvi.append(r["representatives"]["knee"]["pvi"])
                
                hv.append(r["hv_oil_pvi"])
                sp = r["spacing"]["spacing"] if r["spacing"] else np.nan
                spacing.append(sp if sp is not None else np.nan)
            return np.array(oil), np.array(pvi), np.array(hv), np.array(spacing)

        c_oil, c_pvi, c_hv, c_sp = get_vals(cancer_runs, "cancer")
        s_oil, s_pvi, s_hv, s_sp = get_vals(ssa_runs, "ssa")

        metrics = [
            ("Aceite (t)", c_oil, s_oil),
            ("PVI (t/día)", c_pvi, s_pvi),
            ("HV (aceite, PVI)", c_hv, s_hv),
            ("Spacing", c_sp, s_sp)
        ]

        for name, c_arr, s_arr in metrics:
            # Limpiar NaNs para Wilcoxon (por si hay frentes degenerados)
            mask = ~np.isnan(c_arr) & ~np.isnan(s_arr)
            c_clean, s_clean = c_arr[mask], s_arr[mask]

            if len(c_clean) > 0:
                try:
                    # Wilcoxon bilateral
                    stat, p_val = wilcoxon(c_clean, s_clean)
                    p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
                    if p_val >= 0.05:
                        p_str += " (n.s.)"
                except ValueError:
                    p_str = "Error (datos idénticos)"
            else:
                p_str = "N/A"

            c_mean, c_std = np.nanmean(c_arr), np.nanstd(c_arr)
            s_mean, s_std = np.nanmean(s_arr), np.nanstd(s_arr)

            print(f"| {label} | {name} | {c_mean:,.2f} ± {c_std:,.2f} | {s_mean:,.2f} ± {s_std:,.2f} | {p_str} |")

if __name__ == "__main__":
    # Ajusta la ruta si es necesario
    json_path = "results/final_paper_30s/results.json"
    analyze_results(json_path)