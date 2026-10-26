"""
Análisis estadístico CancerGA vs SSA a partir de un results.json.

Metodología
-----------
- Cada semilla es una corrida independiente sobre la MISMA instancia, y el índice de semilla no empareja
  nada entre métodos: se usa la prueba U de Mann-Whitney (muestras independientes), no Wilcoxon pareado.
- Se hacen contratos x métricas pruebas: se aplica la corrección de Holm (control del error de familia).
- Tamaño de efecto: A12 de Vargha-Delaney ORIENTADO = probabilidad de que una corrida de CancerGA sea
  MEJOR que una de SSA (0.5 = sin efecto; >0.5 favorece a CancerGA), con la dirección de "mejor" propia de
  cada métrica (más aceite y más HV son mejores; menos PVI y menos espaciado son mejores).
- Desviaciones estándar muestrales (ddof=1), igual que el resto del repo.
- Sólo depende de numpy y scipy.

Uso:
    python analyze_statistics.py results/results.json [--alpha 0.05] [--csv tabla.csv]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

# (nombre, extractor CancerGA, extractor SSA, ¿mayor es mejor?)
METRICS = [
    ("Producción (L)", lambda r: r["representative"]["oil"], lambda r: r["representatives"]["knee"]["oil"], True),
    ("PVI (t/día)", lambda r: r["representative"]["pvi"], lambda r: r["representatives"]["knee"]["pvi"], False),
    ("HV (aceite, PVI)", lambda r: r["hv_oil_pvi"], lambda r: r["hv_oil_pvi"], True),
    ("Spacing", lambda r: r["spacing"]["spacing"], lambda r: r["spacing"]["spacing"], False),
]


def a12_oriented(x, y, higher_is_better=True):
    """P(x es mejor que y) + 0.5 * P(empate), con x e y muestras independientes (Vargha-Delaney A12)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not higher_is_better:
        x, y = -x, -y
    better = (x[:, None] > y[None, :]).sum()
    ties = (x[:, None] == y[None, :]).sum()
    return float((better + 0.5 * ties) / (len(x) * len(y)))


def holm_adjust(p_values):
    """P-valores ajustados por Holm-Bonferroni (step-down), en el orden original."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def _mean_std(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def collect_rows(data):
    """Una fila por (contrato, métrica) con estadísticos, p-valor crudo y A12 orientado."""
    rows = []
    for c in data["contracts"]:
        if "baseline_ssa" not in c:
            continue
        c_runs, s_runs = c["runs"], c["baseline_ssa"]["runs"]
        for name, get_c, get_s, higher in METRICS:
            c_arr = [get_c(r) for r in c_runs]
            s_arr = [get_s(r) for r in s_runs]
            # el espaciado es None cuando el frente tiene < 3 puntos
            c_arr = [v for v in c_arr if v is not None]
            s_arr = [v for v in s_arr if v is not None]
            if len(c_arr) < 2 or len(s_arr) < 2:
                continue
            p_val = float(mannwhitneyu(c_arr, s_arr, alternative="two-sided").pvalue)
            if np.isnan(p_val):  # muestras constantes e idénticas: sin diferencia detectable
                p_val = 1.0
            c_mean, c_std = _mean_std(c_arr)
            s_mean, s_std = _mean_std(s_arr)
            rows.append(
                {
                    "contract": c["label"],
                    "metric": name,
                    "higher_is_better": higher,
                    "n_cancerga": len(c_arr),
                    "n_ssa": len(s_arr),
                    "cancerga_mean": c_mean,
                    "cancerga_std": c_std,
                    "ssa_mean": s_mean,
                    "ssa_std": s_std,
                    "p_raw": p_val,
                    "a12": a12_oriented(c_arr, s_arr, higher),
                }
            )
    p_adj = holm_adjust([r["p_raw"] for r in rows]) if rows else []
    for r, pa in zip(rows, p_adj):
        r["p_holm"] = pa
    return rows


def _effect_label(a12):
    d = abs(a12 - 0.5)
    size = "despreciable" if d < 0.06 else "pequeño" if d < 0.14 else "mediano" if d < 0.21 else "grande"
    return f"{size}, {'CancerGA' if a12 > 0.5 else 'SSA' if a12 < 0.5 else 'ninguno'}"


def analyze_results(json_path, alpha=0.05, csv_path=None, quiet=False):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows = collect_rows(data)
    for r in rows:
        r["significant"] = r["p_holm"] < alpha

    if not quiet:
        print(f"Resultados: {json_path} | semillas: {len(data['meta']['seeds'])} | alfa (Holm): {alpha}\n")
        print(
            "| Contrato | Métrica | CancerGA (media ± sd) | SSA (media ± sd) | n | p (Holm) | A12 (P[CancerGA mejor]) |"
        )
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            mark = " *" if r["significant"] else " (n.s.)"
            print(
                f"| {r['contract']} | {r['metric']} | {r['cancerga_mean']:,.2f} ± {r['cancerga_std']:,.2f} | "
                f"{r['ssa_mean']:,.2f} ± {r['ssa_std']:,.2f} | {r['n_cancerga']}/{r['n_ssa']} | "
                f"{r['p_holm']:.4f}{mark} | {r['a12']:.3f} ({_effect_label(r['a12'])}) |"
            )
        print("\nA12 > 0.5 favorece a CancerGA en TODAS las métricas (ya está orientado por métrica).")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", help="results.json generado por run_harvest_experiment.py")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--csv", default=None, help="guardar la tabla en CSV")
    a = ap.parse_args()
    analyze_results(a.json_path, alpha=a.alpha, csv_path=a.csv)
