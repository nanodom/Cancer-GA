"""
Dashboard HTML autocontenido para los resultados de `cancer_ga.experiment`.

Recibe el diccionario de resultados (o el JSON guardado) y produce un único
archivo HTML sin dependencias de red: Chart.js se incrusta en el propio
archivo (ver `utils/vendor/chart.umd.js`), así que funciona offline.

Secciones
---------
1. Comparación entre contratos: SÓLO con indicadores comparables (aceite, PVI
   e hipervolumen en el espacio (aceite, PVI)). f2 se excluye a propósito: el
   paper advierte que f2 no es comparable entre contratos.
2. Detalle por contrato (selector): convergencia real por generación (HV,
   mejor aceite, mínimo f2) con una curva por semilla, frentes de Pareto,
   utilización de recursos, Gantt y tabla por semilla.
"""

import json
from pathlib import Path

_CHARTJS_PATH = Path(__file__).parent / "vendor" / "chart.umd.js"

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CancerGA · Re-planificación de cosecha</title>
<script>__CHARTJS_INLINE__</script>
<style>
  :root {
    --olive-deep: #3a4a2e; --olive: #6b7c4a; --clay: #b3673a;
    --cream: #f4f1e6; --ink: #2a2822; --muted: #7a765f; --line: #d9d4c1;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--cream); color: var(--ink);
         font-family: 'Iowan Old Style', 'Palatino Linotype', Georgia, serif; }
  header { padding: 2.2rem clamp(1.25rem, 5vw, 4rem) 1.5rem; border-bottom: 1px solid var(--line); }
  header h1 { margin: 0 0 .35rem; font-size: clamp(1.5rem, 3vw, 2.1rem); font-weight: 600;
              color: var(--olive-deep); letter-spacing: -.01em; }
  header p { margin: 0; max-width: 70ch; line-height: 1.5; color: #57543f; font-size: .96rem; }
  main { padding: 1.5rem clamp(1.25rem, 5vw, 4rem) 3rem; display: grid; gap: 1.5rem; }
  h2.section { margin: .5rem 0 0; font-size: 1.25rem; color: var(--olive-deep); font-weight: 600; }
  .callout { background: #fbf6e3; border: 1px solid #e3d8a8; border-left: 4px solid var(--clay);
             padding: .9rem 1.1rem; font-size: .9rem; line-height: 1.55; }
  .callout b { color: var(--clay); }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1rem; }
  .card { background: #fff; border: 1px solid var(--line); padding: 1rem 1.2rem; }
  .card .label { font-size: .78rem; color: var(--muted); margin-bottom: .3rem; }
  .card .value { font-family: 'Iosevka','SF Mono',Consolas,monospace; font-size: 1.35rem; color: var(--olive-deep); }
  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  @media (max-width: 900px) { .panels { grid-template-columns: 1fr; } }
  .panel { background: #fff; border: 1px solid var(--line); padding: 1.15rem 1.3rem 1rem; min-width: 0; }
  .panel h3 { margin: 0 0 .8rem; font-size: 1rem; font-weight: 600; color: var(--olive-deep); }
  .panel-full { grid-column: 1 / -1; }
  canvas { max-height: 320px; }
  .note { font-size: .8rem; color: var(--muted); margin-top: .6rem; line-height: 1.5; }
  .scroll-x { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: .88rem; }
  th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { color: var(--muted); font-weight: 500; font-size: .78rem; }
  td.num, th.num { text-align: right; font-family: 'Iosevka','SF Mono',Consolas,monospace; }
  tr:last-child td { border-bottom: none; }
  tr.best td { background: #eef2e4; }
  .swatch { display: inline-block; width: .7rem; height: .7rem; margin-right: .45rem; vertical-align: -1px; }
  .toggle { display: flex; flex-wrap: wrap; gap: .5rem; }
  .toggle button { font-family: inherit; font-size: .85rem; padding: .4rem .85rem; border: 1px solid var(--line);
                   background: #fff; color: var(--ink); cursor: pointer; }
  .toggle button.active { background: var(--olive-deep); color: #fff; border-color: var(--olive-deep); }
  #ganttScroll { overflow-x: auto; padding-bottom: .5rem; }
  #ganttGrid { display: grid; gap: 1px; width: max-content; }
  .g-label { font-size: .72rem; color: #57543f; display: flex; align-items: center; padding-right: .6rem;
             position: sticky; left: 0; background: #fff; white-space: nowrap; }
  .g-cell { width: 13px; height: 20px; background: #efece0; border-radius: 1px; }
  .g-day { font-size: .55rem; color: #a6a189; text-align: center; }
</style>
</head>
<body>

<header>
  <h1>Re-planificación de cosecha con CancerGA</h1>
  <p id="subtitle"></p>
</header>

<main>
  <div class="callout">
    <b>Cómo leer este dashboard.</b> El paper advierte que <b>f2 no es comparable entre tipos de contrato</b>:
    cada contrato mide la discrepancia con otra fórmula y otras unidades (toneladas, hectáreas u horas). Por eso
    la comparación entre contratos usa sólo indicadores que se definen igual para todos:
    la <b>producción de aceite</b>, el <b>PVI</b> (Plan Variability Index: desviación absoluta diaria promedio
    entre el plan nuevo y el base, en toneladas de aceituna por día) y el <b>hipervolumen en el espacio
    (aceite, PVI)</b>. El detalle de f2 se muestra sólo dentro de cada contrato.
  </div>

  <section class="cards" id="cards"></section>

  <h2 class="section">1 · Comparación entre contratos</h2>
  <section class="panels">
    <div class="panel panel-full">
      <h3>Indicadores comparables (media ± desv. estándar entre semillas)</h3>
      <div class="scroll-x"><table id="cmpTable"></table></div>
      <p class="note">Solución evaluada: el punto de compromiso de cada corrida (el más cercano al ideal en el espacio
      normalizado del propio contrato). <b>HV (aceite, PVI)</b> mide, con cotas comunes a todos los contratos, cuánto
      del espacio aceite–estabilidad logra cubrir el archivo final de cada corrida: mayor es mejor.</p>
    </div>
    <div class="panel">
      <h3>Aceite vs. PVI (soluciones no dominadas, todas las semillas)</h3>
      <canvas id="cmpScatter"></canvas>
      <p class="note">Más aceite (derecha) y menos PVI (abajo) es mejor. Un contrato cuya nube esté más abajo y a la
      derecha permite producir más con menor variabilidad del plan.</p>
    </div>
    <div class="panel">
      <h3>Hipervolumen en (aceite, PVI) por contrato</h3>
      <canvas id="cmpHv"></canvas>
      <p class="note">Barras = media entre semillas; el segmento muestra mínimo–máximo.</p>
    </div>
  </section>

  <h2 class="section">2 · Detalle por contrato</h2>
  <div class="toggle" id="contractToggle"></div>
  <section class="panels">
    <div class="panel">
      <h3>Convergencia: hipervolumen por generación</h3>
      <canvas id="convHv"></canvas>
      <p class="note">Una curva por semilla (tenue) y la media (gruesa). HV en el espacio (aceite, f2) normalizado con
      cotas fijas: sólo comparable entre semillas de este mismo contrato.</p>
    </div>
    <div class="panel">
      <h3>Convergencia: mejor aceite del archivo</h3>
      <canvas id="convOil"></canvas>
    </div>
    <div class="panel">
      <h3 id="f2Title">Convergencia: mínimo f2 del archivo</h3>
      <canvas id="convF2"></canvas>
      <p class="note" id="f2Note"></p>
    </div>
    <div class="panel">
      <h3>Frentes de Pareto finales (aceite vs. f2)</h3>
      <canvas id="paretoChart"></canvas>
      <p class="note">Puntos = soluciones no dominadas de cada semilla; triángulos = solución de compromiso.</p>
    </div>
    <div class="panel">
      <h3 id="resTitle">Utilización por recurso</h3>
      <canvas id="resChart"></canvas>
    </div>
    <div class="panel panel-full">
      <h3 id="ganttTitle">Plan de cosecha (Gantt)</h3>
      <div id="ganttScroll"><div id="ganttGrid"></div></div>
      <p class="note">Filas = recursos, columnas = días. El color identifica el lote; celdas claras = recurso ocioso.
      Pasa el cursor sobre una celda para ver lote y toneladas. Se muestra la solución de compromiso de la semilla
      con mayor hipervolumen.</p>
    </div>
    <div class="panel panel-full">
      <h3>Resultados por semilla</h3>
      <div class="scroll-x"><table id="seedTable"></table></div>
    </div>
  </section>
</main>

<script>
const DATA = __DATA_JSON__;
const META = DATA.meta;
const CONTRACTS = DATA.contracts;
const PALETTE = ["#6b7c4a", "#b3673a", "#3f6b8a", "#8a5a83", "#c9a227", "#4f8a7b", "#a14f4f"];
CONTRACTS.forEach((c, i) => { c.color = PALETTE[i % PALETTE.length]; });

const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "–" :
  Number(v).toLocaleString("es-CL", { minimumFractionDigits: d, maximumFractionDigits: d });
const pm = (s, d = 1) => `${fmt(s.mean, d)} ± ${fmt(s.std, d)}`;
const alpha = (hex, a) => hex + Math.round(a * 255).toString(16).padStart(2, "0");

const charts = {};
function mk(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), cfg);
}
const axis = (text) => ({ title: { display: true, text } });

/* ---------- Encabezado y tarjetas ---------- */
function renderHeader() {
  document.getElementById("subtitle").textContent =
    `${META.instance_label}: ${META.n_lots} lotes, ${META.n_days} días y ${META.n_resources} recursos. ` +
    `${META.seeds.length} semillas por contrato; parada tras ${META.stop.patience ?? "–"} generaciones sin mejora ` +
    `del hipervolumen (máx. ${META.stop.max_generations}).`;
  const cards = [
    ["Aceite del plan base", `${fmt(META.baseline_oil, 0)}`],
    ["Contratos evaluados", `${CONTRACTS.length}`],
    ["Semillas por contrato", `${META.seeds.length}`],
    ["Población (cáncer / linf.)", `${META.ga_params.n_cancer} / ${META.ga_params.n_lymph}`],
    ["Tiempo total", `${fmt(META.total_runtime_s / 60, 1)} min`],
  ];
  document.getElementById("cards").innerHTML = cards.map(([l, v]) =>
    `<div class="card"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");
}

/* ---------- 1. Comparación entre contratos ---------- */
function renderComparison() {
  const best = Math.max(...CONTRACTS.map(c => c.summary.hv_oil_pvi.mean));
  let html = `<tr><th>Contrato</th><th class="num">Aceite</th><th class="num">Δ vs. base</th>` +
    `<th class="num">PVI (t/día)</th><th class="num">HV (aceite, PVI)</th>` +
    `<th class="num">Generaciones</th><th class="num">Paradas por paciencia</th></tr>`;
  CONTRACTS.forEach(c => {
    const s = c.summary, delta = s.oil.mean - META.baseline_oil;
    html += `<tr class="${s.hv_oil_pvi.mean === best ? "best" : ""}">` +
      `<td><span class="swatch" style="background:${c.color}"></span>${c.label}</td>` +
      `<td class="num">${pm(s.oil, 0)}</td>` +
      `<td class="num">${delta >= 0 ? "+" : ""}${fmt(delta, 0)}</td>` +
      `<td class="num">${pm(s.pvi, 1)}</td>` +
      `<td class="num">${pm(s.hv_oil_pvi, 3)}</td>` +
      `<td class="num">${fmt(s.generations.mean, 1)}</td>` +
      `<td class="num">${s.stopped_by_patience}/${s.n_runs}</td></tr>`;
  });
  document.getElementById("cmpTable").innerHTML = html;

  mk("cmpScatter", {
    type: "scatter",
    data: { datasets: CONTRACTS.map(c => ({
      label: c.label, backgroundColor: alpha(c.color, 0.55), borderColor: c.color, pointRadius: 3,
      data: c.runs.flatMap(r => r.front.map(p => ({ x: p.oil, y: p.pvi }))),
    })) },
    options: { responsive: true, scales: { x: axis("Aceite producido"), y: axis("PVI (t/día)") },
               plugins: { legend: { position: "bottom" } } },
  });

  // Barras de HV medio con rango min–max dibujado por un plugin mínimo
  const rangePlugin = {
    id: "rangeBars",
    afterDatasetsDraw(chart) {
      const { ctx, scales: { y } } = chart;
      const meta = chart.getDatasetMeta(0);
      ctx.save(); ctx.strokeStyle = "#2a2822"; ctx.lineWidth = 1.5;
      meta.data.forEach((bar, i) => {
        const s = CONTRACTS[i].summary.hv_oil_pvi;
        const x = bar.x, y0 = y.getPixelForValue(s.min), y1 = y.getPixelForValue(s.max);
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1);
        ctx.moveTo(x - 5, y0); ctx.lineTo(x + 5, y0); ctx.moveTo(x - 5, y1); ctx.lineTo(x + 5, y1);
        ctx.stroke();
      });
      ctx.restore();
    },
  };
  mk("cmpHv", {
    type: "bar",
    data: { labels: CONTRACTS.map(c => c.label),
            datasets: [{ data: CONTRACTS.map(c => c.summary.hv_oil_pvi.mean),
                         backgroundColor: CONTRACTS.map(c => c.color) }] },
    options: { responsive: true, scales: { y: { beginAtZero: true, title: { display: true, text: "HV (aceite, PVI)" } },
                                          x: { ticks: { autoSkip: false, maxRotation: 40, minRotation: 25 } } },
               plugins: { legend: { display: false } } },
    plugins: [rangePlugin],
  });
}

/* ---------- 2. Detalle por contrato ---------- */
function curves(c, key, transform) {
  const G = Math.max(...c.runs.map(r => r.log[r.log.length - 1].gen));
  const labels = Array.from({ length: G + 1 }, (_, g) => g);
  const perRun = c.runs.map(r => labels.map(g => transform(r.log[Math.min(g, r.log.length - 1)][key])));
  const mean = labels.map((_, g) => perRun.reduce((a, row) => a + row[g], 0) / perRun.length);
  return { labels, perRun, mean };
}

function convergenceChart(id, c, key, transform, yLabel) {
  const { labels, perRun, mean } = curves(c, key, transform);
  const datasets = perRun.map((row, i) => ({
    label: `semilla ${c.runs[i].seed}`, data: row, borderColor: alpha(c.color, 0.35),
    borderWidth: 1, pointRadius: 0, tension: 0,
  }));
  datasets.push({ label: "Media", data: mean, borderColor: c.color, borderWidth: 3, pointRadius: 0, tension: 0 });
  mk(id, {
    type: "line", data: { labels, datasets },
    options: { responsive: true, scales: { x: axis("Generación"), y: axis(yLabel) },
               plugins: { legend: { position: "bottom", labels: { filter: (item) => item.text === "Media" } } } },
  });
}

function bestRun(c) { return c.runs.reduce((a, r) => (r.hv > a.hv ? r : a), c.runs[0]); }

function renderDetail(idx) {
  const c = CONTRACTS[idx];
  document.querySelectorAll("#contractToggle button").forEach((b, i) => b.classList.toggle("active", i === idx));

  convergenceChart("convHv", c, "hv", v => v, "Hipervolumen (normalizado)");
  convergenceChart("convOil", c, "best_f1", v => -v, "Mejor aceite del archivo");
  convergenceChart("convF2", c, "min_f2", v => v, `Mínimo f2 (${c.unit})`);
  document.getElementById("f2Title").textContent = `Convergencia: mínimo f2 (${c.unit})`;
  document.getElementById("f2Note").textContent =
    `f2 de «${c.label}» está en ${c.unit}: no compararlo con el de otros contratos.`;

  const paretoSets = c.runs.map((r, i) => ({
    label: `semilla ${r.seed}`, data: r.front.map(p => ({ x: p.oil, y: p.f2 })),
    backgroundColor: alpha(c.color, 0.25 + 0.6 * (i + 1) / c.runs.length), borderColor: c.color, pointRadius: 3,
  }));
  paretoSets.push({
    label: "Solución de compromiso", data: c.runs.map(r => ({ x: r.representative.oil, y: r.representative.f2 })),
    pointStyle: "triangle", pointRadius: 8, backgroundColor: "#2a2822", borderColor: "#2a2822",
  });
  mk("paretoChart", {
    type: "scatter", data: { datasets: paretoSets },
    options: { responsive: true, scales: { x: axis("Aceite producido"), y: axis(`f2 (${c.unit})`) },
               plugins: { legend: { position: "bottom",
                                    labels: { filter: (i) => i.text === "Solución de compromiso" } } } },
  });

  const br = bestRun(c);
  document.getElementById("resTitle").textContent = `Utilización por recurso (semilla ${br.seed})`;
  mk("resChart", {
    type: "bar",
    data: { labels: META.resource_labels,
            datasets: [{ data: br.representative.resource_totals, backgroundColor: c.color }] },
    options: { responsive: true, scales: { y: axis("Toneladas cosechadas"),
                                           x: { ticks: { autoSkip: false, maxRotation: 60, minRotation: 45 } } },
               plugins: { legend: { display: false } } },
  });

  renderGantt(br, c);
  renderSeedTable(c, br);
}

function lotColor(lot) { return `hsl(${(lot * 53) % 360}, 55%, 55%)`; }

function renderGantt(run, c) {
  document.getElementById("ganttTitle").textContent = `Plan de cosecha: ${c.label}, semilla ${run.seed}`;
  const grid = document.getElementById("ganttGrid");
  const nDays = META.n_days, labels = META.resource_labels;
  grid.style.gridTemplateColumns = `120px repeat(${nDays}, 13px)`;
  const parts = ["<div></div>"];
  for (let t = 0; t < nDays; t++) parts.push(`<div class="g-day">${(t + 1) % 5 === 0 ? t + 1 : ""}</div>`);
  labels.forEach((label, k) => {
    parts.push(`<div class="g-label">${label}</div>`);
    for (let t = 0; t < nDays; t++) {
      const cell = run.representative.gantt[k][t];
      parts.push(cell
        ? `<div class="g-cell" style="background:${lotColor(cell[0])}" ` +
          `title="${label} · día ${t + 1} · lote ${cell[0]} · ${fmt(cell[1], 1)} t"></div>`
        : `<div class="g-cell" title="${label} · día ${t + 1} · ocioso"></div>`);
    }
  });
  grid.innerHTML = parts.join("");
}

function renderSeedTable(c, br) {
  const stopTxt = { patience: "sin mejora de HV", max_generations: "máx. generaciones" };
  let html = `<tr><th>Semilla</th><th class="num">Generaciones</th><th>Parada</th><th class="num">HV (f1, f2)</th>` +
    `<th class="num">HV (aceite, PVI)</th><th class="num">Aceite</th><th class="num">f2 (${c.unit})</th>` +
    `<th class="num">PVI (t/día)</th><th class="num">Frente</th><th class="num">Tiempo (s)</th></tr>`;
  c.runs.forEach(r => {
    html += `<tr class="${r === br ? "best" : ""}"><td>${r.seed}</td><td class="num">${r.generations_run}</td>` +
      `<td>${stopTxt[r.stop_reason] || r.stop_reason}</td><td class="num">${fmt(r.hv, 4)}</td>` +
      `<td class="num">${fmt(r.hv_oil_pvi, 4)}</td><td class="num">${fmt(r.representative.oil, 0)}</td>` +
      `<td class="num">${fmt(r.representative.f2, 1)}</td><td class="num">${fmt(r.representative.pvi, 1)}</td>` +
      `<td class="num">${r.front.length}</td><td class="num">${fmt(r.elapsed_s, 1)}</td></tr>`;
  });
  const s = c.summary;
  html += `<tr><td><b>media ± desv.</b></td><td class="num">${fmt(s.generations.mean, 1)}</td><td></td>` +
    `<td class="num">${pm(s.hv, 4)}</td><td class="num">${pm(s.hv_oil_pvi, 4)}</td>` +
    `<td class="num">${pm(s.oil, 0)}</td><td class="num">${pm(s.f2, 1)}</td>` +
    `<td class="num">${pm(s.pvi, 1)}</td><td></td><td class="num">${pm(s.elapsed_s, 1)}</td></tr>`;
  document.getElementById("seedTable").innerHTML = html;
}

function renderToggle() {
  const toggle = document.getElementById("contractToggle");
  CONTRACTS.forEach((c, i) => {
    const b = document.createElement("button");
    b.textContent = c.label;
    b.addEventListener("click", () => renderDetail(i));
    toggle.appendChild(b);
  });
}

renderHeader();
renderComparison();
renderToggle();
renderDetail(0);
</script>
</body>
</html>
"""


def build_dashboard_html(results, output_path):
    """Escribe el dashboard HTML autocontenido a partir de `results` (dict
    devuelto por `cancer_ga.experiment.run_experiment`, o el JSON guardado)."""
    if not _CHARTJS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró Chart.js empaquetado en {_CHARTJS_PATH}. "
            "Debe distribuirse junto a este módulo (cancer_ga/utils/vendor/chart.umd.js)."
        )
    chartjs_src = _CHARTJS_PATH.read_text(encoding="utf-8")
    # "</" dentro de un <script> podría cerrar la etiqueta antes de tiempo
    data_json = json.dumps(results, allow_nan=False).replace("</", "<\\/")

    html = _TEMPLATE.replace("__CHARTJS_INLINE__", chartjs_src).replace("__DATA_JSON__", data_json)
    Path(output_path).write_text(html, encoding="utf-8")
    return str(output_path)
