# Cancer-Inspired Genetic Algorithm (CancerGA)

Algoritmo Genético Co-Evolutivo Bio-Inspirado en la dinámica entre células cancerosas y linfocitos dentro de un microambiente tumoral. Soporta optimización mono-objetivo, multiobjetivo (NSGA-II / Deb rules) y espacios de búsqueda continuos o mixtos con restricciones.

## 🚀 Características
- **Duelo biológico** estocástico entre subpoblaciones.
- **Mutación adaptativa** tipo ES y **Quimiotaxis** guiada por gradiente.
- **Metástasis** mediada por saltos con vuelos de Lévy.
- **Microambiente tumoral** opcional con soporte de fibroblastos.
- Soporte para familias benchmark integradas: **CTP**, **CF** y **MW**.

## 💻 Instalación

```bash
git clone [https://github.com/tu-usuario/cancer-ga.git](https://github.com/tu-usuario/cancer-ga.git)
cd cancer-ga
pip install -r requirements.txt