# Data visualization

- `loaders.py`: reads legacy and symbolic runs into one per-sample table (final probabilities, requirements met, shortfall, time, tokens). Symbolic runs without a parquet (aborted or in progress) are rebuilt from their worker logs; token counts are unavailable in that case.
- `plot_comparison.py`: grid-by-grid legacy vs symbolic figure, plus a CSV of the plotted numbers.
- `figures/`: generated output.

```bash
python viz/plot_comparison.py --legacy out/results/legacy_grid20 --symbolic out/results/symbolic_grid20_capped
```

```bash
python viz/plot_comparison.py --legacy out/results/legacy_grid20 --symbolic out/results/symbolic_grid20_aborted_uncapped --samples 0-4 --include-partial --out viz/figures/first5_uncapped.png
```
