# Regression: legacy policies as symbolic rules

- Legacy run: `out\results\legacy_grid20`
- Policies checked: 100 (all iterations of 20 samples); requirement values: 900
- Uncovered situations in translated policies (max): 0
- **Stored check** (best case at default PRISM settings vs the legacy run's reported values): max diff 1.44e-15, 0 above 1e-09
- **Exact check** (legacy DTMC vs new MDP best and worst; interval iteration, epsilon 1e-9): max diff 4.56e-10, 0 above 1e-07
- Max |best - worst| (interval iteration): 4.56e-10
- Policies solved with the Gauss-Seidel fallback (interval iteration did not converge): sample 9 iteration 4
- Policies PRISM could not solve (skipped): 0

**PASS**
