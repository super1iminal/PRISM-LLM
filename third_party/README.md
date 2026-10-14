# Third-party code

Clones live in their own directories here and are not committed (see `.gitignore`); this file records what each is.

| Directory | Source | Version | License | Used for |
|---|---|---|---|---|
| `123go/` | https://github.com/muqsit-azeem/dtstrat-123go-artifact (1-2-3-Go!, Azeem et al., VMCAI 2025) | commit 805665e8 (2024-10-02) | Apache-2.0 | Its result archives (`results/*.zip`) hold run statuses and times but no Pac-Man values; its own values are only in the paper (0.869 at horizon 25). Building it needs Storm and dtstrat from source (`README.md` there). Our reimplementation is `PRISM-Guided-Learning/src/baselines/decision_tree.py`. |
| `tectonic/` | https://github.com/tectonic-typesetting/tectonic/releases (Windows binary) | 0.15.0 | MIT | Builds the paper: `third_party/tectonic/tectonic.exe -X compile paper/seams27/main.tex` (fetches LaTeX packages on first use). |

Fetch: `git clone https://github.com/muqsit-azeem/dtstrat-123go-artifact third_party/123go && git -C third_party/123go checkout 805665e8`
