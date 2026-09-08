# Expected checks on the provided assignment dataset

These values are included as sanity checks so you can immediately tell whether your local run used the correct filters.

| Check | Expected |
|---|---:|
| Total samples | 10,500 |
| Part 2 population rows | 52,500 |
| Part 4 baseline melanoma + miraclib + PBMC samples | 656 |
| Baseline samples, prj1 | 384 |
| Baseline samples, prj3 | 272 |
| Baseline responder subjects | 331 |
| Baseline non-responder subjects | 325 |
| Baseline male subjects | 344 |
| Baseline female subjects | 312 |
| Required avg B cells: melanoma male responders, time=0, all sample/treatment types | **10,206.15** |
| Matching samples for required B-cell query | 485 |

## Part 3 expected statistical pattern

The lowest nominal raw p-value is for `cd4_t_cell` (about 0.0133). After Benjamini-Hochberg correction across the five populations, no cell population has adjusted p < 0.05.

This project intentionally reports both raw and adjusted p-values and bases the headline significance conclusion on the adjusted result.
