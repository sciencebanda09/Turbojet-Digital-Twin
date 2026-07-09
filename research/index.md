# Research Materials

| Path | Description |
|------|-------------|
| [`references.bib`](references.bib) | BibTeX bibliography (27 references) |
| [`paper.md`](paper.md) | Full paper draft (abstract, methods, results, discussion) |
| [`poster.md`](poster.md) | Conference poster template (architecture, results, conclusions) |
| [`figures/`](figures/) | Publication-quality figures (T–s, maps, EKF, SHAP, bar charts) |
| [`experiments/`](experiments/) | Experiment configs, run logs, C-MAPSS validation report |
| [`ablations/`](ablations/) | Subsystem contribution ablation study (6 variants, 18% stacking gain) |

## Quick Links

- **[Chapter 2: Theory](../docs/Theory.md)** — all methods with diagrams and citations
- **[Chapter 3: Equations](../docs/Equations.md)** — full mathematical formulation with per-section sources
- **[Chapter 6: Validation](../docs/Validation.md)** — quantitative model comparison with per-target metrics
- **[Paper draft](paper.md)** — ready for journal/conference submission
- **[Poster](poster.md)** — single-page conference poster

## Citation

```bibtex
@misc{dtwin2025,
  title = {Turbojet Digital Twin},
  howpublished = {\url{https://github.com/anomalyco/turbojet-dtwin}},
  year = {2025}
}
```

## Figures Checklist

- [ ] T–s diagram of Brayton cycle with station labels (see Theory.md §1)
- [ ] Engine cross-section with annotated components
- [ ] Compressor/turbine efficiency maps (see Theory.md §2)
- [ ] EKF predict-update-clamp block diagram (see Theory.md §8)
- [ ] Residual learning pipeline diagram (see Theory.md §6.2)
- [ ] Health indicator evolution over life (all 4 subsystems)
- [ ] RUL prediction vs actual (with conformal intervals)
- [ ] SHAP summary plot for top-10 features per target
- [ ] Confusion matrix for failure mode classification
- [ ] Model comparison bar chart (RMSE per target, all models)
- [ ] C-MAPSS RMSE comparison (FD001–FD004, tree vs LSTM vs DCNN)
