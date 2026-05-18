## Hotel Booking Clustering — course release v1

### Dataset (versioning & governance)

| Campo | Valor |
|-------|--------|
| Release | **course release v1** |
| Ficheiro | `hotel_bookings_course_release_v1.csv` |
| SHA-256 | `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06` |
| Fonte | [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |

### Ambiente
```bash
conda env create -f environment.yml
conda activate hotel-booking-clustering
```

### Execução
```bash
cd hotel_booking_course_release_v1_pack
python runAll.py
```

### Estrutura de outputs
- `tables/` — missingness, resumo de outliers
- `src/experiments.csv` — log (`run_id`, `date`, `rep_id`, `metric`, `metric_sentence`, `sample_rule`, …)
- `src/cluster_profile_scaler_sensitivity.csv` — Std vs Robust
- `src/cluster_profile_adr_sensitivity.csv` — SemADR vs ComADR
- `src/graficos_relatorio/` — todas as figuras (EDA, dendrograma, E4)
- `src/k_sweep_summary.csv` — sweep completo de K por representação
- `src/selected_k.txt` — K escolhido (Silhouette + tamanho mínimo de cluster)
- `src/stability_report.csv` — ARI e Silhouette (10 seeds)
- `src/cluster_profile_k*.csv` — perfil final (Std-SemADR)
