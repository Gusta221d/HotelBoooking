## Hotel Booking Clustering — course release v1
Estudo experimental reprodutível de clustering sobre reservas hoteleiras

### Dataset (obrigatório — não incluído no repositório)
O ficheiro bruto **não** está no GitHub (política do course release). Antes de correr a pipeline:

1. Obter `hotel_bookings_course_release_v1.csv` (course release v1 na plataforma da UC)
2. Colocar em `hotel_booking_course_release_v1_pack/hotel_bookings_course_release_v1.csv`.
3. Verificar integridade (SHA-256):

Release  **course release v1** 
Ficheiro `hotel_bookings_course_release_v1.csv` 
SHA-256  `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06` 

Metadados adicionais: `hotel_booking_course_release_v1_pack/DATASET_MANIFEST.yml`, `SHA256SUMS.txt`, `column_roles.csv`

### Ambiente
Na raiz do repositório:
```bash
conda env create -f environment.yml
conda activate hotel-booking-clustering
```

Requisitos: Python 3.11, pandas, numpy, scikit-learn, scipy, matplotlib, seaborn

### Execução (reproduzir todos os resultados)
```bash
cd hotel_booking_course_release_v1_pack
python runAll.py
```

`runAll.py` apaga outputs anteriores e corre por ordem: `edaClean` → `edaVisuals` → `train_models` → `estabel_hierar` → `cluster_profile` → `e4_pca_svd`

`src/pipeline_run.json` (`run_id`) e `src/experiments.csv`.

### Estrutura do repositório

 `hotel_booking_course_release_v1_pack/runAll.py`  Orquestrador da pipeline 
 `hotel_booking_course_release_v1_pack/src/`  Código e outputs gerados 
 `hotel_booking_course_release_v1_pack/tables/`  Missingness, outliers (EDA) 
 `hotel_booking_course_release_v1_pack/src/graficos_relatorio/`  Figuras do relatório 
 `hotel_booking_course_release_v1_pack/src/experiments.csv`  Log de experiências 
 `environment.yml`  Especificação do ambiente 

### Outputs principais
- `tables/` — missingness, resumo de outliers
- `src/experiments.csv` — log (`run_id`, `date`, `rep_id`, métricas, seeds, …)
- `src/k_sweep_summary.csv`, `src/selected_k.txt` — seleção de K
- `src/stability_report.csv` — estabilidade (10 seeds)
- `src/cluster_profile_k*.csv` — perfis finais
- `src/e4_pca_results.csv` — extensão PCA (E4)
- `src/graficos_relatorio/` — figuras EDA, dendrograma, PCA

