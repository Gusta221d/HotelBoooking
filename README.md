## Estrutura do Repositório

- `hotel_booking_course_release_v1_pack/runAll.py` -> ponto de entrada único para executar a pipeline completa
- `hotel_booking_course_release_v1_pack/src/` -> scripts fonte
- `hotel_booking_course_release_v1_pack/src/experiments.csv` -> registo de experiências
- `hotel_booking_course_release_v1_pack/src/stability_report.csv` -> resumo de estabilidade (ARI)
- `hotel_booking_course_release_v1_pack/src/cluster_profile_k8.csv` -> tabela de perfil inicial de clusters (k=8)
- `hotel_booking_course_release_v1_pack/src/graficos_relatorio/` -> figuras geradas
- `environment.yml` -> especificação do ambiente

## Dataset

- Dataset utilizado: **Hotel Booking Demand - course release v1**
- Nome do ficheiro bruto: `hotel_bookings_course_release_v1.csv`
- SHA-256: `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`
- Fonte pública: Kaggle (Jesse Mostipak)  
  <https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand>


## Configuração do Ambiente

A partir da raiz do repositório:

```bash
conda env create -f environment.yml
conda activate hotel-booking-clustering
```

## Execução Reprodutível 

A partir da raiz do repositório:

```bash
python hotel_booking_course_release_v1_pack/runAll.py
```

Isto executa, por ordem:

1. `edaClean.py`
2. `edaVisuals.py`
3. `train_models.py`
4. `estabel_hierar.py`
5. `cluster_profile.py`

## Principais Outputs

Após correr a pipeline, os principais artefactos gerados são:

- `hotel_booking_course_release_v1_pack/hotel_bookings_clean.csv`
- `hotel_booking_course_release_v1_pack/src/experiments.csv`
- `hotel_booking_course_release_v1_pack/src/stability_report.csv`
- `hotel_booking_course_release_v1_pack/src/cluster_profile_k8.csv`
- `hotel_booking_course_release_v1_pack/src/graficos_relatorio/` (boxplot, histograma e dendrograma Ward)
