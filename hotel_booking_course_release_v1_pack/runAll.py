import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

print("\n[Sistema] Limpeza total de outputs da pipeline anterior...")

#lista de ficheiros CSV/JSON gerados pelos scripts
explicit_files = [
    os.path.join(ROOT, "hotel_bookings_clean.csv"),
    os.path.join(SRC, "experiments.csv"),
    os.path.join(SRC, "pipeline_run.json"),
    os.path.join(SRC, "stability_report.csv"),
    os.path.join(SRC, "profile_stability_report.csv"),
    os.path.join(SRC, "k_sweep_summary.csv"),
    os.path.join(SRC, "selected_k.txt"),
    os.path.join(SRC, "selection_notes.txt"),
    os.path.join(SRC, "e4_pca_results.csv"),
    os.path.join(SRC, "sample_indices_20k_seed42.txt"),
    os.path.join(SRC, "cluster_profile_sensitivity.csv"),
    os.path.join(SRC, "cluster_profile_scaler_sensitivity.csv"),
    os.path.join(SRC, "cluster_profile_adr_sensitivity.csv"),
    os.path.join(SRC, "profile_quality_flags.csv"),
    os.path.join(SRC, "experiments_meta.txt"),
]
for path in explicit_files:
    if os.path.isfile(path):
        os.remove(path)
        print(f" -> Removido: {os.path.relpath(path, ROOT)}")

for path in glob.glob(os.path.join(SRC, "cluster_profile_k*.csv")):
    os.remove(path)
    print(f" -> Removido: {os.path.relpath(path, ROOT)}")

#remove pastas de figuras e tabelas para regenerar tudo
for folder in [
    os.path.join(SRC, "graficos_relatorio"),
    os.path.join(ROOT, "tables"),
]:
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        print(f" -> Removida pasta: {os.path.relpath(folder, ROOT)}/")

#run_id unico para toda a corrida (lido por pipeline_utils.get_run_context)
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(SRC, "pipeline_run.json"), "w", encoding="utf-8") as f:
    json.dump({"run_id": run_id, "date": run_date}, f, indent=2)
print(f"\n[Sistema] run_id unico: {run_id}")

#ordem dos scripts da pipeline (core + extensao E4)
scripts_to_run = [
    "edaClean.py",
    "edaVisuals.py",
    "train_models.py",
    "estabel_hierar.py",
    "cluster_profile.py",
    "e4_pca_svd.py",
]

print("\n[Sistema] A iniciar pipeline...")
for script in scripts_to_run:
    print(f"\n>>> {script} <<<")
    print("-" * 50)
    try:
        subprocess.run([sys.executable, script], cwd=SRC, check=True)
    except subprocess.CalledProcessError:
        print(f"\n[ERRO] Falha em {script}")
        sys.exit(1)

print("\n[OK] Pipeline concluida. Todos os outputs sao desta execucao.")
print(f"  run_id: {run_id}")
