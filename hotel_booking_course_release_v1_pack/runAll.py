import os
import sys
import subprocess
import shutil

print("\n[Sistema] A limpar antigos para garantir uma execução limpa")

files_to_remove = [
    'hotel_bookings_clean.csv',     
    'src/experiments.csv',
    'src/stability_report.csv',
    'src/cluster_profile_k8.csv'
]

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f" -> Removido: {file}")

plot_folder = 'src/graficos_relatorio'
if os.path.exists(plot_folder):
    shutil.rmtree(plot_folder)
    print(f" -> Removida: Pasta {plot_folder}/")

scripts_to_run = [
    "edaClean.py",              
    "edaVisuals.py",            
    "train_models.py",          
    "estabel_hierar.py",
    "cluster_profile.py"
]
print("\n[Sistema] A iniciar a pipeline de Data Science...")

for script in scripts_to_run:
    print(f"\n>>> A EXECUTAR: {script} <<<")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, script], cwd='src', check=True)
        
    except subprocess.CalledProcessError:
        print(f"\n[ERRO CRÍTICO] A pipeline falhou durante a execução do {script}!")
        sys.exit(1)
        
    except FileNotFoundError:
        print(f"\n[ERRO CRÍTICO] Não encontrei a pasta 'src' ou o ficheiro '{script}'.")
        sys.exit(1)

print("Verifica:")
print(" 1. O dataset limpo está na raiz do projeto (hotel_bookings_clean.csv)")
print(" 2. Os gráficos estão em src/graficos_relatorio/")
print(" 3. O histórico de logs está em src/experiments.csv")
print(" 4. A estabilidade está em src/stability_report.csv")
print(" 5. O perfil de clusters está em src/cluster_profile_k8.csv")