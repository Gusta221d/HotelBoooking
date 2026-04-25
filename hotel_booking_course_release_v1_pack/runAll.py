import os # Biblioteca para interagir com o sistema de ficheiros
import sys # Biblioteca para capturar erros e gerir o sistema
import subprocess # Biblioteca para executar outros ficheiros Python automaticamente
import shutil # Biblioteca para apagar pastas antigas

print("==================================================")
print("   HOTEL BOOKING DEMAND - REPRODUCIBILITY PIPELINE")
print("==================================================")

# 1. RESET DA PIPELINE (Garante reprodutibilidade total)
print("\n[Sistema] A limpar artefactos antigos para garantir uma execução limpa...")

# Caminhos dos ficheiros que queremos "resetar" (apagar antes de começar)
files_to_remove = [
    'hotel_bookings_clean.csv',     # Apaga o dataset limpo antigo na raiz
    'src/experiments.csv'           # Apaga o ficheiro de logs antigo na pasta src
]

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f" -> Removido: {file}")

# Apagar a pasta de gráficos para gerar uns novos
plot_folder = 'src/graficos_relatorio'
if os.path.exists(plot_folder):
    shutil.rmtree(plot_folder)
    print(f" -> Removida: Pasta {plot_folder}/")

# 2. DEFINIÇÃO DA ORDEM DOS SCRIPTS
# A ordem é crucial: não podemos treinar modelos sem primeiro limpar os dados!
scripts_to_run = [
    "edaClean.py",              # Altera para o nome exato que tens!
    "edaVisuals.py",            # Altera para o nome exato que tens!
    "train_models.py",          
    "estabel_hierar.py"         # Altera para o nome exato que tens!
]
# 3. EXECUÇÃO SEQUENCIAL
print("\n[Sistema] A iniciar a pipeline de Data Science...")

# Iteramos sobre a lista e corremos um script de cada vez
for script in scripts_to_run:
    print(f"\n>>> A EXECUTAR: {script} <<<")
    print("-" * 50)
    
    try:
        # O subprocess 'finge' que tu abriste o terminal dentro da pasta 'src'
        # e escreveste 'python nome_do_script.py'
        # cwd='src' diz-lhe para assumir a pasta src como o diretório de trabalho
        result = subprocess.run([sys.executable, script], cwd='src', check=True)
        
    except subprocess.CalledProcessError:
        # Se um script falhar (ex: falta de memória ou erro matemático), a pipeline para imediatamente
        print(f"\n[ERRO CRÍTICO] A pipeline falhou durante a execução do {script}!")
        print("Verifica os erros impressos acima antes de continuar.")
        sys.exit(1)
        
    except FileNotFoundError:
        # Se te esqueceres de colocar algum script na pasta 'src'
        print(f"\n[ERRO CRÍTICO] Não encontrei a pasta 'src' ou o ficheiro '{script}'.")
        print("Garante que a estrutura de pastas está correta!")
        sys.exit(1)

# 4. MENSAGEM DE SUCESSO
print("\n==================================================")
print("   PIPELINE CONCLUÍDA COM SUCESSO! 🎉")
print("==================================================")
print("Verifica os teus artefactos fresquinhos:")
print(" 1. O dataset limpo está na raiz do projeto (hotel_bookings_clean.csv)")
print(" 2. Os gráficos estão em src/graficos_relatorio/")
print(" 3. O histórico de logs está em src/experiments.csv")