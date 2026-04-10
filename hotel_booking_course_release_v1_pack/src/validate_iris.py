import numpy as np
import pandas as pd
from ikmeans import iKMeans

#vai carregar os dados do Iris
print("A carregar o ficheiro iris.dat...")
df_iris = pd.read_csv('../iris.dat', delim_whitespace=True, header=None)

#extrai os nº da tabela para processar
X_iris = df_iris.values

print(f"Os Dados Iris carregados: {X_iris.shape[0]} linhas e {X_iris.shape[1]} colunas.")

#inicia o ikmeans descartando <=10
ikm = iKMeans(min_cluster_size=11, random_state=42)

#inicia a procura do grupos
print("\nA procurar os padrões e grupos (...)")
ikm.fit_predict(X_iris)

print("\n--- RESULTADOS FINAIS ---")
print(f"Total de Grupos Encontrados: {len(ikm.cluster_centers_)}")
print("\nAs médias de cada grupo são:")
print(np.round(ikm.cluster_centers_, 2))