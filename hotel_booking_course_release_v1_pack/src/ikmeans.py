import numpy as np
from sklearn.cluster import KMeans

class iKMeans:
    def __init__(self, min_cluster_size=11, random_state=42):
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.initial_centroids_ = None
    #distancia ao quadrado normalizada por coluna: (u-v)^2 / r^2 com r = amplitude (max-min) por dimensao
    def _normalized_squared_distance(self, u, v, r):
        return np.sum(((u - v) / r) ** 2, axis=-1)

    def find_anomalous_patterns(self, X):
        #Converte os dados para nº decimais
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        #calcula a media de todas as colunas
        mu = np.mean(X, axis=0)

        #calcula a diferença entre o máx e o mín de cada coluna
        r = np.max(X, axis=0) - np.min(X, axis=0)
        #evita que de erro se a diferença for 0
        r[r == 0] = 1.0
        remaining_indices = list(range(n))
        centroids = []
        
        #procura pelos padroes anomalos
        while len(remaining_indices) > 0:
            X_rem = X[remaining_indices]

            #encontra o ponto mais longe dos dados
            dists_to_mu_initial = self._normalized_squared_distance(X_rem, mu, r)
            farthest_local_idx = np.argmax(dists_to_mu_initial)
            c = X_rem[farthest_local_idx].copy()
            
            prev_ap_indices = set()

            #ciclo de atualização de um cluster anómalo
            while True:
                dists_c = self._normalized_squared_distance(X_rem, c, r)
                dists_mu = self._normalized_squared_distance(X_rem, mu, r)
                
                #vai ver quem esta mais perto do novo grupo
                ap_mask = dists_c < dists_mu
                
                #obtem os indices da matriz 
                ap_indices = [remaining_indices[i] for i, is_ap in enumerate(ap_mask) if is_ap]

                #caso os membros do grupo nao mudem este para de procurar
                if set(ap_indices) == prev_ap_indices:
                    break
                
                #protecao contra erros
                if len(ap_indices) == 0:
                    break
                    
                prev_ap_indices = set(ap_indices)
                
                #atualiza a posicao do novo grupo com base nos membros que obteve
                c = np.mean(X[ap_indices], axis=0)
            

            #verifica que este tem tamanho min 
            if len(ap_indices) >= self.min_cluster_size:
                centroids.append(c)
            
            #remove os pontos usados para nao a voltar a testar
            remaining_indices = [idx for idx in remaining_indices if idx not in ap_indices]
            
        #verifica se o codigo conseguiu encontrar pelo menos 1 grupo valido
        if len(centroids) == 0:
            raise ValueError(f"Nenhum cluster atingiu o tamanho mínimo de {self.min_cluster_size}.")
            
        return np.array(centroids)

    def fit_predict(self, X):
        X = np.asarray(X, dtype=np.float64)

        #obtem as centros
        self.initial_centroids_ = self.find_anomalous_patterns(X)
        k_encontrado = len(self.initial_centroids_)
        print(f"iK-Means: Encontrei {k_encontrado} clusters válidos.")
        
        #usa esses centros para executar o k-means
        kmeans = KMeans(n_clusters=k_encontrado, init=self.initial_centroids_, n_init=1, random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(X)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.inertia_ = kmeans.inertia_
        
        return self.labels_