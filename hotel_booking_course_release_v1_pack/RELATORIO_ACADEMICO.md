# Segmentação não supervisionada de reservas hoteleiras: estudo experimental reprodutível

**Unidade curricular:** Unsupervised Learning · 2025/2026  
**Dataset:** Hotel Booking Demand — *course release v1*  
**Repositório:** `hotel_booking_course_release_v1_pack/`  
**Execução de referência:** `run_id = 20260601T174204Z` · gerada por `python runAll.py`

---

## Resumo

Este relatório documenta um estudo de *clustering* sobre reservas hoteleiras realizado sem etiquetas de cancelamento no momento da formação dos grupos. O objectivo é identificar perfis de reserva — padrões de antecedência, duração da estadia, canal de distribuição, segmento de mercado e volume de pedidos especiais — que possam apoiar decisão comercial no instante em que a reserva é criada, e não depois de o hóspede ter cancelado ou alterado a estadia.

O trabalho implementa uma pipeline reprodutível de seis etapas, desde a verificação de integridade do ficheiro bruto até à comparação entre espaço original e espaço reduzido por PCA, com registo automático de trinta e três experiências no ficheiro `experiments.csv`. A solução adoptada é K-Means com quatro clusters na representação Std-SemADR, sobre 117 429 reservas e 52 dimensões após codificação: Silhouette 0,1256, Calinski–Harabasz 10 911,93, Davies–Bouldin 2,0977, e cluster mais pequeno com 2,93% das observações. Complementam o estudo o iK-Means, que descobre automaticamente doze grupos, o clustering hierárquico Ward numa amostra fixa de vinte mil reservas, análise de estabilidade com dez *seeds* (ARI médio 0,6675), três variantes de sensibilidade de representação e a extensão E4 com dezoito componentes principais.

A interpretação qualitativa distingue um segmento corporativo de resposta rápida, com cerca de três por cento das reservas, de um bloco dominante online e OTA que representa mais de noventa e cinco por cento da massa. Dentro deste bloco, os clusters separam-se sobretudo pela mediana de *lead time* e pelo número de noites, e não pelo canal ou segmento modal isolados, que se repetem em vários grupos e exigem leitura cuidadosa das estatísticas descritivas.

---

## 1. Introdução

### 1.1 Contexto e motivação

A gestão hoteleira contemporânea opera sobre um fluxo contínuo de reservas provenientes de canais muito distintos: operadores turísticos, plataformas de reserva online, contratos empresariais e reservas directas. Cada canal traz padrões diferentes de antecedência, duração da estadia, sensibilidade ao preço e propensão para pedidos especiais. Conhecer estes padrões de forma sistemática permite alinhar políticas de inventário, campanhas promocionais e alocação de capacidade sem depender de rótulos pré-definidos pelo negócio.

O conjunto de dados *Hotel Booking Demand*, publicado por António, de Almeida e Nunes (2019), reúne mais de cento e dezanove mil reservas com dezenas de atributos numéricos e categóricos. Apesar da riqueza descritiva, o dataset não fornece uma coluna do tipo “perfil de cliente” que possa ser usada directamente como alvo de segmentação. A aprendizagem não supervisionada por *clustering* surge assim como abordagem adequada: o algoritmo agrupa reservas semelhantes no espaço das variáveis disponíveis no momento da reserva, e o investigador avalia depois se esses grupos têm significado operacional, se são estáveis e se resistem a escolhas alternativas de pré-processamento.

Este projecto não pretende prever cancelamentos nem optimizar receita directamente. Pretende responder a uma pergunta mais fundamental: **que tipos de comportamento de reserva emergem naturalmente dos dados**, quando se respeitam regras rigorosas de governação de variáveis e se comparam várias famílias de algoritmos sob um protocolo fixo e reprodutível.

### 1.2 Objectivos e âmbito do estudo

O estudo foi desenhado para cumprir os requisitos de um trabalho experimental reprodutível em aprendizagem não supervisionada. Em primeiro lugar, foi construída uma pipeline completa desde o dataset bruto até perfis interpretáveis, incluindo verificação criptográfica do ficheiro de entrada, relatórios de missingness e outliers, e um registo estruturado de todas as corridas de modelação. Em segundo lugar, foram comparadas três abordagens de agrupamento: K-Means como *baseline* obrigatório, iK-Means como método que descobre o número de grupos de forma automática, e clustering hierárquico aglomerativo com ligação de Ward como família conceptualmente distinta. Em terceiro lugar, o número de clusters foi seleccionado com uma regra explícita que combina qualidade interna medida pelo Silhouette com um critério mínimo de tamanho de cluster, evitando soluções que maximizam métricas à custa de micro-segmentos inúteis para o negócio. Em quarto lugar, a robustez das conclusões foi testada através de sensibilidade a escalonadores, inclusão ou exclusão do preço médio diário, enriquecimento do calendário, redução de dimensionalidade por PCA, e repetição do K-Means com dez inicializações aleatórias diferentes.

A unidade de análise é sempre uma reserva individual. O instante de segmentação adoptado é o da criação da reserva: apenas entram variáveis que, no cenário metodológico escolhido, estariam disponíveis nesse momento. Variáveis de cancelamento, estado da reserva após a confirmação, identificadores de agente ou empresa, e atributos suspeitos de serem actualizados após a reserva inicial foram excluídos do conjunto de *features* de clustering. O preço médio diário, ADR, não entra no modelo principal para que a segmentação não se reduza a faixas tarifárias. O cancelamento não foi utilizado nem para treinar nem para caracterizar os clusters neste relatório, embora pudesse ser analisado num estudo *post-hoc* futuro, separado e claramente etiquetado como tal.

---

## 2. Dados e preparação

### 2.1 Proveniência, integridade e governação

O ponto de partida é a versão fixa *course release v1* do dataset *Hotel Booking Demand*, distribuída na plataforma da unidade curricular. O ficheiro autoritativo é `hotel_bookings_course_release_v1.csv`, com 119 390 linhas e 32 colunas na versão bruta. Para garantir que qualquer reprodução do estudo utiliza exactamente o mesmo ficheiro, o script `edaClean.py` calcula no início de cada execução o digesto SHA-256 do ficheiro e imprime-o no log. O valor obtido na corrida de referência foi `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`. O dataset está licenciado sob Creative Commons Attribution 4.0, conforme indicado no manifesto do *release* e na página Kaggle de origem. O ficheiro bruto não deve ser commitado no repositório Git do projecto; em vez disso, o pacote inclui somas de verificação, um manifesto legível por máquina e o ficheiro `column_roles.csv`, que documenta o papel recomendado de cada variável para evitar *leakage* e uso indevido de identificadores.

A Tabela seguinte resume os metadados essenciais do dataset utilizado.

| Atributo | Valor |
|----------|--------|
| Nome | Hotel Booking Demand |
| Versão | Course release v1 |
| Ficheiro | `hotel_bookings_course_release_v1.csv` |
| Dimensão bruta | 119 390 × 32 |
| SHA-256 | `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06` |
| Licença | CC BY 4.0 |
| Referência bibliográfica | António et al. (2019), *Data in Brief* |

### 2.2 Controlo de *leakage* e variáveis excluídas

O princípio de *leakage control* exige que nenhuma informação sobre o desfecho da reserva ou sobre eventos posteriores à decisão inicial de reserva entre no vector de *features* usado para formar clusters. Na prática, foram removidas dez colunas antes de qualquer modelação. A variável `is_canceled` é um *outcome* directo e não pode ser usada para segmentar reservas se o objectivo é perfilar comportamento à data da reserva. As variáveis `reservation_status` e `reservation_status_date` descrevem o estado da reserva após confirmação ou alteração e foram igualmente excluídas. Os campos `agent`, `company` e `country` funcionam como identificadores ou quase-identificadores de cardinalidade muito elevada; incluí-los com *one-hot encoding* espalharia a massa de observações por milhares de dimensões esparsas sem ganho interpretável claro. Os atributos `assigned_room_type`, `booking_changes` e `days_in_waiting_list` foram retirados por poderem reflectir alterações operacionais posteriores à reserva inicial, dependendo do instante de segmentação que se adopta. Por fim, `arrival_date_year` foi eliminado por redundância com a informação de mês de chegada, que se mantém como variável categórica no modelo principal.

A Tabela seguinte documenta estas exclusões de forma explícita, como exige o enunciado do projecto.

| Variável | Motivo da exclusão |
|----------|-------------------|
| `is_canceled` | Variável de resultado |
| `reservation_status`, `reservation_status_date` | Informação pós-decisão |
| `agent`, `company`, `country` | ID / cardinalidade elevada |
| `assigned_room_type`, `booking_changes`, `days_in_waiting_list` | Potencial pós-reserva |
| `arrival_date_year` | Redundância temporal |

### 2.3 Transformações aplicadas e base limpa final

Após a remoção das colunas indicadas, a base de trabalho passou por uma sequência de transformações documentadas e reprodutíveis. A variável `children` apresentava apenas quatro valores em falta, correspondendo a 0,0034% das linhas; estes foram imputados a zero, interpretando a ausência de registo como ausência de crianças na reserva, em linha com a prática habitual neste dataset. Em seguida aplicou-se um filtro de qualidade sobre o ADR, mantendo apenas reservas com preço médio diário estritamente positivo e não superior a 5000 unidades monetárias. Este passo eliminou 1961 reservas, cerca de 1,64% do total original, correspondentes a tarifas inválidas ou valores extremos administrativos que não devem influenciar a geometria dos clusters. Por último, todas as variáveis categóricas com níveis cuja frequência relativa é inferior a um por cento foram agrupados na classe `Other`, reduzindo esparsidade e estabilizando o *one-hot encoding*.

O resultado desta etapa é o ficheiro `hotel_bookings_clean.csv`, com 117 429 linhas e 22 colunas, que alimenta todos os scripts subsequentes de modelação, estabilidade e perfilagem.

### 2.4 Outliers e decisão de não remoção global

Na representação principal, designada SemADR, os outliers identificados pelo método do intervalo interquartil não são removidos de forma global, excepto no caso do ADR inválido já filtrado no passo de qualidade. O relatório `tables/outlier_summary.csv` quantifica estes valores para transparência metodológica, sem que a sua remoção automática seja imposta ao *clustering*.

**Tabela A.** Outliers IQR por variável (base limpa, \(n = 117\,429\)).

| Variável | Limite inferior | Limite superior | N.º outliers | % |
|----------|-----------------|-----------------|--------------|---|
| lead_time | −195,0 | 373,0 | 3 005 | 2,52 |
| stays_in_week_nights | −2,0 | 6,0 | 3 354 | 2,81 |
| adults | 2,0 | 2,0 | 29 710 | 24,88 |
| adr | 0,01 | 211,07 | 5 753 | 4,82 |
| children | 0,0 | 0,0 | 8 590 | 7,19 |
| previous_cancellations | 0,0 | 0,0 | 6 484 | 5,43 |
| required_car_parking_spaces | 0,0 | 0,0 | 7 416 | 6,21 |
| total_of_special_requests | −1,5 | 2,5 | 2 877 | 2,41 |

A leitura desta tabela merece cuidado. No caso da variável `adults`, o intervalo interquartil degenera porque o primeiro e o terceiro quartil coincidem no valor dois, fazendo com que reservas perfeitamente normais com um ou três adultos apareçam como outliers formais do método. Isto ilustra porque o IQR, embora útil para relatório exploratório, não deve ser aplicado de forma mecânica como critério universal de exclusão em todas as variáveis. O filtro IQR sobre o ADR, com limites aproximados entre 0,01 e 211,07, afectaria 5753 linhas, cerca de 4,82% do conjunto; este filtro é activado apenas na variante de sensibilidade ComADR, onde o ADR entra como variável numérica de clustering, e não na base principal SemADR.

### 2.5 Análise exploratória visual

O módulo `edaVisuals.py` produz seis figuras guardadas em `src/graficos_relatorio/`, que complementam as tabelas numéricas com evidência visual. O *boxplot* comparativo do ADR antes e depois do filtro de qualidade mostra como a remoção de tarifas inválidas comprime a cauda inferior da distribuição sem eliminar a assimetria típica de preços hoteleiros. Os *boxplots* de *lead time*, noites de semana, adultos e pedidos especiais confirmam a presença de caudas longas e valores extremos que justificam a decisão de não podar globalmente o dataset antes do clustering. Os histogramas de *lead time* e ADR evidenciam concentración de massa em valores baixos e uma cauda longa de reservas muito antecipadas ou tarifas elevadas. O dendrograma de Ward numa subamostra de 1500 observações documenta o corte hierárquico associado a quatro clusters. Por fim, o *scree plot* e o gráfico comparativo de Silhouette e Davies–Bouldin entre espaço original e espaço PCA pertencem à extensão E4 e sustentam a discussão sobre utilidade da redução de dimensionalidade.

---

## 3. Metodologia

### 3.1 Pipeline computacional e rastreabilidade

Toda a execução experimental é orquestrada pelo script `runAll.py`. No início de cada corrida, este script remove os artefactos gerados anteriormente — ficheiros CSV de resultados, figuras e a pasta `tables/` — de modo a evitar mistura de resultados de execuções distintas. Em seguida grava em `pipeline_run.json` um identificador único `run_id` e a data UTC da corrida, que passam a constar em todas as linhas do ficheiro `experiments.csv`. Por fim, executa por ordem fixa os seis módulos Python: limpeza e documentação dos dados, visualizações exploratórias, modelação principal e *sweep* de hiperparâmetros, análise de estabilidade e comparação com Ward na subamostra, construção de perfis e sensibilidades, e extensão PCA.

Cada linha de `experiments.csv` funciona como registo auditável de uma experiência: identifica a representação dos dados através dos campos `rep` e `rep_id`, o algoritmo utilizado, o número de clusters, a *seed* aleatória, o número de linhas efectivamente clusterizadas, a regra de amostragem quando aplicável, as três métricas internas principais, a percentagem do cluster mais pequeno, o tempo de execução em segundos, uma descrição textual dos parâmetros e notas livres que contextualizam decisões específicas, como a comparação justa entre iK-Means e K-Means quando o primeiro descobre um número diferente de grupos. Este desenho responde directamente ao requisito do enunciado de manter um *experiment log* legível por máquina e por humanos.

### 3.2 Representação dos dados para clustering

A matriz de entrada dos algoritmos é construída pelo `ColumnTransformer` implementado na função `build_preprocessor`. O bloco numérico inclui, na representação SemADR, onze variáveis: *lead time*, noites em fim-de-semana e em dias de semana, número de adultos, crianças e bebés, indicador de hóspede repetido, histórico de cancelamentos e de reservas não canceladas, lugares de estacionamento requisitados e total de pedidos especiais. Este bloco é escalado com `StandardScaler` na configuração principal, o que centra cada variável na média amostral e divide pelo desvio padrão, colocando todas as dimensões numéricas em comparabilidade de escala antes da distância euclidiana. O bloco categórico inclui oito variáveis — hotel, mês de chegada, regime de refeições, segmento de mercado, canal de distribuição, tipo de quarto reservado, tipo de depósito e tipo de cliente — transformadas por `OneHotEncoder` com a opção `handle_unknown="ignore"`, de forma que categorias não vistas durante o ajuste do pré-processador não provoquem erro em execuções futuras.

Após o `fit_transform`, a representação Std-SemADR possui 52 colunas e recebe o identificador `R-EUCLID-d52-Std-ADR-no-country-excl-rare-gov-month-OHE`, que resume de forma compacta a métrica euclidiana, a dimensão final, o escalonador, a ausência de ADR nas *features*, a exclusão de país e agentes, a governação de categorias raras e o tratamento do calendário através do mês em codificação *one-hot*.

Para além da representação principal, foram definidas três variantes experimentais que testam a sensibilidade das conclusões. A representação Robust-SemADR substitui o `StandardScaler` pelo `RobustScaler`, baseado na mediana e no intervalo interquartil, menos sensível a valores extremos nas variáveis numéricas. A representação Std-ComADR inclui o ADR como variável numérica adicional e aplica previamente o filtro IQR sobre o ADR, reduzindo o conjunto a 113 380 reservas. A representação Std-SemADR+weekday acrescenta a semana e o dia do mês como variáveis numéricas brutas para além do mês já codificado em *one-hot*, elevando a dimensão para 54.

| Nome | Escalonador | ADR nas features | Calendário | Dimensão | \(n\) |
|------|-------------|------------------|------------|----------|-------|
| Std-SemADR | Standard | Não | Mês (OHE) | 52 | 117 429 |
| Robust-SemADR | Robust | Não | Mês (OHE) | 52 | 117 429 |
| Std-ComADR | Standard | Sim (+ IQR) | Mês (OHE) | 53 | 113 380 |
| Std-SemADR+weekday | Standard | Não | Mês OHE + semana/dia | 54 | 117 429 |

A métrica implícita em todas estas representações, quando usadas com K-Means ou Ward, é a distância euclidiana no espaço resultante. Esta escolha é coerente com o uso de Ward, que exige geometria euclidiana, e é documentada textualmente em cada linha do registo de experiências.

### 3.3 Índices de validação interna e o seu significado

A avaliação da qualidade dos clusters baseia-se em índices internos calculados no mesmo espaço em que o modelo foi treinado, sem recurso a rótulos externos. O coeficiente de Silhouette mede, para cada observação, o quão mais próxima está do seu próprio cluster do que dos clusters vizinhos, e resume essa informação num valor médio entre menos um e um. Valores mais altos indicam clusters mais densos e mais separados uns dos outros. Dado o tamanho do dataset, o Silhouette foi calculado sobre uma amostra aleatória de até trinta mil pontos com *seed* fixa, mantendo a comparabilidade entre corridas sem impor custo computacional proibitivo.

O índice de Calinski–Harabasz é a razão entre a dispersão média entre clusters e a dispersão média dentro dos clusters; valores maiores indicam melhor separação relativa. O índice de Davies–Bouldin mede, para cada cluster, a sua semelhança com o cluster mais parecido e faz a média desses valores; neste caso, valores menores são preferíveis porque indicam menor sobreposição entre grupos. Para além destes índices clássicos, foi calculada em cada experiência a percentagem de reservas pertencentes ao cluster mais pequeno, designada `min_cluster_pct`. Esta grandeza não é um índice tradicional de validação, mas funciona como critério de governança: um cluster que reúne apenas 0,12% das reservas, cerca de cento e quarenta observações em cento e dezessete mil, pode inflacionar o Silhouette ao absorver pontos muito atípicos sem representar um segmento de mercado acionável.

### 3.4 Algoritmos de clustering utilizados

O K-Means foi implementado como *baseline* obrigatório do projecto. O algoritmo particiona as observações em exatamente k grupos, minimizando a soma das distâncias quadráticas intra-cluster em torno dos centroides. O intervalo de k explorado foi fixado *a priori* em {3, 4, 5, 6, 7, 8}, sem pesquisa ilimitada de hiperparâmetros. Em cada valor de k da representação principal, utilizou-se `n_init=10` com inicialização *k-means++* e `random_state=42` na corrida de referência, de modo a reduzir a dependência de uma única inicialização desfavorável.

O iK-Means implementa uma lógica diferente. Em vez de fixar k, o método procura sequencialmente padrões de observações que estão mais próximas de um centroide móvel do que da média global do dataset, usando uma distância ao quadrado normalizada pela amplitude de cada dimensão. Quando o conjunto de observações associadas a esse padrão estabiliza e contém pelo menos onze pontos, o centroide é guardado e as observações utilizadas são retiradas do conjunto restante. O processo repete-se até não restarem pontos. O número de centroides encontrados define então k para uma corrida final de K-Means com inicialização fixa nesses centroides e `n_init=1`. Este protocolo permite comparar tempo de execução e índices internos com o *baseline*, mantendo o mesmo espaço de representação.

O clustering hierárquico aglomerativo com ligação de Ward foi escolhido como segunda família de algoritmos. O Ward funde clusters de forma a minimizar o aumento da variância intra-grupo, o que o torna naturalmente compatível com distâncias euclidianas no espaço pré-processado. A complexidade computacional e de memória, tipicamente quadrática no número de observações, impôs a aplicação deste método a uma subamostra fixa de vinte mil reservas, cujos índices estão guardados no ficheiro `sample_indices_20k_seed42.txt` com *seed* quarenta e dois, permitindo reproduzir exactamente o mesmo subconjunto.

### 3.5 Regra de selecção do número de clusters

A selecção de k na representação Std-SemADR não se baseou apenas no máximo do Silhouette. Para cada valor de k no intervalo pré-definido, treinou-se o K-Means e registaram-se Silhouette, Calinski–Harabasz, Davies–Bouldin e a percentagem do cluster mais pequeno. Em seguida identificaram-se os valores de k **admissíveis**, definidos como aqueles em que o cluster mais pequeno contém pelo menos um por cento das reservas. Entre os admissíveis, escolheu-se o valor com maior Silhouette.

A aplicação desta regra aos resultados obtidos com *seed* quarenta e dois mostra porque k igual a oito, apesar de apresentar o Silhouette mais elevado de 0,1413, foi rejeitado: o menor cluster nessa solução representa apenas 0,12% das reservas. Os valores k igual a cinco, seis e sete também ficaram abaixo do limiar de um por cento no cluster mínimo. Entre k igual a três e k igual a quatro, ambos admissíveis, o Silhouette de 0,1256 para quatro clusters supera o de 0,1213 para três clusters. A decisão final de adoptar **quatro clusters** ficou assim documentada em `selection_notes.txt` e reflectida numa linha especial de tipo K-selection no ficheiro `experiments.csv`, incluindo a menção explícita de que k igual a oito não foi seleccionado apesar do Silhouette superior.

### 3.6 Estabilidade, sensibilidade e extensão PCA

A estabilidade da partição foi avaliada repetindo o K-Means com k fixo em quatro no dataset completo, usando dez *seeds* diferentes: 42, 67, 123, 2026, 71, 7, 99, 314, 555 e 1001. Para cada par de partições obtidas, calculou-se o Adjusted Rand Index, que mede a concordância entre duas classificações corrigindo o acordo esperado ao acaso. Um ARI igual a um indicaria partições idênticas; um ARI próximo de zero indicaria concordância semelhante à aleatória. Para além do ARI, calculou-se a concordância de perfis entre cada partição e a partição de referência associada à *seed* quarenta e dois. Esta medida alinha pares de clusters com o algoritmo húngaro sobre a matriz de contingência e verifica, para cada par alinhado, se o canal modal coincide, se o segmento modal coincide e se as medianas de *lead time* diferem em no máximo trinta dias. A média destes três critérios binários por par, e depois sobre todos os pares, fornece uma leitura de estabilidade narrativa que complementa o ARI.

A sensibilidade à representação foi testada comparando StandardScaler com RobustScaler, e SemADR com ComADR no mesmo subconjunto de linhas após filtro IQR de ADR. A extensão E4 aplicou análise de componentes principais ao espaço com cinquenta e quatro dimensões usado nesse módulo, reteve componentes até a variância acumulada atingir pelo menos noventa por cento — resultando em dezoito componentes — e repetiu o *sweep* de K-Means com cinco *seeds*, reportando média e desvio padrão das métricas.

---

## 4. Resultados experimentais

### 4.1 K-Means na representação principal Std-SemADR

A Tabela 1 apresenta o *sweep* completo de K-Means sobre as 117 429 reservas da representação Std-SemADR, com *seed* quarenta e dois. Os valores de Silhouette situam-se na ordem de 0,09 a 0,14, o que é típico em espaços de alta dimensão com muitas variáveis categóricas codificadas de forma esparsa; não deve ser lido como evidência de separação perfeita, mas como base comparativa entre valores de k e entre representações alternativas analisadas mais adiante.

**Tabela 1.** *Sweep* completo, Std-SemADR, \(n = 117\,429\), seed 42.

| \(k\) | Silhouette ↑ | Calinski–Harabasz ↑ | Davies–Bouldin ↓ | Menor cluster (%) | Tempo (s) | Admissível (≥1%) |
|------|--------------|---------------------|------------------|-------------------|-----------|------------------|
| 3 | 0,1213 | 10 977,85 | 2,3581 | 2,83 | 1,18 | Sim |
| **4** | **0,1256** | **10 911,93** | **2,0977** | **2,93** | **0,99** | **Sim (escolhido)** |
| 5 | 0,0953 | 10 422,20 | 1,8804 | 0,76 | 1,04 | Não |
| 6 | 0,1140 | 10 920,39 | 1,6876 | 0,12 | 1,19 | Não |
| 7 | 0,1201 | 11 421,09 | 1,5764 | 0,18 | 1,03 | Não |
| 8 | 0,1413 | 11 963,67 | 1,4881 | 0,12 | 1,46 | Não |

A leitura integrada desta tabela explica a decisão final de forma explícita. A partir de k igual a cinco, o algoritmo começa a criar micro-clusters com menos de um por cento das reservas. Estes micro-clusters tendem a melhorar o Davies–Bouldin, porque absorvem pontos muito periféricos e tornam os clusters principais mais compactos, e em k igual a oito chegam mesmo a produzir o Silhouette mais alto de todas as configurações testadas nesta representação. No entanto, uma solução em que o cluster mais pequeno tem apenas cerca de cento e quarenta reservas não é adequada para segmentação operacional em marketing ou revenue management, que exige segmentos com massa crítica mínima. Entre os valores admissíveis, k igual a quatro oferece o melhor Silhouette. O Calinski–Harabasz é ligeiramente inferior ao de k igual a três, mas a diferença é modesta face ao ganho de uma segmentação com quatro narrativas distintas em vez de três, o que se tornará evidente na interpretação dos perfis.

### 4.2 Resultados do iK-Means em todas as representações

**Tabela 2.** iK-Means (`min_cluster_size=11`, seed 42).

| Representação | \(k\) auto | Silhouette | CH | DB | Menor cluster (%) | Tempo (s) |
|---------------|------------|------------|-----|-----|-------------------|-----------|
| Std-SemADR | 12 | 0,1092 | 9 145,63 | 1,8918 | 0,76 | 171,46 |
| Robust-SemADR | 12 | 0,0921 | 10 892,26 | 1,8597 | 0,08 | 41,49 |
| Std-ComADR | 12 | 0,0914 | 7 911,16 | 1,9726 | 0,14 | 34,58 |
| Std-SemADR+weekday | 12 | 0,0905 | 7 831,29 | 1,9093 | 0,76 | 37,64 |

Em todas as representações testadas, o iK-Means convergiu para doze clusters automáticos com o parâmetro de tamanho mínimo fixado em onze observações. Na representação principal, o Silhouette de 0,1092 ficou abaixo do 0,1256 obtido pelo K-Means com quatro clusters, e o menor cluster reuniu apenas 0,76% das reservas, o que violaria a regra de um por cento se esta fosse aplicada ao iK-Means da mesma forma. O tempo de execução na Std-SemADR, cerca de cento e setenta e um segundos, reflecte a natureza iterativa da descoberta de padrões anómalos e contrasta com cerca de um segundo para cada valor de k no *sweep* de K-Means. A conclusão metodológica é que o iK-Means cumpre o papel de exploração comparativa exigido pelo enunciado — mostra o que acontece quando k não é imposto — mas não substitui o *sweep* principiado com regra de tamanho mínimo como base da segmentação oficial do projecto.

### 4.3 Sensibilidade às variantes de representação

As Tabelas 3 a 5 documentam o *sweep* completo de K-Means nas três representações alternativas, permitindo avaliar como mudanças de escalonador, inclusão de ADR e enriquecimento do calendário afectam as métricas internas.

**Tabela 3.** Robust-SemADR — K-Means, seed 42.

| \(k\) | Silhouette | CH | DB | Menor cluster (%) |
|------|------------|-----|-----|-------------------|
| 3 | 0,0987 | 15 902,61 | 2,1438 | 0,23 |
| 4 | 0,0999 | 14 944,99 | 1,7091 | **0,12** |
| 5 | 0,1124 | 14 859,58 | 1,6571 | 0,12 |
| 6 | 0,1174 | 14 378,79 | 1,5047 | 0,09 |
| 7 | 0,1279 | 14 037,88 | 1,5381 | 0,08 |
| 8 | 0,1304 | 13 556,31 | 1,5412 | 0,04 |

**Tabela 4.** Std-ComADR (\(n = 113\,380\); IQR ADR; 4 049 linhas removidas).

| \(k\) | Silhouette | CH | DB | Menor cluster (%) |
|------|------------|-----|-----|-------------------|
| 3 | 0,1143 | 10 483,96 | 2,3364 | 2,89 |
| 4 | 0,1132 | 9 675,45 | 2,1963 | 2,88 |
| 5 | 0,0944 | 9 551,62 | 2,0731 | 2,98 |
| 6 | 0,0983 | 9 893,52 | 1,7673 | 0,13 |
| 7 | 0,1212 | 10 028,53 | 1,7395 | 0,71 |
| 8 | 0,1331 | 10 748,46 | 1,5257 | 0,13 |

**Tabela 5.** Std-SemADR+weekday (54 dimensões).

| \(k\) | Silhouette | CH | DB | Menor cluster (%) |
|------|------------|-----|-----|-------------------|
| 3 | 0,1025 | 9 469,41 | 2,5891 | 2,83 |
| 4 | 0,0842 | 8 736,65 | 2,5065 | 2,83 |
| 5 | 0,0901 | 8 921,23 | 2,1128 | 0,76 |
| 6 | 0,0903 | 9 172,26 | 1,8759 | 0,12 |
| 7 | 0,1053 | 9 064,62 | 1,8114 | 0,12 |
| 8 | 0,1018 | 9 484,30 | 1,8213 | 0,18 |

No caso do RobustScaler, os valores de Calinski–Harabasz são visivelmente mais elevados do que na representação Standard, o que pode sugerir à primeira vista clusters “melhores”. Contudo, em k igual a quatro o menor cluster tem apenas 0,12% das reservas, e em nenhum valor de k do intervalo testado se atinge o limiar de um por cento. Isto reforça a escolha do StandardScaler para a segmentação final, porque a escala robusta, ao comprimir a influência de extremos, redistribui massa de forma que fragmenta grupos pequenos artificialmente. Na variante ComADR, o melhor Silhouette bruto surge em k igual a oito, mas novamente com micro-cluster de 0,13%; entre configurações admissíveis, k igual a três e quatro são comparáveis. A análise de perfis no mesmo subconjunto pós-IQR, apresentada mais adiante, mostra que a leitura qualitativa permanece alinhada entre SemADR e ComADR. Por fim, acrescentar semana e dia do mês como numéricos brutos piora o Silhouette em k igual a quatro de 0,1256 para 0,0842, sugerindo que o mês em *one-hot* já captura a sazonalidade relevante e que a granularidade extra introduz sobretudo ruído geométrico.

### 4.4 Comparação entre K-Means, Ward e iK-Means na amostra de vinte mil

**Tabela 6.** Comparação na amostra de 20 000 (Std-SemADR, \(k=4\), seed 42).

| Algoritmo | Silhouette | CH | DB | Menor cluster (%) | Tempo (s) |
|-----------|------------|-----|-----|-------------------|-----------|
| K-Means | 0,1197 | 1 667,77 | 2,2201 | 2,72 | 0,26 |
| Ward | **0,2925** | 1 714,25 | **1,6444** | **0,15** | 11,68 |
| iK-Means | 0,1197 | 1 667,77 | 2,2201 | 2,72 | 1,85 |

A comparação na subamostra de vinte mil reservas, correspondente a cerca de dezassete por cento do dataset completo, merece interpretação cuidadosa. O Ward obtém Silhouette de 0,2925, mais do que o dobro do valor do K-Means na mesma amostra, e Davies–Bouldin inferior, o que à primeira vista poderia sugerir substituir o *baseline*. Contudo, o menor cluster do Ward representa apenas 0,15% da amostra, cerca de trinta reservas, o que é ainda pior do ponto de vista da governança de tamanho mínimo. Além disso, o Ward optimiza uma função objetivo hierárquica distinta da inércia do K-Means, e os índices internos calculados no mesmo espaço euclidiano não são directamente comparáveis entre famílias de algoritmos da mesma forma que o são entre duas corridas de K-Means com *seeds* diferentes. O dendrograma produzido numa subamostra de 1500 observações serve sobretudo como diagnóstico visual da estrutura hierárquica e do nível de corte associado a quatro clusters, não como prova de superioridade operacional sobre a partição de cento e dezessete mil pontos. Quanto ao iK-Means nesta amostra, o método descobriu doze grupos, mas as métricas reportadas em k igual a quatro seguem o protocolo de avaliação justa com inicialização *k-means++*, reproduzindo exactamente os valores do K-Means, conforme documentado nas notas de `experiments.csv`.

### 4.5 Estabilidade com dez inicializações aleatórias

**Tabela 7.** Resumo de estabilidade (`stability_report.csv`).

| Indicador | Média | Desvio padrão |
|-----------|-------|---------------|
| Silhouette | 0,1204 | 0,0082 |
| ARI (45 pares) | 0,6675 | 0,2567 |
| Profile agreement vs seed 42 | 0,8889 | 0,1111 |

A repetição do K-Means com k fixo em quatro e dez *seeds* diferentes no dataset completo produz uma imagem nuanceada de estabilidade. O Silhouette médio de 0,1204 com desvio padrão de apenas 0,0082 indica que a qualidade interna média das soluções é estável: diferentes inicializações levam a partições com cohesão semelhante quando medida por este índice. Já o Adjusted Rand Index médio de 0,6675, com desvio padrão de 0,2567 entre os quarenta e cinco pares de comparações possíveis, classifica a estabilidade como moderada. Na prática, duas execuções com *seeds* distintas concordam em cerca de dois terços da estrutura de atribuição de pontos a clusters, o que é suficiente para identificar tendências globais — como a existência de um bloco online massivo e um nicho corporativo — mas insuficiente para tratar os IDs de cluster como etiquetas permanentes e imutáveis sem revisão periódica.

A concordância de perfis face à partição de referência da *seed* quarenta e dois é mais reconfortante do ponto de vista narrativo. Com média de 0,8889, a maioria dos pares de clusters alinhados mantém o mesmo canal modal, o mesmo segmento modal e medianas de *lead time* próximas dentro da tolerância de trinta dias, mesmo quando a fronteira exacta entre clusters se desloca. O relatório automático classifica esta combinação como *moderately unstable* pelo critério ARI inferior a 0,7, o que deve ser comunicado honestamente em qualquer apresentação do trabalho, mas não invalida a leitura qualitativa dominante.

### 4.6 Sensibilidade ao escalonador: StandardScaler versus RobustScaler

**Tabela 8.** Pares alinhados (Hungarian), \(k=4\). Concordância global = 0,75.

| Cluster Std | % Std | Cluster Robust | % Robust | Canal Std / Rob | Segmento Std / Rob | LT med. Std / Rob |
|-------------|-------|----------------|----------|-----------------|---------------------|-------------------|
| 0 | 38,23% | 1 | 49,92% | TA/TO / TA/TO | Online TA / Online TA | 76 / 64 |
| 1 | 53,11% | 0 | 49,77% | TA/TO / TA/TO | Online TA / Groups | 78 / 81 |
| 2 | 5,73% | 2 | **0,12%** | TA/TO / TA/TO | Online TA / Groups | **25 / 244** |
| 3 | 2,93% | 3 | 0,19% | Corporate / Corporate | Corporate / Corporate | 5 / 4 |

A comparação entre partições obtidas com StandardScaler e RobustScaler, com k fixo em quatro e alinhamento óptimo de clusters pelo algoritmo húngaro, revela concordância global de perfis de 0,75. Os clusters zero e um da solução Standard, que juntos concentram mais de noventa por cento das reservas com canal modal TA/TO e segmento Online TA, são redistribuídos entre os clusters um e zero da solução Robust de forma a inverter parcialmente a hierarquia de tamanhos. O caso mais eloquente é o cluster dois da solução Standard, que reúne 5,73% das reservas com mediana de *lead time* de vinte e cinco dias e segmento Online TA, e que no Robust é absorvido por um micro-cluster de 0,12% com mediana de *lead time* de duzentos e quarenta e quatro dias e segmento modal Groups. Este par claramente não representa o mesmo fenómeno de negócio; é um artefacto da mudança de escala. O cluster corporativo, por contraste, permanece estável entre escalonadores, com canal e segmento Corporate e *lead time* mediano de cerca de cinco dias em ambas as soluções.

### 4.7 Sensibilidade SemADR versus ComADR no subconjunto pós-IQR

**Tabela 9.** Perfis no subconjunto partilhado (`cluster_profile_sensitivity.csv`).

| Cluster | Representação | N | % | Lead time med. | Noites med. | Canal | Segmento |
|---------|---------------|---|---|----------------|-------------|-------|----------|
| 0 | Std-SemADR | 37 121 | 32,74 | 48 | 3 | TA/TO | Online TA |
| 0 | Std-ComADR | 40 411 | 35,64 | 49 | 3 | TA/TO | Online TA |
| 1 | Std-SemADR | 15 138 | 13,35 | 133 | 7 | TA/TO | Online TA |
| 3 | Std-ComADR | 14 431 | 12,73 | 136 | 7 | TA/TO | Online TA |
| 2 | Std-SemADR | 57 857 | 51,03 | 84 | 2 | TA/TO | Groups |
| 1 | Std-ComADR | 55 272 | 48,75 | 87 | 2 | TA/TO | Groups |
| 3 | Std-SemADR | 3 264 | 2,88 | 4 | 1 | Corporate | Corporate |
| 2 | Std-ComADR | 3 266 | 2,88 | 4 | 1 | Corporate | Corporate |

Quando ambos os modelos são treinados no mesmo subconjunto de 113 380 reservas restantes após o filtro IQR de ADR, a concordância global de perfis alinhados atinge 1,0. Isto significa que, para cada par de clusters correspondentes, o canal modal coincide, o segmento modal coincide e as medianas de *lead time* estão dentro da tolerância de trinta dias. A inclusão do ADR como variável numérica na variante ComADR altera sobretudo os tamanhos relativos dos grandes clusters online — por exemplo, o cluster de reservas com estadias longas e muito antecipadas passa de treze por cento para cerca de doze a treze por cento conforme a codificação — mas não muda a leitura qualitativa de que existe um bloco Groups/Online de massa, um bloco de estadias muito longas e um nicho Corporate de resposta imediata. Esta evidência suporta a opção SemADR para o modelo principal por parcimónia e por evitar que a segmentação se confunda com estratificação tarifária.

### 4.8 Perfis finais da solução adoptada

**Tabela 11.** Perfil completo (`cluster_profile_k4.csv`).

| Cluster | N reservas | % | Lead time (dias) | Noites | Canal | Segmento | Depósito |
|---------|------------|---|------------------|--------|-------|----------|----------|
| 0 | 44 891 | 38,23 | 76 | 4 | TA/TO | Online TA | No Deposit |
| 1 | 62 366 | 53,11 | 78 | 2 | TA/TO | Online TA | No Deposit |
| 2 | 6 726 | 5,73 | 25 | 3 | TA/TO | Online TA | No Deposit |
| 3 | 3 446 | 2,93 | 5 | 2 | Corporate | Corporate | No Deposit |

A solução oficial do projecto é a partição de quatro clusters produzida por K-Means na representação Std-SemADR sobre o dataset completo. O cluster um é o maior, com 62 366 reservas e cinquenta e três por cento do total. O seu perfil combina canal modal TA/TO, segmento Online TA, mediana de *lead time* de setenta e oito dias e mediana de duas noites de estadia. Trata-se do padrão clássico de reserva OTA antecipada para estadias curtas, compatível com city-breaks e escapadinhas reservadas com grande antecedência através de plataformas online.

O cluster zero, com 44 891 reservas e trinta e oito por cento do total, partilha o mesmo canal e segmento modais, mas difere na mediana de quatro noites de estadia com antecedência semelhante de setenta e seis dias. A distinção entre cluster zero e cluster um não está no canal de distribuição, que é idêntico no modo, mas na duração da estadia. Para o negócio, isto sugere políticas de pacote e upsell de noches adicionais dirigidas ao cluster zero, enquanto o cluster um pode ser alvo de campanhas de ocupação rápida e estadias curtas.

O cluster dois, com 6 726 reservas e 5,73% do total, mantém canal TA/TO e segmento Online TA, mas a mediana de *lead time* cai para vinte e cinco dias. Este grupo aproxima-se de reservas online com horizonte de planeamento mais curto, potencialmente mais sensíveis a disponibilidade de última hora e a promoções de curto prazo.

O cluster três, com 3 446 reservas e 2,93% do total, é qualitativamente distinto. O canal e o segmento modais são Corporate, a mediana de *lead time* é de apenas cinco dias e a estadia mediana é de duas noites. Este é o único segmento claramente B2B do modelo e deve ser tratado separadamente em estratégia comercial, com contratos negociados e não com campanhas OTA de massa.

O módulo automático de qualidade assinalou que os modais de canal TA/TO e segmento Online TA se repetem nos clusters zero, um e dois, gerando três avisos de modais duplicados em `profile_quality_flags.csv`. Esta repetição não invalida o modelo, mas obriga o analista a basear a interpretação nas medianas de *lead time* e no número de noites, e não apenas nos valores modais. O mesmo ficheiro regista que a representação Robust-SemADR com k igual a quatro produz um micro-cluster de 0,12%, confirmando que o RobustScaler foi utilizado apenas como teste de sensibilidade e não como base da segmentação final.

### 4.9 Extensão E4: PCA e clustering no espaço reduzido

**Tabela 12.** Espaço original (54 dim.) — K-Means, média de 5 seeds.

| \(k\) | Silhouette | ± std | CH | DB | ± std | Tempo (s) |
|------|------------|-------|-----|-----|-------|-----------|
| 3 | 0,0973 | 0,0067 | 9 155,27 | 2,6465 | 0,0966 | 1,35 |
| 4 | 0,0937 | 0,0098 | 8 984,95 | 2,3517 | 0,1238 | 1,08 |
| 5 | 0,0948 | 0,0055 | 8 893,77 | 2,1198 | 0,0674 | 1,49 |
| 6 | 0,0928 | 0,0023 | 9 001,61 | 1,9391 | 0,1430 | 1,47 |
| 7 | 0,1019 | 0,0078 | 9 259,47 | 1,8132 | 0,1233 | 1,56 |
| 8 | 0,1052 | 0,0043 | 9 429,59 | 1,8111 | 0,0312 | 1,73 |

**Tabela 13.** Espaço PCA-18 — K-Means, média de 5 seeds.

| \(k\) | Silhouette | ± std | CH | DB | ± std | Tempo (s) |
|------|------------|-------|-----|-----|-------|-----------|
| 3 | 0,1429 | **0,0603** | 10 178,86 | 2,2709 | 0,3532 | 0,64 |
| 4 | 0,1171 | 0,0119 | 10 321,86 | 2,1024 | 0,1212 | 0,61 |
| 5 | 0,1083 | 0,0104 | 10 071,80 | 2,0238 | 0,1148 | 0,80 |
| 6 | 0,1183 | 0,0077 | 10 404,66 | 1,8084 | 0,0583 | 0,77 |
| 7 | **0,1324** | 0,0085 | 10 759,98 | **1,6806** | 0,0623 | 0,86 |
| 8 | 0,1314 | 0,0063 | 11 383,66 | 1,5552 | 0,0774 | 1,01 |

A extensão E4 investiga se a projeção linear por análise de componentes principais melhora o clustering face ao espaço original de cinquenta e quatro dimensões utilizado neste módulo. Retiveram-se dezoito componentes, correspondentes a pelo menos noventa por cento da variância acumulada, e repetiu-se o *sweep* de K-Means com cinco *seeds*. Para k igual a quatro, o valor de referência do projecto principal, o Silhouette médio sobe de 0,0937 no espaço original para 0,1171 no espaço PCA, um ganho relativo de cerca de vinte e cinco por cento, e o Davies–Bouldin diminui, sugerindo clusters mais compactos na representação reduzida. No entanto, o melhor Silhouette global no espaço PCA ocorre em k igual a sete, não em k igual a quatro, o que demonstra que a redução de dimensionalidade altera a geometria do problema e desloca o óptimo de k. Mais preocupante ainda, em k igual a três no espaço PCA, o desvio padrão do Silhouette entre *seeds* atinge 0,0603, muito superior ao observado no espaço original, indicando soluções menos estáveis.

O *caveat* metodológico exigido pelo enunciado da extensão E4 é central: a PCA preserva direcções de máxima variância marginal, que não coincidem necessariamente com direcções que separam clusters. Por isso, embora as métricas internas possam melhorar em certas configurações, a segmentação oficial do projecto permanece no espaço Std-SemADR sem PCA, onde cada dimensão do *one-hot encoding* e cada variável numérica escalada mantêm interpretação directa para o analista de negócio.

---

## 5. Discussão

O estudo demonstra que o dataset de reservas hoteleiras admite uma segmentação em quatro grupos com significado operacional reconhecível, embora a separação geométrica medida pelos índices clássicos seja modesta em valor absoluto. Esta modestia não é surpresa: espaços de dezenas de dimensões com codificação *one-hot* esparsa raramente produzem Silhouettes elevados, porque a distância euclidiana deixa de ser uma proxy perfeita de semelhança perceptiva entre tipos de cliente.

A decisão de excluir o cancelamento e outras variáveis de desfecho do input é a premissa que define o significado dos clusters. Os grupos descrevem **comportamento de reserva à data da confirmação**, não risco de *no-show* nem probabilidade de cancelamento. Qualquer análise futura que cruze clusters com taxas de cancelamento deve ser explicitamente rotulada como estudo *post-hoc* supervisionado, separado da segmentação não supervisionada aqui reportada.

A escolha de k igual a quatro em vez de k igual a oito ilustra um princípio mais geral de governação de modelos de clustering aplicados a grandes volumes transaccionais: métricas internas devem ser subordinadas a critérios de utilidade e tamanho mínimo de segmento. Ignorar o micro-cluster de 0,12% em k igual a oito não é desprezar evidência estatística; é recusar confundir optimização algorítmica com segmentação accionável.

A comparação entre famílias de algoritmos reforça que números impressionantes em subamostras — como o Silhouette de 0,29 do Ward em vinte mil pontos — não devem ser generalizados sem qualificação. O Ward foi útil como contraponto hierárquico e como fonte de diagnóstico visual, mas a partição oficial permanece no K-Means sobre o dataset completo. O iK-Means mostrou que a descoberta automática de doze grupos não produz necessariamente uma solução melhor nem mais interpretável do que um *sweep* principiado com regra de tamanho mínimo.

A estabilidade moderada medida pelo ARI implica que qualquer implementação operacional deve prever re-estimação periódica dos clusters e comunicação de incerteza aos decisores. A alta concordância de perfis, por outro lado, tranquiliza quanto à narrativa dominante: existe um bloco massivo de reservas online e um nicho corporativo pequeno mas distinto, e esta leitura sobrevive a mudanças de *seed* mesmo quando as fronteiras numéricas se deslocam.

Entre as limitações, a métrica euclidiana em *one-hot* trata todos os níveis categóricos como igualmente distantes, o que pode não reflectir semântica de negócio. Não foram exploradas famílias de densidade nem modelos de mistura com selecção por AIC ou BIC. A extensão E4 não substitui a interpretabilidade das variáveis originais. Variáveis excluídas por prudência de *leakage* podem esconder heterogeneidade relevante noutro instante de segmentação. Do ponto de vista ético, trata-se de dados reais sob licença aberta: os resultados devem ser apresentados de forma agregada, sem possibilidade de re-identificação.

---

## 6. Conclusões

Foi implementado e documentado um estudo de clustering reprodutível sobre 117 429 reservas hoteleiras, com pipeline automatizada, quatro representações experimentais, três famílias de algoritmos, dez testes de estabilidade e extensão PCA integrada no mesmo protocolo de avaliação.

A segmentação oficial adoptada é K-Means com quatro clusters na representação Std-SemADR, seleccionada por máximo Silhouette entre configurações admissíveis com cluster mínimo de pelo menos um por cento das reservas. Os resultados quantitativos de referência são Silhouette 0,1256, menor cluster 2,93%, ARI médio 0,6675 e concordância de perfis 0,8889. A interpretação qualitativa distingue três grupos majoritários no universo online/OTA, diferenciados por antecedência e duração da estadia, e um grupo corporativo de resposta rápida.

O repositório, executado através de `python runAll.py`, regenera todas as tabelas e figuras citadas. Como evolução natural do trabalho, recomenda-se perfilagem *post-hoc* do cancelamento por cluster sem usar essa variável no *fit*, exploração de modelos de mistura, e análise de anomalias condicionadas aos clusters conforme a extensão E1 do enunciado.

---

## Referências

António, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

Unidade curricular *Unsupervised Learning* (2025/2026). Enunciado do projecto — *A Reproducible Experimental Study of Clustering*.

---

## Apêndice — Artefactos reprodutíveis

O ficheiro `experiments.csv` contém as trinta e três experiências da corrida de referência. Os perfis finais estão em `cluster_profile_k4.csv`. A estabilidade está em `stability_report.csv`. A extensão PCA está em `e4_pca_results.csv`. As figuras exploratórias e de diagnóstico estão em `src/graficos_relatorio/`. Para actualizar todos os números, executar `python runAll.py` em `hotel_booking_course_release_v1_pack/` e verificar o novo `run_id` em `pipeline_run.json`.
