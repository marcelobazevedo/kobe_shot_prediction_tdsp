
# Kobe Bryant Shot Prediction - Framework TDSP

## Projeto de Engenharia de Machine Learning - Previsão de Acertos do Kobe Bryant

### Objetivo

Desenvolver um preditor de arremessos de Kobe Bryant usando duas abordagens: Regressão e Classificação. O projeto deve prever se Kobe Bryant acertou ou errou um arremesso, utilizando o Framework TDSP (Team Data Science Process).

---

## Como executar o projeto

### Criar ambiente Conda e instalar dependências:
Com YAML 
```bash
conda env create -f environment.yml
```

### Executar o pipeline completo do plrojeto:
```bash
python code/pipeline.py
```

### Exibir o dashboard:
```bash
streamlit run outuputs/dashboards/streamlit_app.py
```

### Abrir o MLflow:
```bash
mlflow ui
```
Acessar: [http://localhost:5000](http://localhost:5000)


## Principais tecnologias utilizadas no projeto 
- Python 3.10
- Pycaret 3.0.0
- Seaborn 0.13.2
- Pandas 2.2.3
- Scikit-Learn 1.5.1
- MLflow 2.21.2
- Streamlit 1.43.2
- NumPy 2.1.3
- Matplotlib 3.9.2

## Métricas Finais (Produção)

| **Métrica**           | **Valor**               |
|-----------------------|------------------------|
| Modelo Escolhido      | Árvore de Decisão       |
| Log Loss (Produção)   | 0.5698                  |
| F1-Score (Produção)   | 0.5324                  |

O modelo de **Árvore de Decisão** foi selecionado para produção por apresentar melhor desempenho geral em métricas críticas como `F1-Score` e `Log Loss` em comparação à Regressão Logística. Além disso, a **Árvore de Decisão** oferece maior interpretabilidade e melhor desempenho na base de produção.  

Embora o `Log Loss` não seja perfeito, o modelo apresentou um desempenho consistente em dados fora da amostra, sugerindo que está bem ajustado para o problema.


## Estrutura do Projeto

O projeto segue a estrutura proposta pelo TDSP:

```
📁 kobe_shot_prediction_tdsp/
├── code/
│   ├── preparacao_dados.py
│   ├── treinamento.py
│   ├── aplicacao.py
│   ├── pipeline.py
├── data/
│   ├── raw/
│   │   ├── dataset_kobe_dev.parquet
│   │   └── dataset_kobe_prod.parquet
│   ├── processed/
│       ├── data_filtered.parquet
│       ├── base_train.parquet
│       └── base_test.parquet
├── docs/
│   ├── diagramas/
│   ├── relatorios/
├── outputs/
│   └── dashboards/
│       └── streamlit_app.py
├── README.md
├── requirements.txt
└── environment.yml
```

---

# **Respostas**

### Questão 1) A solução criada nesse projeto deve ser disponibilizada em repositório git e disponibilizada em servidor de repositórios (Github (recomendado), Bitbucket ou Gitlab). O projeto deve obedecer o Framework TDSP da Microsoft (estrutura de arquivos, arquivo requirements.txt e arquivo README - com as respostas pedidas nesse projeto, além de outras informações pertinentes). Todos os artefatos produzidos deverão conter informações referentes a esse projeto (não serão aceitos documentos vazios ou fora de contexto). Escreva o link para seu repositório.
### Resposta: https://github.com/marcelobazevedo/kobe_shot_prediction_tdsp
---
### Questão 2) Iremos desenvolver um preditor de arremessos usando duas abordagens (regressão e classificação) para prever se o "Black Mamba" (apelido de Kobe) acertou ou errou a cesta. Baixe os dados de desenvolvimento e produção aqui (datasets: dataset_kobe_dev.parquet e dataset_kobe_prod.parquet). Salve-os numa pasta /data/raw na raiz do seu repositório. Para começar o desenvolvimento, desenhe um diagrama que demonstra todas as etapas necessárias para esse projeto, desde a aquisição de dados, passando pela criação dos modelos, indo até a operação do modelo.
### Resposta: 

1. **Aquisição de Dados**
   - Coleta dos arquivos e armazenamento na pasta `data/raw`.

2. **Pré-processamento**
   - Remoção de nulos, seleção de colunas relevantes e salvamento em `data/processed`.

3. **Separação Treino/Teste**
   - Divisão estratificada 80/20 dos dados e armazenamento em `base_train.parquet` e `base_test.parquet`.

4. **Treinamento**
   - Aplicação de modelos de Regressão Logística e Árvore de Decisão utilizando PyCaret e MLFlow.

5. **Avaliação**
   - Métricas calculadas: Log-Loss, F1-score, entre outras.
   - Seleção do melhor modelo baseado nas métricas calculadas.

6. **Deploy/Operacionalização**
   - Deploy do modelo utilizando MLFlow, API Flask ou Streamlit.

7. **Monitoramento**
   - Monitoramento contínuo das métricas de desempenho em produção.

8. **Atualização do Modelo**
   - Reavaliação e ajuste do modelo com novas coletas de dados.

### Diagrama das etapas necessária para o projeto

![Diagrama do Pipeline](./docs/diagramas/questao_2_fluxograma.png)

---
### Questão 3) Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines descritos anteriormente? A resposta deve abranger os seguintes aspectos:

#### - Rastreamento de experimentos;
#### - Funções de treinamento;
#### - Monitoramento da saúde do modelo;
#### - Atualização de modelo;
#### - Provisionamento (Deployment).
### Resposta: 

####  PyCaret
- Facilita o treinamento e a comparação de modelos através de funções simplificadas.
- Permite testar múltiplos algoritmos de classificação e regressão.

####  Scikit-Learn
- Fornece algoritmos de machine learning essenciais para o projeto.
- É utilizado em conjunto com o PyCaret para o treinamento dos modelos.

####  MLFlow
- Usado para rastrear experimentos e registrar métricas.
- Facilita o deploy do modelo final.
- Permite monitorar o desempenho do modelo.

####  Streamlit
- Permite a criação de dashboards interativos para visualização dos resultados.
- Facilita o monitoramento do desempenho do modelo.

---

### Questão 4) Com base no diagrama realizado na questão 2, aponte os artefatos que serão criados ao longo de um projeto. Para cada artefato, a descrição detalhada de sua composição.
### Resposta: 

- `data_filtered.parquet`: Conjunto de dados limpo e filtrado.
- `base_train.parquet`: Dados de treino (80%).
- `base_test.parquet`: Dados de teste (20%).
- `final_model.pkl`: Modelo treinado e registrado no MLFlow.
- **Dashboard Streamlit**: Visualização dos resultados.

---


### Questão 5) Implemente o pipeline de processamento de dados com o mlflow, rodada (run) com o nome `"PreparacaoDados"`:

- **a.** Os dados devem estar localizados em:
  - `/data/raw/dataset_kobe_dev.parquet` 
  - `/data/raw/dataset_kobe_prod.parquet` 

- **b.** Observe que há dados faltantes na base de dados! As linhas que possuem dados faltantes devem ser desconsideradas. Para esse exercício serão apenas consideradas as colunas: 
  - `lat`
  - `lon`
  - `minutes_remaining`
  - `period`
  - `playoffs`
  - `shot_distance`
  - `shot_made_flag` (variável alvo onde: `0` = erro, `1` = acerto)
  
  O dataset resultante será armazenado na pasta:  
  `/data/processed/data_filtered.parquet`  

  Ainda sobre essa seleção, qual a dimensão resultante do dataset?

- **c.** Separe os dados em treino (80%) e teste (20%) usando uma escolha aleatória e estratificada. 

  - Armazene os datasets resultantes em:  
    - `/data/processed/base_train.parquet`
    - `/data/processed/base_test.parquet`
    
  Explique como a escolha de treino e teste afetam o resultado do modelo final. Quais estratégias ajudam a minimizar os efeitos de viés de dados?

- **d.** Registre os parâmetros (% teste) e métricas (tamanho de cada base) no MLFlow.


### Resposta: 

A pipeline de processamento de dados inclui as seguintes etapas:
- Carregamento e limpeza dos dados.
- Seleção das colunas especificadas (`lat`, `lon`, `minutes_remaining`, `period`, `playoffs`, `shot_distance`, `shot_made_flag`).
- Remoção de linhas com valores ausentes.
- Separação dos dados em treino e teste (80% e 20%).
- Registro de métricas no MLFlow.

---
### Questão 6) Implementar o pipeline de treinamento do modelo com o MlFlow usando o nome "Treinamento" 
- **a.** Com os dados separados para treinamento, treine um modelo com regressão logística do sklearn usando a biblioteca pyCaret.
- **b.** Registre a função custo "log loss" usando a base de teste
- **c.** Com os dados separados para treinamento, treine um modelo de árvore de decisão do sklearn usando a biblioteca pyCaret.
- **d.** Registre a função custo "log loss" e F1_score para o modelo de árvore.
- **e.** Selecione um dos dois modelos para finalização e justifique sua escolha.

### Resposta: 
## 6. Pipeline de Treinamento do Modelo com o MLFlow (`Treinamento`)

Implementar o pipeline de treinamento do modelo com o MLFlow usando o nome `"Treinamento"`.

### a. Treinamento do Modelo de Regressão Logística

- Com os dados separados para treinamento, um modelo de **Regressão Logística** é treinado utilizando a biblioteca **PyCaret**.
- O PyCaret facilita o processo de configuração, treinamento, comparação de modelos e registro automático no MLFlow.

### b. Registro da Função Custo - `Log Loss` (Regressão Logística)

- A métrica **Log Loss** é calculada utilizando a base de teste (`base_test.parquet`).
- O resultado é registrado no MLFlow para posterior comparação com outros modelos.

### c. Treinamento do Modelo de Árvore de Decisão

- Com os dados separados para treinamento, um modelo de **Árvore de Decisão** é treinado utilizando a biblioteca **PyCaret**.
- Assim como na Regressão Logística, o PyCaret é utilizado para automatizar o processo de treinamento e registro de métricas no MLFlow.

### d. Registro das Métricas - `Log Loss` e `F1 Score` (Árvore de Decisão)

- As métricas **Log Loss** e **F1 Score** são calculadas utilizando a base de teste (`base_test.parquet`).
- Ambas são registradas no MLFlow, permitindo a comparação direta entre os modelos treinados.

### e. Seleção do Modelo Final

- O modelo selecionado para finalização foi a **Árvore de Decisão**, com base na comparação das métricas `Log Loss` e `F1 Score`.
- A **Árvore de Decisão** foi escolhida devido ao seu melhor desempenho em relação ao `F1 Score`, o que indica melhor equilíbrio entre precisão e revocação.
- Além disso, a árvore de decisão oferece maior interpretabilidade, facilitando a análise dos resultados.

> **Justificativa:** Embora o modelo de Regressão Logística apresente um desempenho satisfatório, a **Árvore de Decisão** obteve um melhor `F1 Score`, o que é relevante considerando a necessidade de precisão e revocação balanceadas no problema proposto.
---
### Questão 7) Registre o modelo de classificação e o sirva através do MLFlow (ou como uma API local, ou embarcando o modelo na aplicação). Desenvolva um pipeline de aplicação (aplicacao.py) para carregar a base de produção (/data/raw/dataset_kobe_prod.parquet) e aplicar o modelo. Nomeie a rodada (run) do mlflow como “PipelineAplicacao” e publique, tanto uma tabela com os resultados obtidos (artefato como .parquet), quanto log as métricas do novo log loss e f1_score do modelo.
- **a.** O modelo é aderente a essa nova base? O que mudou entre uma base e outra? Justifique.
- **b.** Descreva como podemos monitorar a saúde do modelo no cenário com e sem a disponibilidade da variável resposta para o modelo em operação.
- **c.** Descreva as estratégias reativa e preditiva de retreinamento para o modelo em operação.

### Resposta: 

## 7. Registro e Servir do Modelo com o MLFlow (`PipelineAplicacao`)

Implementar o pipeline de aplicação do modelo treinado, servir o modelo e registrar os resultados utilizando o **MLFlow** com o nome `"PipelineAplicacao"`.

---

### a. O modelo é aderente a essa nova base? O que mudou entre uma base e outra? Justifique.

- **Aderência do Modelo:** Após aplicar o modelo treinado na base de produção (`/data/raw/dataset_kobe_prod.parquet`), as métricas registradas (`Log Loss` e `F1 Score`) devem ser comparadas com aquelas obtidas durante o treinamento e teste iniciais.
- **Mudanças entre as bases:**
  - A base de produção pode apresentar distribuições diferentes nas variáveis de entrada (`lat`, `lng`, `minutes_remaining`, `period`, `playoffs`, `shot_distance`) em relação aos dados usados para o treinamento e teste (`dataset_kobe_dev.parquet`).
  - Caso as métricas calculadas na base de produção sejam significativamente piores do que aquelas obtidas durante o treinamento, é possível que o modelo não seja adequado para a nova base de produção.
- **Justificativa:** 
  - O modelo é considerado aderente se suas métricas mantiverem um desempenho consistente quando aplicado à nova base.
  - Desvios significativos indicam que o modelo deve ser reavaliado ou retreinado para melhor se adequar às características da base de produção.

---

### b. Descreva como podemos monitorar a saúde do modelo no cenário com e sem a disponibilidade da variável resposta para o modelo em operação.

1. **Cenário com a disponibilidade da variável resposta (`shot_made_flag`):**
   - Quando a variável resposta (`shot_made_flag`) está disponível, é possível monitorar o desempenho do modelo de maneira contínua.
   - Métricas como `Log Loss`, `F1 Score`, `Accuracy`, entre outras, podem ser calculadas e comparadas com os valores obtidos durante o treinamento inicial.
   - Essas métricas devem ser registradas no MLFlow regularmente para facilitar o monitoramento e análise da saúde do modelo.

2. **Cenário sem a disponibilidade da variável resposta (Produção sem labels):**
   - Sem a variável resposta (`shot_made_flag`), o monitoramento deve ser feito de forma indireta.
   - Possíveis abordagens incluem:
     - **Monitoramento de Distribuição:** Comparar a distribuição das previsões atuais com as distribuições obtidas durante o treinamento.
     - **Monitoramento da Confiança:** Acompanhar os `Confidence Scores` das previsões para identificar padrões anômalos.
     - **Detecção de Drift:** Detectar mudanças nas distribuições dos dados de entrada que possam indicar degradação do desempenho do modelo.
   - Este monitoramento pode ser registrado no MLFlow por meio de métricas customizadas ou artefatos gerados periodicamente.

---

### c. Descreva as estratégias reativa e preditiva de retreinamento para o modelo em operação.

1. **Estratégia Reativa:**
   - O retreinamento é realizado **após** a detecção de uma queda significativa no desempenho do modelo.
   - A degradação é identificada por meio de monitoramento contínuo das métricas (`Log Loss`, `F1 Score`) registradas no MLFlow.
   - Quando uma métrica cai abaixo de um limite pré-definido, o retreinamento é acionado.
   - Exemplo: Se o `F1 Score` cair abaixo de 0.50, o modelo é retreinado usando uma nova amostra de dados atualizados.

2. **Estratégia Preditiva:**
   - O retreinamento é realizado de maneira **preventiva**, antes que o desempenho do modelo se deteriore.
   - Técnicas de previsão são aplicadas para identificar quando o modelo precisa ser atualizado.
   - Pode envolver o uso de algoritmos de detecção de drift ou monitoramento contínuo das distribuições dos dados de entrada.
   - Este método é útil para evitar quedas bruscas de desempenho, mantendo o modelo sempre atualizado.

---

### Questão 8) Implemente um dashboard de monitoramento da operação usando Streamlit.

Um dashboard interativo foi implementado usando Streamlit para monitorar o desempenho do modelo.

```bash
streamlit run outputs/dashboards/streamlit_app.py
```
![Dashboard Streamlit](./docs/imagens/dashboard_streamlit.png)
---
