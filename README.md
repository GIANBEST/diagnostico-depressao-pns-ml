# Projeto de Mineração de Dados - Classificação de Depressão

## Objetivo
Desenvolver um pipeline de machine learning para classificação binária de diagnóstico de depressão usando **dados reais da Pesquisa Nacional de Saúde (PNS) 2019 - IBGE**.

## Métricas de Sucesso
- **Recall >= 75%** (métrica principal)
- **Precisão >= 50%**

## Dados
- **Fonte**: Pesquisa Nacional de Saúde (PNS) 2019 - IBGE (dados públicos e de livre acesso)
- **Dataset processado**: 8.301 registros válidos de 293.726 registros originais
- **Variáveis**: 25 features (demográficas, socioeconômicas, saúde, estilo de vida)
- **Variável alvo**: `diagnostico_depressao` (91% com depressão, 9% sem depressão)
- **Licença**: Dados públicos do IBGE - livre uso para pesquisa e análise

### 📊 Sobre os Dados
Os dados utilizados são provenientes da **Pesquisa Nacional de Saúde (PNS) 2019**, uma pesquisa domiciliar de âmbito nacional realizada pelo IBGE. Os dados são **públicos e de livre acesso**, disponíveis em:
- **Site oficial**: https://www.ibge.gov.br/estatisticas/sociais/saude/9160-pesquisa-nacional-de-saude.html

O dataset incluído (`data/dataset_pns_depressao.csv`) é uma versão processada e otimizada contendo apenas as variáveis relevantes para classificação de depressão.

## Estrutura do Projeto
```
├── .venv/                         # Ambiente virtual Python
├── data/                          # Dados do projeto
│   └── dataset_pns_depressao.csv # Dataset (8.301 registros, 25 variáveis)
├── notebooks/                     # Jupyter notebooks com análises
│   └── 01_pipeline_completo.ipynb # Pipeline completo de ML (42 células)
├── src/                           # Scripts Python
│   ├── gerar_dataset.py          # Geração de dataset sintético
│   ├── gerar_csv_final.py        # Processamento dados PNS
│   ├── analisar_pns.py           # Análise estrutura PNS
│   ├── analisar_dicionario.py    # Análise dicionário variáveis
│   ├── extrair_variaveis_depressao.py # Extração variáveis
│   └── ...                        # Outros scripts auxiliares
├── results/                       # Resultados e visualizações
├── requirements.txt               # Dependências (7 bibliotecas)
├── run_pipeline.py                # Script auxiliar de execução
└── README.md                      # Este arquivo
```

## Instalação
```bash
pip install -r requirements.txt
```

## Uso

### Opção 1: Jupyter Notebook (Recomendado)
1. Abra `notebooks/01_pipeline_completo.ipynb` no VS Code
2. Selecione o kernel: Python 3.14.0 (.venv)
3. Execute todas as células: `Ctrl + Shift + P` → "Run All Cells"

### Opção 2: Script Python
```bash
python run_pipeline.py
```

## Características do Projeto
- **Dataset**: 8.301 registros com 25 variáveis
- **Desbalanceamento**: 91% com depressão, 9% sem depressão
- **Split**: 80/20 estratificado (treino/teste)
- **Pré-processamento**: Imputação (mediana) + Padronização (Z-score)
- **Balanceamento**: SMOTE (apenas no treino)
- **Validação**: Cruzada estratificada (k=5)
- **Modelos testados**: 5 (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
- **Random state**: 42 (reprodutibilidade)

## Resultados Obtidos
✅ **Melhor modelo**: Random Forest  
✅ **Recall**: 89.99% (critério: ≥75%)  
✅ **Precision**: 92.21% (critério: ≥50%)  
✅ **ROC-AUC**: 0.7186  
✅ **Accuracy**: 84.01%
