"""
Script para gerar CSV final com dados da PNS 2019 para o projeto de depressão
Inclui apenas variáveis necessárias, bem processadas e prontas para análise
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("GERANDO DATASET FINAL PARA O PROJETO DE DEPRESSÃO")
print("="*80)
print(f"Início: {datetime.now().strftime('%H:%M:%S')}\n")

# ==============================================================================
# 1. ESPECIFICAÇÃO DAS COLUNAS DO ARQUIVO PNS_2019.txt
# ==============================================================================

colspecs = [
    # IDENTIFICAÇÃO
    (0, 2),          # UF
    
    # DEMOGRÁFICAS
    (297, 300),      # C008 - Idade em anos
    (300, 301),      # C009 - Sexo
    (302, 303),      # C009B - Estado civil
    
    # EDUCAÇÃO  
    (309, 310),      # D009 - Sabe ler e escrever
    (312, 314),      # D011 - Anos de estudo
    
    # TRABALHO E RENDA
    (436, 437),      # E001 - Trabalhou na semana
    (1541, 1549),    # Rendimento domiciliar per capita
    
    # SAÚDE GERAL
    (805, 806),      # Q002 - Autoavaliação do estado de saúde
    (852, 853),      # Q030 - Tem plano de saúde
    
    # DOENÇAS CRÔNICAS
    (806, 807),      # Q002 - Hipertensão
    (813, 814),      # Q006 - Diabetes  
    (828, 829),      # Q020 - Colesterol alto
    (863, 864),      # Q063 - Doença do coração
    
    # DEPRESSÃO - VARIÁVEL ALVO E RELACIONADAS
    (986, 987),      # Q092 - Diagnóstico de depressão (VARIÁVEL ALVO)
    (987, 988),      # Q09201 - Medicamento receitado
    (988, 989),      # Q09202 - Usou medicamento (2 semanas)
    (989, 991),      # Q09301 - Idade no primeiro diagnóstico
    (991, 992),      # Q094 - Vai ao médico regularmente
    (994, 995),      # Q09605 - Faz psicoterapia
    (995, 996),      # Q09606 - Toma medicamentos
    (1006, 1007),    # Q109 - Grau de limitação
    
    # ESTILO DE VIDA
    (1087, 1088),    # P001 - Fuma atualmente
    (1090, 1091),    # P004 - Frequência fumo
    (1103, 1104),    # P027 - Consumo de álcool
    (1105, 1106),    # P02701 - Frequência álcool
    (1153, 1154),    # P034 - Atividade física (150min/semana)
    (1156, 1157),    # P036 - Atividade física deslocamento
    
    # PESO E ALTURA (para calcular IMC)
    (1223, 1227),    # P00101 - Peso (kg)
    (1227, 1230),    # P00401 - Altura (cm)
    
    # SAÚDE MENTAL (PHQ-2 simplificado)
    (949, 950),      # Q06301 - Pouco interesse/prazer
    (951, 952),      # Q06302 - Sentindo-se deprimido
]

names = [
    'uf',
    'idade',
    'sexo',
    'estado_civil',
    'sabe_ler',
    'anos_estudo',
    'trabalha',
    'renda_familiar',
    'estado_saude',
    'plano_saude',
    'hipertensao',
    'diabetes',
    'colesterol',
    'doenca_coracao',
    'diagnostico_depressao',  # VARIÁVEL ALVO
    'med_receitado',
    'usou_medicamento',
    'idade_diag_depressao',
    'vai_medico_regular',
    'faz_psicoterapia',
    'toma_medicamentos',
    'grau_limitacao',
    'fuma',
    'freq_fumo',
    'consome_alcool',
    'freq_alcool',
    'atividade_fisica',
    'atividade_deslocamento',
    'peso',
    'altura',
    'phq_interesse',
    'phq_deprimido',
]

print(f"📊 Total de variáveis a extrair: {len(names)}")
print(f"🎯 Variável alvo: diagnostico_depressao")

# ==============================================================================
# 2. CARREGAR DADOS
# ==============================================================================

print("\n" + "="*80)
print("CARREGANDO DADOS DA PNS_2019.txt...")
print("="*80)

df = pd.read_fwf(
    '../data/PNS_2019.txt',
    colspecs=colspecs,
    names=names,
    encoding='latin1',
    dtype=str  # Ler tudo como string primeiro
)

print(f"✓ Carregado: {len(df):,} registros")

# ==============================================================================
# 3. PROCESSAR E LIMPAR DADOS
# ==============================================================================

print("\n" + "="*80)
print("PROCESSANDO E LIMPANDO DADOS...")
print("="*80)

# Converter para tipos apropriados
print("\n1️⃣ Convertendo tipos de dados...")

# Numéricas
numericas = ['idade', 'anos_estudo', 'peso', 'altura', 'idade_diag_depressao']
for col in numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Renda (remover espaços e converter)
df['renda_familiar'] = pd.to_numeric(df['renda_familiar'].str.strip(), errors='coerce')

# Categóricas -> converter para int
categoricas = [c for c in names if c not in numericas and c != 'renda_familiar']
for col in categoricas:
    df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')

print("✓ Tipos convertidos")

# ==============================================================================
# 4. PROCESSAR VARIÁVEL ALVO
# ==============================================================================

print("\n2️⃣ Processando variável alvo (diagnostico_depressao)...")

# Manter apenas registros com resposta válida (1=Sim, 2=Não)
df_valid = df[df['diagnostico_depressao'].isin([1, 2])].copy()
print(f"   Registros com resposta válida: {len(df_valid):,}")

# Transformar: 1=Sim (depressão) -> 1, 2=Não -> 0
df_valid['diagnostico_depressao'] = df_valid['diagnostico_depressao'].map({1: 1, 2: 0})

dist = df_valid['diagnostico_depressao'].value_counts()
print(f"   Distribuição:")
print(f"      Com depressão: {dist.get(1, 0):,} ({dist.get(1, 0)/len(df_valid)*100:.1f}%)")
print(f"      Sem depressão: {dist.get(0, 0):,} ({dist.get(0, 0)/len(df_valid)*100:.1f}%)")

# ==============================================================================
# 5. PROCESSAR OUTRAS VARIÁVEIS
# ==============================================================================

print("\n3️⃣ Processando outras variáveis...")

# Sexo: 1=Masculino -> 0, 2=Feminino -> 1
df_valid['sexo'] = df_valid['sexo'].map({1: 0, 2: 1})

# Estado civil: 1=Casado, 2=Separado, 3=Divorciado, 4=Viúvo, 5=Solteiro
# Simplificar: 0=Solteiro/Divorciado/Separado/Viúvo, 1=Casado
df_valid['estado_civil'] = df_valid['estado_civil'].apply(lambda x: 1 if x == 1 else 0 if x in [2,3,4,5] else np.nan)

# Trabalha: 1=Sim -> 1, 2=Não -> 0
df_valid['trabalha'] = df_valid['trabalha'].map({1: 1, 2: 0})

# Estado de saúde: 1=Muito bom, 2=Bom, 3=Regular, 4=Ruim, 5=Muito ruim
# Manter como está (escala ordinal)

# Plano de saúde: 1=Sim -> 1, 2=Não -> 0
df_valid['plano_saude'] = df_valid['plano_saude'].map({1: 1, 2: 0})

# Doenças crônicas: 1=Sim -> 1, 2=Não -> 0
for col in ['hipertensao', 'diabetes', 'colesterol', 'doenca_coracao']:
    df_valid[col] = df_valid[col].map({1: 1, 2: 0})

# Fuma: 1=Sim diariamente, 2=Menos que diariamente, 3=Não fuma
df_valid['fuma'] = df_valid['fuma'].apply(lambda x: 1 if x in [1, 2] else 0 if x == 3 else np.nan)

# Álcool: 1=Sim -> 1, 2=Não -> 0
df_valid['consome_alcool'] = df_valid['consome_alcool'].map({1: 1, 2: 0})

# Atividade física: 1=Sim -> 1, 2=Não -> 0
df_valid['atividade_fisica'] = df_valid['atividade_fisica'].map({1: 1, 2: 0})

print("✓ Variáveis categóricas processadas")

# ==============================================================================
# 6. CRIAR VARIÁVEIS DERIVADAS
# ==============================================================================

print("\n4️⃣ Criando variáveis derivadas...")

# IMC (Índice de Massa Corporal)
df_valid['imc'] = df_valid['peso'] / ((df_valid['altura'] / 100) ** 2)
df_valid['imc'] = df_valid['imc'].replace([np.inf, -np.inf], np.nan)
print(f"   IMC calculado para {df_valid['imc'].notna().sum():,} registros")

# Score PHQ-2 simplificado (soma de interesse e deprimido)
# Valores: 1=Nenhuma vez, 2=Vários dias, 3=Mais da metade dos dias, 4=Quase todos os dias
# Recodificar para 0-3
df_valid['phq_interesse_score'] = df_valid['phq_interesse'].apply(lambda x: x-1 if x in [1,2,3,4] else np.nan)
df_valid['phq_deprimido_score'] = df_valid['phq_deprimido'].apply(lambda x: x-1 if x in [1,2,3,4] else np.nan)
df_valid['phq2_total'] = df_valid['phq_interesse_score'] + df_valid['phq_deprimido_score']

# Contagem de doenças crônicas
df_valid['num_doencas_cronicas'] = (
    df_valid['hipertensao'].fillna(0) +
    df_valid['diabetes'].fillna(0) +
    df_valid['colesterol'].fillna(0) +
    df_valid['doenca_coracao'].fillna(0)
)

print("✓ Variáveis derivadas criadas")

# ==============================================================================
# 7. SELECIONAR COLUNAS FINAIS
# ==============================================================================

print("\n5️⃣ Selecionando colunas finais...")

# Colunas finais para o projeto
colunas_finais = [
    # ALVO
    'diagnostico_depressao',
    
    # DEMOGRÁFICAS
    'idade',
    'sexo',
    'estado_civil',
    'uf',
    
    # SOCIOECONÔMICAS
    'anos_estudo',
    'sabe_ler',
    'trabalha',
    'renda_familiar',
    
    # SAÚDE GERAL
    'estado_saude',
    'plano_saude',
    'imc',
    'num_doencas_cronicas',
    
    # DOENÇAS CRÔNICAS
    'hipertensao',
    'diabetes',
    'colesterol',
    'doenca_coracao',
    
    # ESTILO DE VIDA
    'fuma',
    'consome_alcool',
    'atividade_fisica',
    
    # SAÚDE MENTAL
    'phq2_total',
    'grau_limitacao',
    'vai_medico_regular',
    'faz_psicoterapia',
    'toma_medicamentos',
]

df_final = df_valid[colunas_finais].copy()

print(f"✓ {len(colunas_finais)} colunas selecionadas")

# ==============================================================================
# 8. ESTATÍSTICAS FINAIS
# ==============================================================================

print("\n" + "="*80)
print("INFORMAÇÕES DO DATASET FINAL")
print("="*80)

print(f"\n📊 Dimensões: {df_final.shape[0]:,} linhas × {df_final.shape[1]} colunas")
print(f"\n📋 Variáveis:")
for col in df_final.columns:
    non_null = df_final[col].notna().sum()
    pct = non_null / len(df_final) * 100
    print(f"   {col:30s} - {non_null:5,} valores ({pct:5.1f}%)")

print(f"\n🎯 Variável Alvo (diagnostico_depressao):")
print(df_final['diagnostico_depressao'].value_counts().to_string())

print(f"\n📈 Estatísticas Descritivas:")
print(df_final.describe())

# ==============================================================================
# 9. SALVAR ARQUIVO FINAL
# ==============================================================================

print("\n" + "="*80)
print("SALVANDO DATASET FINAL...")
print("="*80)

output_file = '../data/dataset_pns_depressao_final.csv'
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ SUCESSO!")
print(f"📁 Arquivo salvo: {output_file}")
print(f"📊 Tamanho: {df_final.shape[0]:,} registros × {df_final.shape[1]} variáveis")
print(f"💾 Memória: {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n⏱️  Fim: {datetime.now().strftime('%H:%M:%S')}")

print("\n" + "="*80)
print("✓ DATASET PRONTO PARA O PROJETO!")
print("="*80)
print("\n💡 Próximo passo: Abrir o notebook e usar este arquivo CSV")
print(f"   Arquivo: {output_file}")
