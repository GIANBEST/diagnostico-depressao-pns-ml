"""
Script para carregar o dataset COMPLETO da PNS 2019 (293.726 registros)
e preparar para análise de depressão
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("CARREGANDO DATASET COMPLETO DA PNS 2019")
print("="*80)
print(f"Início: {datetime.now().strftime('%H:%M:%S')}")

# Especificação de colunas
colspecs = [
    (0, 2),          # UF
    (8, 10),         # Número domicílio
    (19, 21),        # Idade
    (21, 22),        # Sexo
    (48, 49),        # Cor ou raça
    (22, 23),        # Estado civil
    (49, 50),        # Sabe ler
    (51, 52),        # Nível instrução
    (154, 155),      # Trabalha
    (805, 806),      # Estado saúde
    (806, 807),      # Doença crônica
    (986, 987),      # Diagnóstico depressão (Q092) - VARIÁVEL ALVO
    (987, 988),      # Medicamento receitado
    (988, 989),      # Usou medicamento
    (989, 991),      # Idade primeiro diagnóstico
    (991, 992),      # Vai ao médico regular
    (994, 995),      # Psicoterapia
    (995, 996),      # Toma medicamentos
    (997, 998),      # Medicamento público
    (1006, 1007),    # Grau limitação
    (949, 951),      # PHQ-9
    (1087, 1088),    # Fuma
    (1103, 1104),    # Álcool
    (1153, 1154),    # Atividade física
    (1223, 1227),    # Peso
    (1227, 1230),    # Altura
]

names = [
    'uf', 'num_domicilio', 'idade', 'sexo', 'cor_raca', 'estado_civil',
    'sabe_ler_escrever', 'nivel_instrucao', 'trabalha', 'estado_saude',
    'doenca_cronica', 'diagnostico_depressao', 'medicamento_receitado',
    'usou_medicamento_2sem', 'idade_primeiro_diagnostico', 'vai_medico_regular',
    'faz_psicoterapia', 'toma_medicamentos', 'medicamento_servico_publico',
    'grau_limitacao_depressao', 'phq9_score', 'fuma', 'consumo_alcool',
    'atividade_fisica', 'peso', 'altura'
]

print(f"\nCarregando {len(names)} variáveis...")
print("⏳ Isso pode levar alguns minutos...")

# Carregar dataset completo
df = pd.read_fwf(
    '../data/PNS_2019.txt',
    colspecs=colspecs,
    names=names,
    encoding='latin1'
)

print(f"\n✓ Dados carregados! ({datetime.now().strftime('%H:%M:%S')})")
print(f"Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

# Processar dados
print("\n" + "="*80)
print("PROCESSANDO DADOS...")
print("="*80)

# 1. Transformar variável alvo (1=Sim depressão, 2=Não, outros=NaN)
print("\n1️⃣ Processando variável alvo...")
df['diagnostico_depressao_original'] = df['diagnostico_depressao'].copy()
df['diagnostico_depressao'] = df['diagnostico_depressao'].apply(
    lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
)

print(f"Distribuição original: {df['diagnostico_depressao_original'].value_counts().to_dict()}")
print(f"Distribuição processada:")
print(df['diagnostico_depressao'].value_counts())
print(f"Missing values: {df['diagnostico_depressao'].isna().sum():,}")

# 2. Filtrar apenas pessoas com resposta válida para depressão
print("\n2️⃣ Filtrando registros com resposta válida...")
df_valido = df[df['diagnostico_depressao'].notna()].copy()
print(f"Registros com resposta válida: {len(df_valido):,}")
print(f"Taxa de depressão: {df_valido['diagnostico_depressao'].mean():.2%}")

# 3. Criar IMC
print("\n3️⃣ Criando variável IMC...")
df_valido['imc'] = df_valido['peso'] / ((df_valido['altura'] / 100) ** 2)
print(f"IMC calculado para {df_valido['imc'].notna().sum():,} registros")

# 4. Processar outras variáveis
print("\n4️⃣ Processando variáveis categóricas...")

# Sexo: 1=Masculino, 2=Feminino -> 0=M, 1=F
df_valido['sexo'] = df_valido['sexo'].apply(lambda x: 0 if x == 1 else (1 if x == 2 else np.nan))

# Trabalha: 1=Sim, 2=Não
df_valido['trabalha'] = df_valido['trabalha'].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))

# Fuma: 1=Sim diariamente, 2=Menos que diariamente, 3=Não fuma
df_valido['fuma'] = df_valido['fuma'].apply(lambda x: 1 if x in [1, 2] else (0 if x == 3 else np.nan))

print("✓ Variáveis processadas")

# 5. Informações finais
print("\n" + "="*80)
print("INFORMAÇÕES DO DATASET PROCESSADO")
print("="*80)
df_valido.info()

print("\n" + "="*80)
print("ESTATÍSTICAS DESCRITIVAS")
print("="*80)
print(df_valido.describe())

# 6. Salvar dataset processado
output_file = '../data/pns_2019_depressao_completo.csv'
print(f"\n💾 Salvando dataset processado em: {output_file}")
df_valido.to_csv(output_file, index=False)

print(f"\n✓ Dataset salvo com sucesso!")
print(f"Fim: {datetime.now().strftime('%H:%M:%S')}")

print("\n" + "="*80)
print("📊 RESUMO FINAL")
print("="*80)
print(f"Total de registros: {len(df_valido):,}")
print(f"Total de features: {df_valido.shape[1]}")
print(f"Casos de depressão: {df_valido['diagnostico_depressao'].sum():,} ({df_valido['diagnostico_depressao'].mean():.2%})")
print(f"Casos sem depressão: {(df_valido['diagnostico_depressao']==0).sum():,}")
print(f"\n✓ PRONTO PARA USO NO NOTEBOOK!")
