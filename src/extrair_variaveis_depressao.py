"""
Script para extrair informações detalhadas sobre variáveis de depressão
"""

import pandas as pd

print("="*80)
print("EXTRAINDO VARIÁVEIS DE DEPRESSÃO DO DICIONÁRIO PNS 2019")
print("="*80)

# Ler dicionário
df = pd.read_excel('../data/dicionario_PNS_microdados_2019_23062023.xls', engine='xlrd')

# Buscar Q00201
idx_q00201 = df[df['Unnamed: 2'].astype(str).str.contains('Q00201', na=False)].index[0]

print("\n📋 VARIÁVEL Q00201 - DEPRESSÃO")
print("="*80)
row = df.iloc[idx_q00201]
print(f"Posição inicial: {row.iloc[0]}")
print(f"Tamanho: {row.iloc[1]}")
print(f"Código: {row.iloc[2]}")
print(f"Quesito nº: {row.iloc[3]}")
print(f"Descrição: {row.iloc[4]}")

print(f"\nCategorias:")
for i in range(idx_q00201, idx_q00201+6):
    if i < len(df):
        cat = df.iloc[i]['Unnamed: 5']
        desc = df.iloc[i]['Unnamed: 6']
        if pd.notna(cat) and pd.notna(desc):
            print(f"  {cat} - {desc}")

# Buscar mais variáveis relacionadas
print("\n"+"="*80)
print("TODAS AS VARIÁVEIS RELACIONADAS À SAÚDE MENTAL/DEPRESSÃO")
print("="*80)

# Buscar linhas com "depres" ou "mental"
mask_depres = df['Unnamed: 4'].astype(str).str.lower().str.contains('depres', na=False)
mask_mental = df['Unnamed: 4'].astype(str).str.lower().str.contains('saúde mental', na=False)
mask = mask_depres | mask_mental

variaveis_saude_mental = df[mask][['Dicionário das variáveis da PNS 2019', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 4']]
variaveis_saude_mental = variaveis_saude_mental.dropna(subset=['Unnamed: 2'])
variaveis_saude_mental = variaveis_saude_mental[variaveis_saude_mental['Unnamed: 2'].str.startswith('Q', na=False)]

print(f"\nEncontradas {len(variaveis_saude_mental)} variáveis:")
for _, row in variaveis_saude_mental.iterrows():
    print(f"\n  Código: {row['Unnamed: 2']}")
    print(f"  Posição: {row['Dicionário das variáveis da PNS 2019']}")
    print(f"  Tamanho: {row['Unnamed: 1']}")
    print(f"  Pergunta: {str(row['Unnamed: 4'])[:80]}...")

print("\n"+"="*80)
print("✓ Análise concluída!")
print("="*80)
