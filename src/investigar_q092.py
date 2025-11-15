"""
Investigar a estrutura da variável Q092 (depressão)
"""

import pandas as pd

df = pd.read_excel('../data/dicionario_PNS_microdados_2019_23062023.xls', engine='xlrd')

# Encontrar Q092
idx = df[df['Unnamed: 2'].astype(str).str.contains('Q092', na=False)].index[0]

print("="*80)
print("VARIÁVEL Q092 - DIAGNÓSTICO DE DEPRESSÃO")
print("="*80)

for i in range(idx-2, min(idx+15, len(df))):
    r = df.iloc[i]
    pos = r.iloc[0]
    tam = r.iloc[1]
    cod = r.iloc[2]
    desc = str(r.iloc[4])[:80] if pd.notna(r.iloc[4]) else ''
    cat = r.iloc[5]
    val = r.iloc[6]
    
    print(f"\n{i}: Pos={pos} | Tam={tam} | Cod={cod}")
    if desc:
        print(f"   Desc: {desc}")
    if pd.notna(cat) and pd.notna(val):
        print(f"   {cat} = {val}")

# Verificar se Q092 é condicional (só responde quem tem depressão)
print("\n" + "="*80)
print("CONTEXTO: Verificando se Q092 é condicional...")
print("="*80)

# Procurar perguntas anteriores
for i in range(max(0, idx-20), idx):
    r = df.iloc[i]
    cod = str(r.iloc[2])
    desc = str(r.iloc[4])
    
    if 'depress' in desc.lower() or 'Q09' in cod:
        print(f"\n{cod}: {desc[:100]}")
