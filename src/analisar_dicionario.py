"""
Script para analisar o dicionário da PNS 2019 e encontrar variáveis de depressão
"""

import pandas as pd

print("="*80)
print("ANALISANDO DICIONÁRIO DA PNS 2019")
print("="*80)

# Ler o arquivo Excel
try:
    # Tentar ler com xlrd (formato .xls antigo)
    df_dict = pd.read_excel('../data/dicionario_PNS_microdados_2019_23062023.xls', engine='xlrd')
    print(f"\n✓ Dicionário carregado com sucesso!")
    print(f"Shape: {df_dict.shape}")
    print(f"\nColunas disponíveis:")
    print(df_dict.columns.tolist())
    
    print("\n" + "="*80)
    print("PRIMEIRAS 10 LINHAS DO DICIONÁRIO:")
    print("="*80)
    print(df_dict.head(10))
    
    # Procurar variáveis relacionadas à depressão
    print("\n" + "="*80)
    print("PROCURANDO VARIÁVEIS DE DEPRESSÃO...")
    print("="*80)
    
    # Buscar em todas as colunas por termos relacionados
    termos = ['depres', 'mental', 'psico', 'diagnos', 'saude mental', 'q002', 'q00201']
    
    for col in df_dict.columns:
        for termo in termos:
            mask = df_dict[col].astype(str).str.lower().str.contains(termo, na=False)
            if mask.any():
                print(f"\n🔍 Encontrado '{termo}' na coluna '{col}':")
                print(df_dict[mask][[col]].head(10))
    
    # Salvar informações do dicionário
    print("\n" + "="*80)
    print("SALVANDO INFORMAÇÕES DO DICIONÁRIO...")
    print("="*80)
    df_dict.to_csv('../data/dicionario_analise.csv', index=False, encoding='utf-8-sig')
    print("✓ Salvo em: ../data/dicionario_analise.csv")
    
except Exception as e:
    print(f"❌ Erro ao ler dicionário: {e}")
    print("\nTentando com openpyxl...")
    try:
        df_dict = pd.read_excel('../data/dicionario_PNS_microdados_2019_23062023.xls', engine='openpyxl')
        print(f"✓ Dicionário carregado!")
        print(df_dict.head())
    except Exception as e2:
        print(f"❌ Erro com openpyxl: {e2}")
