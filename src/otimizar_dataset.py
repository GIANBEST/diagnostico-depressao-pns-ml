"""
Script para criar versão otimizada e leve do dataset PNS
Mantém apenas variáveis essenciais para análise de depressão
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("CRIANDO DATASET OTIMIZADO DA PNS 2019")
print("="*80)
print(f"Início: {datetime.now().strftime('%H:%M:%S')}")

# Carregar dataset completo processado
print("\n📂 Carregando dataset processado...")
df = pd.read_csv('../data/pns_2019_depressao_completo.csv')
print(f"✓ Carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

print("\n" + "="*80)
print("SELECIONANDO APENAS VARIÁVEIS ESSENCIAIS")
print("="*80)

# Variáveis essenciais para o modelo de ML
colunas_essenciais = [
    # VARIÁVEL ALVO
    'diagnostico_depressao',
    
    # DEMOGRÁFICAS
    'idade',
    'sexo',
    'estado_civil',
    
    # SOCIOECONÔMICAS
    'nivel_instrucao',
    'trabalha',
    
    # SAÚDE GERAL
    'estado_saude',
    'doenca_cronica',
    'imc',
    
    # SAÚDE MENTAL / DEPRESSÃO
    'medicamento_receitado',
    'usou_medicamento_2sem',
    'vai_medico_regular',
    'faz_psicoterapia',
    'toma_medicamentos',
    'grau_limitacao_depressao',
    
    # ESTILO DE VIDA
    'fuma',
    'consumo_alcool',
    'atividade_fisica',
]

print(f"\nVariáveis selecionadas: {len(colunas_essenciais)}")
for i, col in enumerate(colunas_essenciais, 1):
    print(f"  {i:2d}. {col}")

# Criar dataset otimizado
df_otimizado = df[colunas_essenciais].copy()

print("\n" + "="*80)
print("LIMPEZA E OTIMIZAÇÃO")
print("="*80)

# 1. Remover linhas com muitos NaN
print("\n1️⃣ Removendo linhas com muitos valores faltantes...")
threshold = 0.5  # Remover linhas com mais de 50% de NaN
max_nan = len(colunas_essenciais) * threshold
linhas_antes = len(df_otimizado)
df_otimizado = df_otimizado[df_otimizado.isna().sum(axis=1) <= max_nan]
linhas_removidas = linhas_antes - len(df_otimizado)
print(f"   Linhas removidas: {linhas_removidas:,}")
print(f"   Linhas restantes: {len(df_otimizado):,}")

# 2. Converter tipos de dados para economizar memória
print("\n2️⃣ Otimizando tipos de dados...")
antes = df_otimizado.memory_usage(deep=True).sum() / 1024**2

# Converter floats para int8/int16 onde possível
colunas_binarias = ['diagnostico_depressao', 'sexo', 'trabalha', 'doenca_cronica', 
                    'medicamento_receitado', 'usou_medicamento_2sem', 'vai_medico_regular',
                    'faz_psicoterapia', 'toma_medicamentos', 'fuma']

for col in colunas_binarias:
    if col in df_otimizado.columns:
        df_otimizado[col] = df_otimizado[col].astype('Int8')

# Converter categorias para int8
colunas_categoricas = ['estado_civil', 'nivel_instrucao', 'estado_saude', 
                       'grau_limitacao_depressao', 'consumo_alcool', 'atividade_fisica']

for col in colunas_categoricas:
    if col in df_otimizado.columns:
        df_otimizado[col] = df_otimizado[col].astype('Int8')

# Idade e IMC como float32
if 'idade' in df_otimizado.columns:
    df_otimizado['idade'] = df_otimizado['idade'].astype('float32')
if 'imc' in df_otimizado.columns:
    df_otimizado['imc'] = df_otimizado['imc'].astype('float32')

depois = df_otimizado.memory_usage(deep=True).sum() / 1024**2
print(f"   Memória antes: {antes:.2f} MB")
print(f"   Memória depois: {depois:.2f} MB")
print(f"   Redução: {(1 - depois/antes)*100:.1f}%")

# 3. Estatísticas finais
print("\n" + "="*80)
print("ESTATÍSTICAS DO DATASET OTIMIZADO")
print("="*80)

print(f"\n📊 Dimensões: {df_otimizado.shape[0]:,} linhas × {df_otimizado.shape[1]} colunas")
print(f"\n🎯 Distribuição da variável alvo:")
print(df_otimizado['diagnostico_depressao'].value_counts())
print(f"   Taxa de depressão: {df_otimizado['diagnostico_depressao'].mean()*100:.2f}%")

print(f"\n📉 Valores faltantes por coluna:")
missing = df_otimizado.isna().sum()
missing_pct = (missing / len(df_otimizado) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentual': missing_pct
})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
print(missing_df)

print(f"\n💾 Informações do DataFrame:")
df_otimizado.info(memory_usage='deep')

# 4. Salvar dataset otimizado
print("\n" + "="*80)
print("SALVANDO DATASET OTIMIZADO")
print("="*80)

# CSV
output_csv = '../data/pns_2019_depressao_OTIMIZADO.csv'
df_otimizado.to_csv(output_csv, index=False)
tamanho_csv = pd.io.common.file_exists(output_csv)
import os
tamanho_csv_mb = os.path.getsize(output_csv) / 1024**2
print(f"\n✓ CSV salvo: {output_csv}")
print(f"  Tamanho: {tamanho_csv_mb:.2f} MB")

# Pickle (mais eficiente)
output_pkl = '../data/pns_2019_depressao_OTIMIZADO.pkl'
df_otimizado.to_pickle(output_pkl)
tamanho_pkl_mb = os.path.getsize(output_pkl) / 1024**2
print(f"\n✓ Pickle salvo: {output_pkl}")
print(f"  Tamanho: {tamanho_pkl_mb:.2f} MB")
print(f"  Redução vs CSV: {(1 - tamanho_pkl_mb/tamanho_csv_mb)*100:.1f}%")

print(f"\n✓ Concluído! ({datetime.now().strftime('%H:%M:%S')})")

print("\n" + "="*80)
print("📋 RESUMO FINAL")
print("="*80)
print(f"✅ Dataset otimizado criado com sucesso!")
print(f"\n📊 Características:")
print(f"   • Registros: {len(df_otimizado):,}")
print(f"   • Features: {len(colunas_essenciais)}")
print(f"   • Variável alvo: diagnostico_depressao")
print(f"   • Taxa de depressão: {df_otimizado['diagnostico_depressao'].mean()*100:.1f}%")
print(f"   • Tamanho CSV: {tamanho_csv_mb:.2f} MB")
print(f"   • Tamanho Pickle: {tamanho_pkl_mb:.2f} MB")
print(f"\n💡 Use o arquivo OTIMIZADO no notebook para melhor performance!")
print(f"\n🚀 Pronto para upload e uso no projeto!")
