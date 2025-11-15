"""
Script para carregar dados reais da PNS 2019
Focado em variáveis relacionadas à depressão
"""

import pandas as pd
import numpy as np

print("="*80)
print("CARREGANDO DADOS DA PNS 2019 - DEPRESSÃO")
print("="*80)

# Definir especificação de colunas baseado no dicionário
# Formato: (posição_inicial-1, posição_final)
# Posições começam em 1 no dicionário, mas Python usa índice 0

colspecs = [
    # Identificação
    (0, 2),          # V0001 - UF
    (8, 10),         # V0024 - Número de seleção do domicílio
    
    # Características demográficas
    (19, 21),        # C006 - Idade
    (21, 22),        # C008 - Sexo
    (48, 49),        # C009 - Cor ou raça
    (22, 23),        # C009A - Estado civil
    
    # Educação
    (49, 50),        # D009 - Sabe ler e escrever
    (51, 52),        # D011 - Nível de instrução
    
    # Trabalho
    (154, 155),      # E001 - Trabalhou semana passada
    
    # Saúde geral
    (805, 806),      # Q001 - Estado de saúde
    (806, 807),      # Q002 - Diagnóstico de doença crônica
    
    # DEPRESSÃO - Variável principal (Q092)
    (986, 987),      # Q092 - Diagnóstico de depressão por médico/profissional
    (987, 988),      # Q09201 - Médico receitou medicamento
    (988, 989),      # Q09202 - Usou medicamento nas últimas 2 semanas
    (989, 991),      # Q09301 - Idade no primeiro diagnóstico
    (991, 992),      # Q094 - Vai ao médico regularmente
    (994, 995),      # Q09605 - Faz psicoterapia
    (995, 996),      # Q09606 - Toma medicamentos
    (997, 998),      # Q098 - Medicamento de serviço público
    (1006, 1007),    # Q109 - Grau de limitação por causa da depressão
    
    # Outras variáveis de saúde mental
    (949, 951),      # Q06306 - PHQ-9 (escala de depressão) - se disponível
    
    # Estilo de vida
    (1087, 1088),    # P001 - Fumo
    (1103, 1104),    # P027 - Consumo de álcool
    (1153, 1154),    # P034 - Atividade física
    
    # Peso e altura (para IMC)
    (1223, 1227),    # P004 - Peso
    (1227, 1230),    # P005 - Altura
    
    # Renda
    (1538, 1548),    # Renda domiciliar per capita
]

names = [
    'uf',
    'num_domicilio',
    'idade',
    'sexo',
    'cor_raca',
    'estado_civil',
    'sabe_ler_escrever',
    'nivel_instrucao',
    'trabalha',
    'estado_saude',
    'doenca_cronica',
    'diagnostico_depressao',  # VARIÁVEL ALVO PRINCIPAL
    'medicamento_receitado',
    'usou_medicamento_2sem',
    'idade_primeiro_diagnostico',
    'vai_medico_regular',
    'faz_psicoterapia',
    'toma_medicamentos',
    'medicamento_servico_publico',
    'grau_limitacao_depressao',
    'phq9_score',
    'fuma',
    'consumo_alcool',
    'atividade_fisica',
    'peso',
    'altura',
    'renda_domiciliar',
]

print(f"\nTotal de variáveis a extrair: {len(names)}")
print("\n🎯 Variável alvo: diagnostico_depressao (Q092)")

# Carregar dados
print("\n" + "="*80)
print("CARREGANDO ARQUIVO PNS_2019.txt...")
print("="*80)

try:
    # Carregar amostra primeiro para testar
    print("\n📊 Carregando amostra de 10.000 registros para teste...")
    
    df = pd.read_fwf(
        '../data/PNS_2019.txt',
        colspecs=colspecs,
        names=names,
        encoding='latin1',
        nrows=10000
    )
    
    print(f"✓ Dados carregados com sucesso!")
    print(f"Shape: {df.shape}")
    
    print("\n" + "="*80)
    print("PRIMEIRAS LINHAS:")
    print("="*80)
    print(df.head())
    
    print("\n" + "="*80)
    print("INFORMAÇÕES DO DATASET:")
    print("="*80)
    df.info()
    
    print("\n" + "="*80)
    print("DISTRIBUIÇÃO DA VARIÁVEL ALVO (diagnostico_depressao):")
    print("="*80)
    print(df['diagnostico_depressao'].value_counts())
    print(f"\nValores únicos: {df['diagnostico_depressao'].unique()}")
    
    # Salvar amostra
    output_file = '../data/pns_2019_depressao_amostra.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Amostra salva em: {output_file}")
    
    print("\n" + "="*80)
    print("✓ SCRIPT CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print("\n💡 Próximos passos:")
    print("   1. Verificar se as variáveis foram extraídas corretamente")
    print("   2. Ajustar posições se necessário")
    print("   3. Carregar dataset completo (293.726 registros)")
    print("   4. Adaptar o notebook para usar esses dados reais")
    
except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
