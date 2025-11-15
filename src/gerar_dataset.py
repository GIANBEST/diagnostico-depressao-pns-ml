"""
Script para gerar dataset sintético baseado na PNS/IBGE
Simula dados de saúde para classificação de depressão
"""

import pandas as pd
import numpy as np

# Definir seed para reprodutibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def gerar_dataset_pns(n_amostras=10000, taxa_depressao=0.12):
    """
    Gera dataset sintético simulando dados da PNS/IBGE
    
    Parâmetros:
    -----------
    n_amostras : int
        Número total de amostras
    taxa_depressao : float
        Proporção de casos positivos (depressão)
    
    Retorna:
    --------
    pd.DataFrame
        Dataset com features e variável alvo
    """
    
    # Número de casos positivos e negativos
    n_positivos = int(n_amostras * taxa_depressao)
    n_negativos = n_amostras - n_positivos
    
    # Criar variável alvo
    diagnostico_depressao = np.concatenate([
        np.ones(n_positivos),
        np.zeros(n_negativos)
    ])
    
    # Embaralhar
    indices = np.random.permutation(n_amostras)
    diagnostico_depressao = diagnostico_depressao[indices]
    
    # Features numéricas
    # Idade (18-90 anos)
    idade = np.random.randint(18, 90, n_amostras)
    # Pessoas com depressão tendem a ser mais jovens ou mais velhas
    idade = np.where(diagnostico_depressao == 1, 
                     idade + np.random.normal(-5, 10, n_amostras),
                     idade)
    idade = np.clip(idade, 18, 90)
    
    # Horas de sono por dia (3-12 horas)
    horas_sono = np.random.uniform(5, 9, n_amostras)
    # Pessoas com depressão tendem a dormir menos ou mais
    horas_sono = np.where(diagnostico_depressao == 1,
                          horas_sono + np.random.normal(-1.5, 1, n_amostras),
                          horas_sono)
    horas_sono = np.clip(horas_sono, 3, 12)
    
    # Atividade física (minutos por semana)
    atividade_fisica = np.random.exponential(120, n_amostras)
    # Pessoas com depressão tendem a fazer menos atividade física
    atividade_fisica = np.where(diagnostico_depressao == 1,
                                 atividade_fisica * 0.6,
                                 atividade_fisica)
    atividade_fisica = np.clip(atividade_fisica, 0, 600)
    
    # Índice de Massa Corporal (IMC)
    imc = np.random.normal(26, 5, n_amostras)
    imc = np.where(diagnostico_depressao == 1,
                   imc + np.random.normal(1.5, 2, n_amostras),
                   imc)
    imc = np.clip(imc, 15, 50)
    
    # Número de consultas médicas no último ano
    consultas_medicas = np.random.poisson(3, n_amostras)
    consultas_medicas = np.where(diagnostico_depressao == 1,
                                  consultas_medicas + np.random.poisson(2, n_amostras),
                                  consultas_medicas)
    consultas_medicas = np.clip(consultas_medicas, 0, 20)
    
    # Nível de estresse (escala 0-10)
    nivel_estresse = np.random.uniform(2, 7, n_amostras)
    nivel_estresse = np.where(diagnostico_depressao == 1,
                              nivel_estresse + np.random.normal(2.5, 1, n_amostras),
                              nivel_estresse)
    nivel_estresse = np.clip(nivel_estresse, 0, 10)
    
    # Renda familiar (em salários mínimos)
    renda_familiar = np.random.exponential(3, n_amostras)
    renda_familiar = np.clip(renda_familiar, 0.5, 20)
    
    # Features categóricas
    # Sexo (0 = Masculino, 1 = Feminino)
    # Mulheres têm maior prevalência de depressão
    sexo = np.random.binomial(1, 0.52, n_amostras)
    prob_depressao_mulher = np.where(diagnostico_depressao == 1, 0.65, 0.52)
    sexo = np.random.binomial(1, prob_depressao_mulher, n_amostras)
    
    # Estado civil (0 = Solteiro, 1 = Casado, 2 = Divorciado, 3 = Viúvo)
    estado_civil = np.random.choice([0, 1, 2, 3], n_amostras, p=[0.3, 0.45, 0.15, 0.1])
    # Divorciados têm maior prevalência
    estado_civil = np.where((diagnostico_depressao == 1) & (np.random.random(n_amostras) < 0.3),
                            2,
                            estado_civil)
    
    # Escolaridade (0 = Fundamental, 1 = Médio, 2 = Superior)
    escolaridade = np.random.choice([0, 1, 2], n_amostras, p=[0.35, 0.45, 0.2])
    
    # Situação de emprego (0 = Desempregado, 1 = Empregado, 2 = Aposentado)
    situacao_emprego = np.random.choice([0, 1, 2], n_amostras, p=[0.15, 0.65, 0.2])
    # Desempregados têm maior prevalência
    situacao_emprego = np.where((diagnostico_depressao == 1) & (np.random.random(n_amostras) < 0.25),
                                0,
                                situacao_emprego)
    
    # Possui plano de saúde
    plano_saude = np.random.binomial(1, 0.3, n_amostras)
    
    # Fuma (0 = Não, 1 = Sim)
    fuma = np.random.binomial(1, 0.15, n_amostras)
    # Fumantes têm maior prevalência
    fuma = np.where((diagnostico_depressao == 1) & (np.random.random(n_amostras) < 0.4),
                    1,
                    fuma)
    
    # Consome álcool (0 = Não, 1 = Ocasionalmente, 2 = Frequentemente)
    consumo_alcool = np.random.choice([0, 1, 2], n_amostras, p=[0.4, 0.45, 0.15])
    
    # Doença crônica (0 = Não, 1 = Sim)
    doenca_cronica = np.random.binomial(1, 0.25, n_amostras)
    doenca_cronica = np.where((diagnostico_depressao == 1) & (np.random.random(n_amostras) < 0.4),
                              1,
                              doenca_cronica)
    
    # Suporte social (0 = Baixo, 1 = Médio, 2 = Alto)
    suporte_social = np.random.choice([0, 1, 2], n_amostras, p=[0.2, 0.5, 0.3])
    # Pessoas com depressão têm menos suporte
    suporte_social = np.where((diagnostico_depressao == 1) & (np.random.random(n_amostras) < 0.5),
                              0,
                              suporte_social)
    
    # Região (0 = Norte, 1 = Nordeste, 2 = Sudeste, 3 = Sul, 4 = Centro-Oeste)
    regiao = np.random.choice([0, 1, 2, 3, 4], n_amostras, p=[0.08, 0.28, 0.42, 0.14, 0.08])
    
    # Criar DataFrame
    df = pd.DataFrame({
        'idade': idade,
        'sexo': sexo,
        'estado_civil': estado_civil,
        'escolaridade': escolaridade,
        'situacao_emprego': situacao_emprego,
        'renda_familiar': renda_familiar,
        'regiao': regiao,
        'plano_saude': plano_saude,
        'imc': imc,
        'horas_sono': horas_sono,
        'atividade_fisica': atividade_fisica,
        'fuma': fuma,
        'consumo_alcool': consumo_alcool,
        'doenca_cronica': doenca_cronica,
        'consultas_medicas': consultas_medicas,
        'nivel_estresse': nivel_estresse,
        'suporte_social': suporte_social,
        'diagnostico_depressao': diagnostico_depressao.astype(int)
    })
    
    # Introduzir valores faltantes (5-10% em algumas colunas)
    colunas_com_nan = ['imc', 'horas_sono', 'atividade_fisica', 'renda_familiar', 'nivel_estresse']
    for col in colunas_com_nan:
        mask = np.random.random(n_amostras) < 0.07
        df.loc[mask, col] = np.nan
    
    return df

if __name__ == "__main__":
    # Gerar dataset
    print("Gerando dataset sintético da PNS/IBGE...")
    df = gerar_dataset_pns(n_amostras=10000, taxa_depressao=0.12)
    
    # Salvar
    output_path = "../data/dataset_pns_depressao.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset gerado com sucesso!")
    print(f"Salvo em: {output_path}")
    print(f"\nShape: {df.shape}")
    print(f"\nDistribuição da variável alvo:")
    print(df['diagnostico_depressao'].value_counts())
    print(f"\nTaxa de depressão: {df['diagnostico_depressao'].mean():.2%}")
    print(f"\nValores faltantes por coluna:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
