"""
Script para carregar dados da PNS_2019.txt
Baseado no dicionário de variáveis da PNS 2019
"""

import pandas as pd
import numpy as np

# Definir colunas principais baseado no dicionário PNS 2019
# Posições baseadas no layout da PNS (ajustar conforme dicionário oficial)

def carregar_pns_depressao(arquivo='../data/PNS_2019.txt', n_amostras=None):
    """
    Carrega dados da PNS 2019 focando em variáveis relacionadas à depressão
    
    IMPORTANTE: As posições das colunas precisam ser ajustadas conforme
    o dicionário oficial da PNS 2019 do IBGE
    """
    
    print("="*80)
    print("CARREGANDO DADOS DA PNS 2019")
    print("="*80)
    
    # Especificação de largura fixa (EXEMPLO - PRECISA SER AJUSTADO)
    # Estas são posições aproximadas - ajustar com dicionário oficial
    colspecs = [
        (0, 2),      # UF
        (2, 9),      # Controle/ID
        (10, 12),    # Idade (aproximado)
        (14, 15),    # Sexo (aproximado)
        # Adicionar mais conforme dicionário
    ]
    
    names = [
        'uf',
        'id_domicilio',
        'idade',
        'sexo',
    ]
    
    print(f"\nLendo arquivo: {arquivo}")
    
    try:
        # Ler arquivo em formato de largura fixa
        if n_amostras:
            df = pd.read_fwf(
                arquivo,
                colspecs=colspecs,
                names=names,
                encoding='latin1',
                nrows=n_amostras
            )
        else:
            df = pd.read_fwf(
                arquivo,
                colspecs=colspecs,
                names=names,
                encoding='latin1'
            )
        
        print(f"✓ Dados carregados: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
        print(f"\nPrimeiras linhas:")
        print(df.head())
        
        return df
    
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")
        print("\n⚠️  ATENÇÃO:")
        print("É necessário o DICIONÁRIO DE VARIÁVEIS da PNS 2019 para definir")
        print("as posições corretas das colunas no arquivo de largura fixa.")
        print("\nO dicionário está disponível em:")
        print("https://www.ibge.gov.br/estatisticas/sociais/saude/9160-pesquisa-nacional-de-saude.html")
        return None

if __name__ == "__main__":
    # Testar com 1000 linhas primeiro
    df = carregar_pns_depressao(n_amostras=1000)
    
    if df is not None:
        print("\n" + "="*80)
        print("ESTATÍSTICAS BÁSICAS")
        print("="*80)
        print(df.describe())
        print(f"\nTipos de dados:")
        print(df.dtypes)
