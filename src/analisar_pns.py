"""
Script para analisar o formato do arquivo PNS_2019.txt
"""

# Ler primeiras linhas
with open('../data/PNS_2019.txt', 'r', encoding='latin1') as f:
    print("="*80)
    print("ANÁLISE DO ARQUIVO PNS_2019.txt")
    print("="*80)
    
    # Ler primeiras 3 linhas
    lines = [f.readline() for _ in range(3)]
    
    print(f"\nNúmero de linhas analisadas: {len(lines)}")
    print(f"Tamanho de cada linha: {len(lines[0])} caracteres")
    
    print("\n" + "="*80)
    print("PRIMEIRA LINHA (primeiros 500 caracteres):")
    print("="*80)
    print(lines[0][:500])
    
    print("\n" + "="*80)
    print("SEGUNDA LINHA (primeiros 500 caracteres):")
    print("="*80)
    print(lines[1][:500])

# Contar total de linhas
print("\n" + "="*80)
print("CONTANDO TOTAL DE LINHAS...")
print("="*80)
with open('../data/PNS_2019.txt', 'r', encoding='latin1') as f:
    total_lines = sum(1 for _ in f)
    print(f"Total de linhas: {total_lines:,}")

print("\n✓ Análise concluída!")
