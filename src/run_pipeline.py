
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    recall_score, precision_score, f1_score, accuracy_score,
    roc_curve, roc_auc_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("✓ Bibliotecas importadas com sucesso!")
print(f"✓ Random State definido: {RANDOM_STATE}")

# Create a directory for plots if it doesn't exist
output_dir = 'c:/Users/GIANBEST/Downloads/Compressed/diagnostico-depressao-pns-ml-main/notebooks/plots'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('c:/Users/GIANBEST/Downloads/Compressed/diagnostico-depressao-pns-ml-main/data/dataset_pns_depressao.csv')

print("=" * 80)
print("INFORMAÇÕES GERAIS DO DATASET")
print("=" * 80)
print(f"\nDimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
print(f"\nPrimeiras 5 linhas:")
print(df.head())

print("\n" + "=" * 80)
print("INFORMAÇÕES DETALHADAS")
print("=" * 80)
df.info()

print("\n" + "=" * 80)
print("ESTATÍSTICAS DESCRITIVAS")
print("=" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("VALORES AUSENTES")
print("=" * 80)
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Valores Faltantes': missing,
    'Porcentagem (%)': missing_percent
})
missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values('Valores Faltantes', ascending=False)
print(missing_df)

print("\n" + "=" * 80)
print("DISTRIBUIÇÃO DA VARIÁVEL ALVO - diagnostico_depressao")
print("=" * 80)
target_dist = df['diagnostico_depressao'].value_counts()
print(f"\nContagem:")
print(target_dist)
print(f"\nProporção:")
print(df['diagnostico_depressao'].value_counts(normalize=True))
print(f"\nTaxa de Depressão: {df['diagnostico_depressao'].mean():.2%}")
print(f"Taxa de Não-Depressão: {(1 - df['diagnostico_depressao'].mean()):.2%}")
print(f"\n⚠️  Dataset ALTAMENTE DESBALANCEADO!")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
df['diagnostico_depressao'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Distribuição de Diagnóstico de Depressão', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Diagnóstico (0=Não, 1=Sim)', fontsize=12)
axes[0].set_ylabel('Quantidade', fontsize=12)
axes[0].set_xticklabels(['Sim (1)', 'Não (0)'], rotation=0)
for i, v in enumerate(df['diagnostico_depressao'].value_counts()):
    axes[0].text(i, v + 100, str(v), ha='center', fontweight='bold')
colors = ['#ff6b6b', '#51cf66']
df['diagnostico_depressao'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                                 labels=['Com Depressão', 'Sem Depressão'],
                                                 colors=colors, startangle=90)
axes[1].set_title('Proporção de Diagnóstico de Depressão', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
df.boxplot(column='idade', by='diagnostico_depressao', ax=axes[0, 0])
axes[0, 0].set_title('Distribuição de Idade por Diagnóstico de Depressão')
axes[0, 0].set_xlabel('Diagnóstico (0=Não, 1=Sim)')
axes[0, 0].set_ylabel('Idade')
plt.sca(axes[0, 0])
plt.xticks([1, 2], ['Não', 'Sim'])
pd.crosstab(df['sexo'], df['diagnostico_depressao'], normalize='columns').plot(
    kind='bar', ax=axes[0, 1], color=['#4a90e2', '#e94b3c']
)
axes[0, 1].set_title('Distribuição de Sexo por Diagnóstico de Depressão')
axes[0, 1].set_xlabel('Sexo (1=Masculino, 2=Feminino)')
axes[0, 1].set_ylabel('Proporção')
axes[0, 1].legend(['Não', 'Sim'], title='Depressão')
axes[0, 1].set_xticklabels(['Masculino', 'Feminino'], rotation=0)
df.boxplot(column='anos_estudo', by='diagnostico_depressao', ax=axes[1, 0])
axes[1, 0].set_title('Distribuição de Anos de Estudo por Diagnóstico de Depressão')
axes[1, 0].set_xlabel('Diagnóstico (0=Não, 1=Sim)')
axes[1, 0].set_ylabel('Anos de Estudo')
plt.sca(axes[1, 0])
plt.xticks([1, 2], ['Não', 'Sim'])
df.boxplot(column='renda_familiar', by='diagnostico_depressao', ax=axes[1, 1])
axes[1, 1].set_title('Distribuição de Renda Familiar por Diagnóstico de Depressão')
axes[1, 1].set_xlabel('Diagnóstico (0=Não, 1=Sim)')
axes[1, 1].set_ylabel('Renda Familiar')
plt.sca(axes[1, 1])
plt.xticks([1, 2], ['Não', 'Sim'])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'demographic_distribution.png'))
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
pd.crosstab(df['estado_saude'], df['diagnostico_depressao'], normalize='columns').plot(
    kind='bar', ax=axes[0, 0], color=['#51cf66', '#ff6b6b']
)
axes[0, 0].set_title('Estado de Saúde por Diagnóstico de Depressão')
axes[0, 0].set_xlabel('Estado de Saúde (1=Muito bom a 5=Muito ruim)')
axes[0, 0].set_ylabel('Proporção')
axes[0, 0].legend(['Não', 'Sim'], title='Depressão')
df.boxplot(column='imc', by='diagnostico_depressao', ax=axes[0, 1])
axes[0, 1].set_title('Distribuição de IMC por Diagnóstico de Depressão')
axes[0, 1].set_xlabel('Diagnóstico (0=Não, 1=Sim)')
axes[0, 1].set_ylabel('IMC')
plt.sca(axes[0, 1])
plt.xticks([1, 2], ['Não', 'Sim'])
pd.crosstab(df['num_doencas_cronicas'], df['diagnostico_depressao'], normalize='columns').plot(
    kind='bar', ax=axes[1, 0], color=['#4a90e2', '#e94b3c']
)
axes[1, 0].set_title('Número de Doenças Crônicas por Diagnóstico de Depressão')
axes[1, 0].set_xlabel('Número de Doenças Crônicas')
axes[1, 0].set_ylabel('Proporção')
axes[1, 0].legend(['Não', 'Sim'], title='Depressão')
df.boxplot(column='phq2_total', by='diagnostico_depressao', ax=axes[1, 1])
axes[1, 1].set_title('Distribuição de PHQ-2 Total por Diagnóstico de Depressão')
axes[1, 1].set_xlabel('Diagnóstico (0=Não, 1=Sim)')
axes[1, 1].set_ylabel('PHQ-2 Total')
plt.sca(axes[1, 1])
plt.xticks([1, 2], ['Não', 'Sim'])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'health_indicators.png'))
plt.close()

plt.figure(figsize=(20, 16))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Dataset PNS 2019', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

X = df.drop('diagnostico_depressao', axis=1)
y = df['diagnostico_depressao']

numeric_features = X.columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE, 
    stratify=y
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)

models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, n_estimators=100, eval_metric='logloss')
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = {
    'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'ROC-AUC': []
}

for name, model in models.items():
    print(f"📊 Avaliando: {name}...")
    accuracy_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='accuracy')
    precision_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='recall')
    f1_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='f1')
    roc_auc_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='roc_auc')
    
    results['Model'].append(name)
    results['Accuracy'].append(accuracy_scores.mean())
    results['Precision'].append(precision_scores.mean())
    results['Recall'].append(recall_scores.mean())
    results['F1-Score'].append(f1_scores.mean())
    results['ROC-AUC'].append(roc_auc_scores.mean())
    
    print(f"   ✅ Concluído!")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Recall', ascending=False)

print("\n" + "=" * 80)
print("RESULTADOS DA VALIDAÇÃO CRUZADA (Ordenado por Recall)")
print("=" * 80)
print("\n", results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

best_model.fit(X_train_balanced, y_train_balanced)

y_pred = best_model.predict(X_test_preprocessed)
y_pred_proba = best_model.predict_proba(X_test_preprocessed)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 80)
print("MÉTRICAS NO CONJUNTO DE TESTE")
print("=" * 80)
print(f"\n📊 Accuracy:   {accuracy:.4f} ({accuracy:.2%})")
print(f"📊 Precision:  {precision:.4f} ({precision:.2%})")
print(f"📊 Recall:     {recall:.4f} ({recall:.2%}) {'✅' if recall >= 0.75 else '❌'} Critério: ≥75%")
print(f"📊 F1-Score:   {f1:.4f} ({f1:.2%})")
print(f"📊 ROC-AUC:    {roc_auc:.4f} ({roc_auc:.2%})")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Sem Depressão (0)', 'Com Depressão (1)'],
            yticklabels=['Sem Depressão (0)', 'Com Depressão (1)'])
plt.title(f'Matriz de Confusão - {best_model_name}\nConjunto de Teste', fontsize=14, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12)
plt.xlabel('Valor Predito', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#e94b3c', linewidth=2, label=f'{best_model_name} (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Chance Aleatória (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos (Recall/Sensibilidade)', fontsize=12)
plt.title(f'Curva ROC - {best_model_name}\nConjunto de Teste', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

precision_curve_vals, recall_curve_vals, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(recall_curve_vals, precision_curve_vals, color='#51cf66', linewidth=2, label=f'{best_model_name}')
plt.axhline(y=0.50, color='red', linestyle='--', linewidth=1, label='Critério Precision ≥ 50%')
plt.axvline(x=0.75, color='blue', linestyle='--', linewidth=1, label='Critério Recall ≥ 75%')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Sensibilidade)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title(f'Curva Precision-Recall - {best_model_name}\nConjunto de Teste', fontsize=14, fontweight='bold')
plt.legend(loc="best", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
plt.close()

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("=" * 80)
    print("IMPORTÂNCIA DAS FEATURES (Top 20)")
    print("=" * 80)
    print("\n", feature_importance.head(20).to_string(index=False))
    
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], color='#4a90e2')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importância', fontsize=12)
    plt.title(f'Top 15 Features Mais Importantes - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

print("Script finalizado.")
