import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#CARREGAMENTO E LIMPEZA
df = pd.read_csv('BreastCancer.csv')
df = df.drop(columns=['Id']) #Remove ID
df['Bare.nuclei'] = df['Bare.nuclei'].fillna(df['Bare.nuclei'].median()) #Limpa nulos

features = df.columns[:-1]

#1. GRÁFICO DE DISTRIBUIÇÃO DE CLASSES
def gerar_distribuicao():
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Class', palette='viridis')
    plt.title('Distribuição de Diagnósticos')
    plt.savefig('01_distribuicao_classes.png')
    plt.show()

#2. GRÁFICOS DE VIOLINO (Gera um por um para o relatório)
def gerar_violinos():
    features = df.columns[:-1]
    for col in features:
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x='Class', y=col, inner='quart', palette='Set2')
        plt.title(f'Distribuição Detalhada: {col}')
        plt.savefig(f'02_violin_{col}.png')
        plt.close() #Fecha para não sobrecarregar a memória
    print("Gráficos de violino gerados com sucesso!")

#3. MATRIZ DE CORRELAÇÃO (HEATMAP)
def gerar_heatmap():
    plt.figure(figsize=(12, 10))
    df_num = df.copy()
    df_num['Class'] = df_num['Class'].map({'benign': 0, 'malignant': 1})
    sns.heatmap(df_num.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mapa de Correlações')
    plt.savefig('03_heatmap_correlacao.png')
    plt.show()

#4. PCA (ANÁLISE MULTIVARIADA)
def gerar_pca():
    X = StandardScaler().fit_transform(df.drop(columns=['Class']))
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame(coords, columns=['Componente_1', 'Componente_2'])
    pca_df['Diagnóstico'] = df['Class'].values
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='Componente_1', y='Componente_2', hue='Diagnóstico', s=100)
    plt.title('Separação Espacial dos Grupos (PCA)')
    plt.savefig('04_pca_analise.png')
    plt.show()

def gerar_boxplots_combinados():
    """Gera uma matriz 3x3 com todos os boxplots em uma única imagem"""
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.boxplot(data=df, x='Class', y=col, ax=axes[i], palette='Set3')
        axes[i].set_title(f'Distribuição de {col}', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Escala (1-10)')

    plt.tight_layout()
    plt.savefig('06_boxplots_combinados.png')
    plt.show()
    print("Gráfico 06 (Boxplots Combinados) gerado.")

gerar_distribuicao()
gerar_violinos()
gerar_heatmap()
gerar_pca()
gerar_boxplots_combinados()