import pandas as pd
import sqlite3
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

#configuração banco de dados
DB_NAME = "memoria_ia.db"

def carregar_ou_criar_dados():
    #se o banco já existir, lê os dados de lá
    if os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM treinamento", conn)
        conn.close()
        return df
    else:
        #se não existe, cria com seus dados iniciais
        dados_iniciais = {
            'descricao': [
                'Sistema paralisado, erro de conexão com banco SQL', 'A tela ficou azul', 
                'Erro crítico no servidor', 'O sistema Teknisa caiu',
                'Meu teclado está falhando', 'O site está lento', 'Impressora atolando papel',
                'Mudar cor da fonte no Word', 'Como organizar pastas', 'Solicitar mouse novo'
            ],
            'prioridade': [
                'Alta', 'Alta', 'Alta', 'Alta',
                'Media', 'Media', 'Media',
                'Baixa', 'Baixa', 'Baixa'
            ]
        }
        df = pd.DataFrame(dados_iniciais)
        salvar_no_banco(df)
        return df

def salvar_no_banco(df_atualizado):
    conn = sqlite3.connect(DB_NAME)
    df_atualizado.to_sql("treinamento", conn, if_exists="replace", index=False)
    conn.close()

#engine da IA
def treinar_modelo(dados_df):
    modelo = Pipeline([
        ('vetorizador', CountVectorizer()),
        ('classificador', RandomForestClassifier(n_estimators=100))
    ])
    modelo.fit(dados_df['descricao'], dados_df['prioridade'])
    return modelo

#início do Sistema
df = carregar_ou_criar_dados()
modelo = treinar_modelo(df)

print("="*40)
print("🤖 IA DE TRIAGEM COM MEMÓRIA SQLITE")
print(f"Status: Aprendendo com {len(df)} exemplos.")
print("="*40)

while True:
    texto = input("\nDescreva o problema (ou 'sair'): ")
    if texto.lower() == 'sair': break
    
    predicao = modelo.predict([texto])[0]
    print(f"🤖 Sugestão da IA: {predicao}")
    
    feedback = input("Está correto? (s/n): ").lower()
    
    if feedback == 'n':
        while True:
            correta = input("Qual a prioridade correta? (Alta/Media/Baixa): ").capitalize()
            if correta in ['Alta', 'Media', 'Baixa']:
                break
            print("❌ Por favor, digite apenas Alta, Media ou Baixa.")
        
        #atualiza em memória
        novo_exemplo = pd.DataFrame({'descricao': [texto], 'prioridade': [correta]})
        df = pd.concat([df, novo_exemplo], ignore_index=True)
        
        #salva no Banco de Dados (Persistência)
        salvar_no_banco(df)
        
        #re-treina a IA
        modelo = treinar_modelo(df)
        print(f"🧠 Conhecimento salvo! Agora tenho {len(df)} exemplos.")

print("\nEncerrando e salvando progresso...")
