Classificador de Tickets Inteligente (Machine Learning + SQL)

Este projeto é um sistema de triagem automática de chamados técnicos que utiliza Inteligência Artificial para classificar a prioridade de problemas, aprendendo continuamente com o feedback humano.

Diferenciais Técnicos
- **Aprendizado Ativo (Human-in-the-loop):** O sistema permite que o analista corrija a classificação da IA, aprimorando o modelo em tempo real.
- **Persistência Relacional:** Utiliza **SQLite** para armazenar novos conhecimentos, garantindo que o aprendizado não seja perdido ao fechar o software.
- **Algoritmo Random Forest:** Implementação robusta de florestas aleatórias para classificação de texto (NLP).

Tecnologias Utilizadas
- **Python 3.13**
- **Scikit-Learn**: Processamento de linguagem natural e Random Forest.
- **Pandas**: Manipulação de dados para treinamento.
- **SQLite3**: Banco de dados para persistência de memória.

Como o Modelo Funciona
O modelo converte descrições de texto em vetores numéricos através do `CountVectorizer` e utiliza um conjunto de árvores de decisão para definir se a urgência é **Alta, Média ou Baixa**.

Como Testar
1.Instale as dependências necessárias:
`pip install pandas scikit-learn`

2.Execute o script principal:
`python classificador_ia.py`

3.Interaja com o sistema:
Descreva um problema técnico quando solicitado (Ex: "Sistema fora do ar").
A IA sugerirá uma prioridade (Alta, Média ou Baixa).

4.Treine a IA:
Caso a sugestão esteja incorreta, informe a prioridade correta.
O sistema salvará o novo exemplo no banco de dados e re-treinará o modelo instantaneamente.

---
**Anderson Parisotto** | Analista de Sistemas
