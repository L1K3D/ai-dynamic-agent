from n2_agent import agente_n2
import os

# Dados para mockar chamadas corretas
dados_validos_forest = {
    "Elevation": 2596, 
    "Aspect": 51, 
    "Slope": 3,
    "Horizontal_Distance_To_Hydrology": 258,
    "Vertical_Distance_To_Hydrology": 0,
    "Horizontal_Distance_To_Roadways": 510,
    "Hillshade_9am": 221, 
    "Hillshade_Noon": 232, 
    "Hillshade_3pm": 148,
    "Horizontal_Distance_To_Fire_Points": 6279
}

for i in range(1, 5):
    dados_validos_forest[f"Wilderness_Area{i}"] = 0
for i in range(1, 41):
    dados_validos_forest[f"Soil_Type{i}"] = 0

# Dados para simular erro de validação (Pydantic deve barrar e pedir correção)
dados_invalidos_forest = {
    "Elevation": -50, # Elevation não pode ser menor que 0
    "Aspect": 500,    # Aspect tem que ser <= 360
    "Slope": 10       
}

# Pegando os caminhos exatamente de onde você extraiu
CAMINHO_IMG_PEDRA = "C:/projects/cnn-rock-paper-scissors/dataset/rps-test-set/rock/testrock01-00.png"
CAMINHO_IMG_PAPEL = "C:/projects/cnn-rock-paper-scissors/dataset/rps-test-set/paper/testpaper01-00.png"
# Usando o arquivo ZIP para simular um formato de imagem inválido e testar os guardrails do agente
CAMINHO_IMG_INVALIDA = "C:/projects/cnn-rock-paper-scissors/dataset/rps-test-set.zip"

# Configurando o "Thread_ID" para o InMemorySaver manter a conversa
config = {"configurable": {"thread_id": "sessao_testes_1"}}

print("\n🚀 [⚡ Jhonny-Log] INICIANDO BATERIA DE 10 TESTES DO PROJETO N2...\n")

casos_teste = [
    # ---- 4 TABULARES ----
    {"nome": "T01 - Tabular Válido", "entrada": {"pergunta": "Qual a cobertura florestal para esses dados?", "dados_tabulares": dados_validos_forest}},
    {"nome": "T02 - Tabular Sem Pergunta (LLM infere por ter dados)", "entrada": {"dados_tabulares": dados_validos_forest}},
    {"nome": "T03 - Tabular Inválido (Elevation negativa)", "entrada": {"pergunta": "Preveja com base nestes dados.", "dados_tabulares": dados_invalidos_forest}},
    {"nome": "T04 - Tabular Fora de Distribuição (Apesar de numérico, sem todas as colunas)", "entrada": {"pergunta": "Qual a classe?", "dados_tabulares": {"Elevation": 1000}}},
    
    # ---- 3 IMAGEM ----
    {"nome": "T05 - Imagem Válida (Pedra)", "entrada": {"pergunta": "Que gesto é esse?", "caminho_imagem": CAMINHO_IMG_PEDRA}},
    {"nome": "T06 - Imagem Válida (Papel)", "entrada": {"pergunta": "Adivinha minha jogada.", "caminho_imagem": CAMINHO_IMG_PAPEL}},
    {"nome": "T07 - Imagem Inválida / Corrompida (Arquivo ZIP)", "entrada": {"pergunta": "Olha essa foto.", "caminho_imagem": CAMINHO_IMG_INVALIDA}},
    
    # ---- 1 COMBINADO ----
    {"nome": "T08 - Combinado", "entrada": {"pergunta": "Avalie a foto e a tabela.", "dados_tabulares": dados_validos_forest, "caminho_imagem": CAMINHO_IMG_PEDRA}},
    
    # ---- 2 FORA DE ESCOPO ----
    {"nome": "T09 - Pergunta Aleatória 1", "entrada": {"pergunta": "Qual é a capital do Brasil?"}},
    {"nome": "T10 - Pergunta Aleatória 2", "entrada": {"pergunta": "Me dê uma receita de bolo."}}
]

resultados_log = []

for idx, teste in enumerate(casos_teste):
    print(f"\n[{idx+1}/10] 🧪 Executando: {teste['nome']}")
    try:
        # Gera um thread_id único para cada teste para isolar a memória do LangGraph
        config = {"configurable": {"thread_id": f"sessao_testes_{idx}"}}
        
        # Invoca o Agente LangGraph
        resposta_agente = agente_n2.invoke(teste["entrada"], config)
        
        # Pega a Rota que foi decidida e a Resposta Final gerada
        rota = resposta_agente.get("rota", "Não definida")
        texto = resposta_agente.get("resposta", "Sem resposta final")
        
        print(f"✅ ROTA SELECIONADA: {rota}")
        print(f"📝 RESPOSTA DO LLM: \n{texto}")
        
        resultados_log.append(f"Teste: {teste['nome']} | Rota Obtida: {rota} | Resposta: {texto.replace(chr(10), ' ')}")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO NO TESTE: {e}")
        resultados_log.append(f"Teste: {teste['nome']} | ERRO: {str(e)}")

# Exportando o log final para documentação
with open("C:/projects/ai-dynamic-agent/python_scripts/resultados_testes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(resultados_log))
    
print("\n🏁 [⚡ Jhonny-Log] BATERIA DE TESTES FINALIZADA! Resultados salvos em 'resultados_testes.txt'.")