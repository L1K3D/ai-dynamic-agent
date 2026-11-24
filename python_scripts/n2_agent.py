from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Importando nossos módulos fresquinhos e organizados da pasta functions
from functions.schemas import EstadoAgente
from functions.tools import no_tabular, no_imagem, no_combinado
from functions.router import no_roteador, no_fora_escopo, no_resposta

print("[⚡ Jhonny-Log] Montando a orquestração do LangGraph...")

# 1. Inicia o Grafo passando o Dicionário de Estado
g = StateGraph(EstadoAgente)

# 2. Adicionando todos os nós que importamos
g.add_node("roteador", no_roteador)
g.add_node("tabular", no_tabular)
g.add_node("imagem", no_imagem)
g.add_node("combinado", no_combinado)
g.add_node("fora_escopo", no_fora_escopo)
g.add_node("resposta", no_resposta)

# 3. Definindo as arestas (o caminho que os dados vão seguir)
g.add_edge(START, "roteador")

# O Roteador usa a inteligência artificial para decidir a aresta condicional
g.add_conditional_edges(
    "roteador", 
    lambda e: e["rota"], 
    {
        "tabular": "tabular", 
        "imagem": "imagem",
        "combinado": "combinado", 
        "fora_escopo": "fora_escopo"
    }
)

# Independentemente de qual ferramenta rodou, todas desaguam no nó de Resposta
for n in ["tabular", "imagem", "combinado", "fora_escopo"]:
    g.add_edge(n, "resposta")

g.add_edge("resposta", END)

# 4. Compilando com Checkpointer (Memória da Sessão)
memoria = InMemorySaver()
agente_n2 = g.compile(checkpointer=memoria)

if __name__ == "__main__":
    # Script rápido apenas para testar a compilação e exportar a imagem do diagrama pro seu README
    try:
        diagrama = agente_n2.get_graph().draw_mermaid()
        with open("C:/projects/ai-dynamic-agent/python_scripts/diagrama_agente.md", "w") as f:
            f.write(f"```mermaid\n{diagrama}\n```")
        print("[⚡ Jhonny-Log] Sucesso! Diagrama exportado em diagrama_agente.md.")
    except Exception as e:
        print(f"[⚡ Jhonny-Log] Ops, erro ao gerar diagrama: {e}")