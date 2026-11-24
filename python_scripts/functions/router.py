from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from functions.schemas import EstadoAgente, Decisao

llm = ChatOllama(model="llama3.1:8b", temperature=0)
classificador = llm.with_structured_output(Decisao)

PROMPT_ROTA = """Analise a pergunta do usuário e escolha a rota:
- 'tabular': prever cobertura florestal (forest cover), dados numéricos ou tabelas.
- 'imagem': classificar imagens, pedra-papel-tesoura, fotos ou jogadas.
- 'combinado': se pedir para avaliar imagem e tabela juntos.
- 'fora_escopo': APENAS para perguntas claramente sobre outros assuntos (ex: capital de países, receitas de bolo).

Pergunta: {pergunta}
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_ROTA)
chain_roteador = prompt_template | classificador

def no_roteador(estado: EstadoAgente) -> dict:
    print("[⚡ Jhonny-Log] Analisando a rota com LLM...")
    pergunta = estado.get("pergunta", "")
    tem_img = bool(estado.get("caminho_imagem"))
    tem_tab = bool(estado.get("dados_tabulares"))
    
    if not pergunta:
        if tem_img and tem_tab: return {"rota": "combinado"}
        if tem_img: return {"rota": "imagem"}
        if tem_tab: return {"rota": "tabular"}
        
    decisao = chain_roteador.invoke({"pergunta": pergunta})
    print(f"[⚡ Jhonny-Log] LLM decidiu a rota: {decisao.rota} | Motivo: {decisao.motivo}")
    
    if decisao.rota == "fora_escopo" and (tem_img or tem_tab):
        print("[⚡ Jhonny-Log] Override: Forçando rota pelos dados presentes.")
        if tem_img and tem_tab: return {"rota": "combinado"}
        if tem_img: return {"rota": "imagem"}
        if tem_tab: return {"rota": "tabular"}
        
    return {"rota": decisao.rota}

def no_fora_escopo(estado: EstadoAgente) -> dict:
    print("[⚡ Jhonny-Log] Rota fora de escopo acionada.")
    return {"erros": ["O pedido está fora do escopo técnico do agente."]}

PROMPT_RESPOSTA = """Você é um agente técnico explicando resultados de Machine Learning.

REGRA DE OURO 1: Se a rota escolhida for 'fora_escopo', responda EXATAMENTE: "Infelizmente, o pedido está fora do escopo técnico do agente." e não escreva mais nada.
REGRA DE OURO 2: Se a rota for 'tabular', 'imagem' ou 'combinado', NUNCA diga que está fora do escopo. Explique as previsões ou os erros das ferramentas detalhadamente ao usuário.

Pergunta do usuário: {pergunta}
Rota escolhida: {rota}

Previsão Tabular: {pred_tabular}
Previsão Imagem: {pred_imagem}
Erros detectados: {erros}

Resposta Final:"""
prompt_resposta_template = ChatPromptTemplate.from_template(PROMPT_RESPOSTA)
chain_resposta = prompt_resposta_template | llm 

def no_resposta(estado: EstadoAgente) -> dict:
    print("[⚡ Jhonny-Log] Gerando resposta final explicativa...")
    res = chain_resposta.invoke({
        "pergunta": estado.get("pergunta", ""),
        "rota": estado.get("rota", "desconhecida"),
        "pred_tabular": estado.get("pred_tabular", "N/A"),
        "pred_imagem": estado.get("pred_imagem", "N/A"),
        "erros": estado.get("erros", [])
    })
    return {"resposta": res.content}