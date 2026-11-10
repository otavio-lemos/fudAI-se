"""Pausa e retomada locais; não envia mensagem nem executa ação externa."""
from typing import TypedDict
from uuid import uuid4
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class Estado(TypedDict):
    rascunho: str
    aprovado: bool
    status: str


def revisar(estado: Estado):
    decisao = interrupt({"rascunho": estado["rascunho"], "acao": "revisar"})
    if not isinstance(decisao, dict) or type(decisao.get("aprovado")) is not bool:
        raise ValueError("A decisão exige aprovado booleano.")
    texto = decisao.get("texto", estado["rascunho"])
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("O texto revisado não pode estar vazio.")
    return {"aprovado": decisao["aprovado"], "rascunho": texto}


def finalizar(estado: Estado):
    return {"status": "aprovado" if estado["aprovado"] else "rejeitado"}


def criar_app():
    grafo = StateGraph(Estado)
    grafo.add_node("revisar", revisar)
    grafo.add_node("finalizar", finalizar)
    grafo.add_edge(START, "revisar")
    grafo.add_edge("revisar", "finalizar")
    grafo.add_edge("finalizar", END)
    return grafo.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    app = criar_app()
    config = {"configurable": {"thread_id": str(uuid4())}}
    pausado = app.invoke({"rascunho": "Resposta a revisar.", "aprovado": False, "status": "pendente"}, config)
    print("Pausa:", pausado["__interrupt__"][0].value)
    # Simulação da decisão humana. Em um produto, vem de uma interface autenticada.
    final = app.invoke(Command(resume={"aprovado": True, "texto": "Resposta editada pelo revisor."}), config)
    print("Resultado:", final["status"], "—", final["rascunho"])
