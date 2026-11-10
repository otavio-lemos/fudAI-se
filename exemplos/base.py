"""Recuperação lexical pequena e configuração explícita de modelo."""
import os
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Documento:
    id: str
    texto: str


DOCUMENTOS = (
    Documento("senha", "Para redefinir a senha, acesse Configurações > Segurança > Redefinir senha."),
    Documento("devolucao", "A política de devolução do produto permite solicitar devolução em até 30 dias."),
)
STOPWORDS = {"a", "o", "as", "os", "de", "da", "do", "e", "em", "um", "uma", "como", "qual", "que", "para", "minha", "meu"}
SEM_CONTEXTO = "Não encontrei informação suficiente na base para responder."


def termos(texto: str) -> set[str]:
    normalizado = unicodedata.normalize("NFKD", texto.casefold())
    normalizado = "".join(c for c in normalizado if not unicodedata.combining(c))
    return set(re.findall(r"[a-z0-9]+", normalizado)) - STOPWORDS


def recuperar(pergunta: str, limite: int = 2) -> list[Documento]:
    """Sobreposição de termos; não implementa embeddings, BM25 ou reranking."""
    if limite < 1:
        raise ValueError("O limite precisa ser positivo.")
    consulta = termos(pergunta)
    candidatos = [(len(consulta & termos(doc.texto)), doc) for doc in DOCUMENTOS]
    candidatos.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in candidatos if score > 0][:limite]


def contexto(documentos: list[Documento]) -> str:
    return "\n\n".join(f"[{doc.id}] {doc.texto}" for doc in documentos)


def modelo_configurado() -> str:
    modelo = os.environ.get("OPENAI_MODEL", "").strip()
    if not modelo:
        raise ValueError("Defina OPENAI_MODEL com um ID disponível na sua conta.")
    return modelo
