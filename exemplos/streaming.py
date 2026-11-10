"""WebSocket local: recuperar primeiro, gerar depois, emitir eventos JSON."""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import APIError, AsyncOpenAI
from exemplos.base import SEM_CONTEXTO, contexto, modelo_configurado, recuperar
from exemplos.rag import INSTRUCOES

app = FastAPI()


async def gerar(pergunta: str):
    documentos = recuperar(pergunta)
    if not documentos:
        yield {"tipo": "texto", "conteudo": SEM_CONTEXTO}
        yield {"tipo": "fim"}
        return
    modelo = modelo_configurado()
    async with AsyncOpenAI(timeout=30.0, max_retries=1) as client:
        stream = await client.responses.create(
            model=modelo,
            instructions=INSTRUCOES,
            input=f"Documentos:\n{contexto(documentos)}\n\nPergunta: {pergunta}",
            stream=True,
            store=False,
        )
        concluido = False
        async with stream:
            async for evento in stream:
                if evento.type == "response.output_text.delta":
                    yield {"tipo": "texto", "conteudo": evento.delta}
                elif evento.type == "response.completed":
                    concluido = True
                elif evento.type in {"error", "response.failed", "response.incomplete", "response.refusal.delta"}:
                    yield {"tipo": "erro", "conteudo": "Geração recusada, incompleta ou interrompida."}
                    return
        yield {"tipo": "fim"} if concluido else {"tipo": "erro", "conteudo": "Stream sem conclusão."}


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            pergunta = await websocket.receive_text()
            if not pergunta.strip() or len(pergunta) > 2000:
                await websocket.send_json({"tipo": "erro", "conteudo": "Envie de 1 a 2000 caracteres."})
                continue
            try:
                async for evento in gerar(pergunta):
                    await websocket.send_json(evento)
            except (APIError, ValueError):
                await websocket.send_json({"tipo": "erro", "conteudo": "Falha na configuração ou no provedor."})
    except WebSocketDisconnect:
        pass
