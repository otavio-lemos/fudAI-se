"""Verifica serialização e streaming do SDK real com transporte local simulado."""
import asyncio
import json
import os
import unittest
from unittest.mock import patch
import httpx2
from openai import AsyncOpenAI, OpenAI
from exemplos import streaming
from exemplos.extracao import extrair
from exemplos.rag import responder


def resposta(texto):
    return {
        "id": "resp_teste", "object": "response", "created_at": 0,
        "model": "modelo-de-teste", "status": "completed",
        "output": [{"id": "msg_teste", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": texto, "annotations": []}]}],
    }


@patch.dict(os.environ, {"OPENAI_MODEL": "modelo-de-teste"})
class SDKTests(unittest.TestCase):
    def test_rag_com_sdk_e_transporte_simulado(self):
        pedidos = []
        def handler(request):
            pedidos.append(json.loads(request.content))
            return httpx2.Response(200, json=resposta("Redefina em Segurança [senha]."))
        with OpenAI(api_key="test-only", http_client=httpx2.Client(transport=httpx2.MockTransport(handler), trust_env=False)) as client:
            self.assertIn("[senha]", responder("senha", client))
        self.assertEqual(pedidos[0]["model"], "modelo-de-teste")
        self.assertFalse(pedidos[0]["store"])

    def test_extracao_com_schema_serializado(self):
        pedidos = []
        def handler(request):
            pedidos.append(json.loads(request.content))
            nota = {"numero": "123", "itens": [{"descricao": "cabo", "quantidade": 2, "preco_unitario": 10.5}], "total": 21.0}
            return httpx2.Response(200, json=resposta(json.dumps(nota)))
        with OpenAI(api_key="test-only", http_client=httpx2.Client(transport=httpx2.MockTransport(handler), trust_env=False)) as client:
            self.assertEqual(extrair("Nota fictícia", client).total, 21.0)
        self.assertEqual(pedidos[0]["text"]["format"]["type"], "json_schema")

    def test_stream_completo_e_incompleto(self):
        for tipo_final, esperado in (("response.completed", "fim"), ("response.incomplete", "erro")):
            with self.subTest(tipo_final=tipo_final):
                pedidos = []
                def handler(request):
                    pedidos.append(json.loads(request.content))
                    eventos = [
                        {"type": "response.output_text.delta", "delta": "Texto [senha]", "item_id": "msg_teste", "output_index": 0, "content_index": 0, "sequence_number": 0},
                        {"type": tipo_final, "response": resposta("Texto [senha]"), "sequence_number": 1},
                    ]
                    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in eventos)
                    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})
                async def coletar():
                    async with AsyncOpenAI(api_key="test-only", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler), trust_env=False)) as client:
                        with patch.object(streaming, "AsyncOpenAI", return_value=client):
                            return [e async for e in streaming.gerar("senha")]
                eventos = asyncio.run(coletar())
                self.assertEqual(eventos[0]["tipo"], "texto")
                self.assertEqual(eventos[-1]["tipo"], esperado)
                self.assertIn("[senha]", pedidos[0]["input"])
                self.assertTrue(pedidos[0]["stream"])
