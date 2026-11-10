import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from langgraph.types import Command
from pydantic import ValidationError
from exemplos.base import SEM_CONTEXTO, recuperar
from exemplos.extracao import Nota, extrair
from exemplos.rag import responder
from exemplos.revisao_humana import criar_app
from exemplos import streaming


class RecuperacaoTests(unittest.TestCase):
    def test_acentos_e_fonte(self):
        self.assertEqual(recuperar("devolução")[0].id, "devolucao")
        self.assertEqual(recuperar("senha")[0].id, "senha")

    def test_sem_evidencia_nao_chama_api(self):
        client = Mock()
        self.assertEqual(responder("astronomia", client), SEM_CONTEXTO)
        client.responses.create.assert_not_called()

    @patch.dict(os.environ, {"OPENAI_MODEL": "modelo-de-teste"})
    def test_contexto_e_citacao_no_pedido(self):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(status="completed", output_text="Veja Segurança [senha].")
        self.assertIn("[senha]", responder("senha", client))
        pedido = client.responses.create.call_args.kwargs
        self.assertIn("[senha]", pedido["input"])
        self.assertNotIn("[devolucao]", pedido["input"])
        self.assertFalse(pedido["store"])

    @patch.dict(os.environ, {"OPENAI_MODEL": "modelo-de-teste"})
    def test_geracao_incompleta(self):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(status="incomplete", output_text="parcial")
        with self.assertRaises(RuntimeError):
            responder("senha", client)


class RevisaoTests(unittest.TestCase):
    def test_aprovar_editar_rejeitar_e_isolar_threads(self):
        app = criar_app()
        for tid, aprovado in (("aceita", True), ("recusa", False)):
            with self.subTest(tid=tid):
                config = {"configurable": {"thread_id": tid}}
                pausa = app.invoke({"rascunho": "original", "aprovado": False, "status": "pendente"}, config)
                self.assertTrue(pausa["__interrupt__"])
                self.assertEqual(pausa["status"], "pendente")
                final = app.invoke(Command(resume={"aprovado": aprovado, "texto": tid}), config)
                self.assertEqual(final["status"], "aprovado" if aprovado else "rejeitado")
                self.assertEqual(final["rascunho"], tid)
        self.assertEqual(app.get_state({"configurable": {"thread_id": "aceita"}}).values["rascunho"], "aceita")

    def test_decisao_nao_aceita_string_como_booleano(self):
        app = criar_app()
        config = {"configurable": {"thread_id": "invalida"}}
        app.invoke({"rascunho": "original", "aprovado": False, "status": "pendente"}, config)
        with self.assertRaises(ValueError):
            app.invoke(Command(resume={"aprovado": "sim"}), config)


class ExtracaoTests(unittest.TestCase):
    def test_total_ausente_e_presente(self):
        dados = {"numero": None, "itens": [{"descricao": "cabo", "quantidade": 2, "preco_unitario": 10.5}], "total": 21.0}
        self.assertEqual(Nota.model_validate(dados).total, 21.0)
        dados["total"] = None
        self.assertIsNone(Nota.model_validate(dados).total)
        dados["total"] = 30.0
        with self.assertRaises(ValidationError):
            Nota.model_validate(dados)

    def test_quantidade_invalida(self):
        with self.assertRaises(ValidationError):
            Nota.model_validate({"numero": None, "itens": [{"descricao": "cabo", "quantidade": -1, "preco_unitario": 1.0}], "total": None})

    @patch.dict(os.environ, {"OPENAI_MODEL": "modelo-de-teste"})
    def test_recusa_sem_resultado_estruturado(self):
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=None)
        with self.assertRaises(RuntimeError):
            extrair("texto", client)


class StreamingTests(unittest.TestCase):
    def test_sem_evidencia_nao_abre_cliente(self):
        async def coletar():
            return [e async for e in streaming.gerar("astronomia")]
        with patch.object(streaming, "AsyncOpenAI") as client:
            eventos = asyncio.run(coletar())
            client.assert_not_called()
        self.assertEqual(eventos[-1], {"tipo": "fim"})
        self.assertEqual(eventos[0]["conteudo"], SEM_CONTEXTO)

    def test_websocket_validacao_e_fim(self):
        with TestClient(streaming.app) as client:
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_text(" ")
                self.assertEqual(ws.receive_json()["tipo"], "erro")
                ws.send_text("astronomia")
                self.assertEqual(ws.receive_json()["conteudo"], SEM_CONTEXTO)
                self.assertEqual(ws.receive_json()["tipo"], "fim")


if __name__ == "__main__":
    unittest.main()
