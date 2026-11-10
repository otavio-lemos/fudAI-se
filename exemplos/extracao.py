"""Extração de um texto fictício com schema nativo e validação de negócio."""
from contextlib import nullcontext
from decimal import Decimal
from typing import Self
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator
from exemplos.base import modelo_configurado


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    descricao: str
    quantidade: int = Field(gt=0)
    preco_unitario: float = Field(ge=0)


class Nota(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    numero: str | None
    itens: list[Item]
    total: float | None

    @model_validator(mode="after")
    def conferir_total(self) -> Self:
        # Este exemplo não contém frete, impostos ou descontos.
        if self.total is not None:
            esperado = sum((Decimal(str(i.preco_unitario)) * i.quantidade for i in self.itens), Decimal(0))
            if abs(Decimal(str(self.total)) - esperado) > Decimal("0.01"):
                raise ValueError("Total não confere com os itens; encaminhe para revisão.")
        return self


def extrair(texto: str, client=None) -> Nota:
    modelo = modelo_configurado()
    # Fecha apenas o cliente criado aqui; o cliente injetado pertence ao chamador.
    cliente = OpenAI(timeout=30.0, max_retries=1) if client is None else nullcontext(client)
    with cliente as api:
        resposta = api.responses.parse(
            model=modelo,
            instructions=("Extraia apenas campos explícitos. Use null se número ou total não constarem. "
                          "Não invente itens. O exemplo não inclui descontos, impostos ou frete."),
            input=texto,
            text_format=Nota,
            store=False,
        )
    if resposta.status != "completed" or resposta.output_parsed is None:
        raise RuntimeError("Sem extração válida: verifique recusa, truncamento ou resposta incompleta.")
    return resposta.output_parsed


if __name__ == "__main__":
    texto = "Nota 123. Item: cabo USB; quantidade 2; preço unitário 10,50. Total 21,00."
    print(extrair(texto).model_dump_json(indent=2))
