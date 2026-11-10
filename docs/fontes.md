# Metodologia e fontes

[← Voltar ao canivete](../README.md) · Revisão: 10/11/2026

## Critérios de curadoria

- Definir o papel de cada tecnologia, caso de uso, limites e link oficial.
- Preferir documentação do mantenedor, model cards, artigos originais e código do projeto.
- Distinguir produto, API, checkpoint, variante e derivado quantizado.
- Tratar versões e disponibilidade como retrato datado; confirmar na conta e na plataforma de destino.
- Não usar “melhor”, “mais usado”, “gratuito” ou ganhos percentuais como fatos sem escopo e evidência.
- Não inferir suporte a vídeo, ferramentas ou fine-tuning apenas pelo nome da família.
- Não classificar licença pela existência de um botão de download; conferir cada artefato.

## Fontes para atualização

| Assunto | Fontes primárias |
| --- | --- |
| OpenAI | [Modelos](https://developers.openai.com/api/docs/models), [referência Python](https://developers.openai.com/api/reference/python), [deprecações](https://developers.openai.com/api/docs/deprecations) |
| Anthropic | [Modelos](https://platform.claude.com/docs/en/models/overview), [ciclo de vida](https://platform.claude.com/docs/en/about-claude/model-deprecations), [Claude Code](https://code.claude.com/docs/en/overview) |
| Google | [Gemini](https://ai.google.dev/gemini-api/docs/models), [preços](https://ai.google.dev/gemini-api/docs/pricing), [Gemma](https://ai.google.dev/gemma/docs), [ADK](https://google.github.io/adk-docs/) |
| Outros provedores | [xAI](https://docs.x.ai/developers/models), [Mistral](https://docs.mistral.ai/), [Cohere](https://docs.cohere.com/docs/models), [Voyage](https://docs.voyageai.com/docs/embeddings) |
| Modelos com pesos disponíveis | Organizações oficiais no Hub e model cards linkadas no [catálogo](modelos-servicos.md) |
| Aplicações e agentes | [LangChain/LangGraph](https://docs.langchain.com/), [LlamaIndex](https://docs.llamaindex.ai/), [PydanticAI](https://ai.pydantic.dev/), [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) |
| Inferência | [vLLM](https://docs.vllm.ai/), [SGLang](https://docs.sglang.io/), [llama.cpp](https://github.com/ggml-org/llama.cpp), [MLX LM](https://github.com/ml-explore/mlx-lm), [llm-d](https://llm-d.ai/) |
| Treinamento | [Transformers](https://huggingface.co/docs/transformers/index), [PEFT](https://huggingface.co/docs/peft/index), [TRL](https://huggingface.co/docs/trl/index) |
| Protocolos | [MCP](https://modelcontextprotocol.io/), [A2A](https://a2a-protocol.org/latest/), [OpenTelemetry](https://opentelemetry.io/docs/) |
| Segurança | [OWASP GenAI](https://genai.owasp.org/), [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework), [ANPD](https://www.gov.br/anpd/pt-br) |

Links de pesquisa e descoberta, como GitHub Trending, artigos no arXiv e leaderboards, ajudam a encontrar candidatos; não substituem a documentação do projeto. Avaliações externas podem servir de referência adicional, mas não são reproduzidas aqui como ranking universal.

## Registro desta revisão

**Data da consulta:** 10 de novembro de 2026.

Foram revisados conceitos, classificações, exemplos, famílias de modelos e referências de manutenção. O guia agora inclui tecnologias ausentes no documento antigo: protocolos, engenharia de contexto, agentes de código, novos checkpoints, inferência distribuída, multimodalidade e parâmetros por ecossistema.

Pontos de manutenção confirmados nas fontes oficiais:

- [AutoGen arquivado](https://github.com/microsoft/autogen), com orientação para [Microsoft Agent Framework](https://github.com/microsoft/agent-framework).
- [TGI em manutenção](https://huggingface.co/docs/inference-endpoints/engines/tgi), com alternativas vLLM/SGLang nos Inference Endpoints.
- [Proxy OpenAI-compatible legado do Helicone](https://docs.helicone.ai/getting-started/integration-method/openai-proxy), com orientação para [AI Gateway](https://docs.helicone.ai/gateway/overview).

**Limitações:** o catálogo não cobre todos os lançamentos. As páginas oficiais podem mudar e variar por idioma; divergências encontradas no catálogo Gemini estão registradas na seção correspondente. Não foram medidos rankings, VRAM mínima, latência, ganhos de treino ou preços comparativos. Recursos dependentes de conta não foram confirmados com credenciais reais.

## Rotina sugerida de manutenção

| Camada | Quando revisar | O que conferir |
| --- | --- | --- |
| Conceitos | Semestralmente ou ao corrigir erro | Definições e distinções fundamentais |
| Catálogo | Mensalmente e antes de adoção/migração | IDs, checkpoints, licenças, modalidade e ciclo de vida |
| Código | A cada mudança de dependência/API | Instalação, imports, contratos, falhas e exemplos |
| Segurança | A cada nova ferramenta/fonte de dados | Permissões, isolamento, retenção e efeitos externos |

Essa rotina é uma proposta de manutenção, não uma automação configurada nem histórico de atualizações trimestrais já realizadas.

## Como enviar uma contribuição

Inclua arquivo/seção, alteração proposta, motivo, fonte oficial e data de consulta. Para modelos, identifique o checkpoint/ID e evite comparações sem condições equivalentes. Para código, informe Python/dependências e comandos de validação; se houver chamada paga ou treino em GPU, deixe isso explícito.
