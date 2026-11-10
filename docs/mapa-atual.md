# O que faltava no mapa

[← Voltar ao canivete](../README.md) · Levantamento: 10/11/2026

Esta página amplia o guia antigo. “Novo no mapa” não significa necessariamente “inventado em 2026”: inclui tecnologias já existentes que não estavam documentadas e desenvolvimentos recentes. A disponibilidade de cada implementação depende da versão.

## 1. Novas famílias e modalidades

| Frente | O que entra nesta revisão | Onde aprofundar |
| --- | --- | --- |
| Modelos hospedados | GPT-6/6.1, Claude Fable/5.5, Gemini 3/4 e Grok atual, com IDs e ressalvas de disponibilidade | [Catálogo de APIs](modelos-servicos.md#1-modelos-via-api) |
| Pesos disponíveis | Qwen3.8, DeepSeek V4.1, Kimi K3.1, MiniMax M3.1, GLM-5.3, Mistral Large 4, Granite 4.1, Gemma 4 e gpt-oss | [Checkpoints](modelos-servicos.md#2-pesos-disponíveis-famílias-e-checkpoints-recentes) |
| Recuperação multimodal | Embeddings de texto, imagem e páginas; reranking e dimensões flexíveis | [Embeddings/rerankers](modelos-servicos.md#3-embeddings-e-rerankers) |
| Imagem/vídeo/voz | GPT Image, FLUX, Qwen Image, Wan, LTX, serviços de vídeo, Qwen3-TTS, Kokoro e realtime | [Modalidades](modelos-servicos.md#5-imagem-e-vídeo) |

## 2. Arquiteturas e inferência

| Tecnologia | Ideia | Quando interessa |
| --- | --- | --- |
| Atenção linear e híbrida | Combinar mecanismos com diferentes custos de contexto | Modelos que alternam atenção e blocos recorrentes/lineares; exige runtime compatível |
| Atenção esparsa | Atender a subconjuntos selecionados de posições | Contextos longos; estratégias de seleção diferem entre arquiteturas |
| Compressão de KV | Reduzir o estado armazenado/transferido | Longo contexto e alta concorrência; avaliar qualidade e kernels |
| MoE granular | Especialistas menores e roteamento seletivo | Escala de modelos; total de pesos e computação ativa divergem |
| MTP/speculative decoding | Propor/verificar mais de um token por etapa | Aceleração dependente de modelo, aceitação e carga |
| Formatos de baixa precisão | FP8, FP4 e formatos com escalas por bloco | Reduzir memória e explorar hardware compatível |
| Prefill/decode separados | Alocar etapas de inferência em pools diferentes | Serving distribuído; exige rede e transferência de estado eficientes |
| Roteamento consciente de cache | Reutilizar prefixos entre réplicas | Workloads com instruções/históricos repetidos |

Fontes: model cards [Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B), [DeepSeek V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [Kimi K3.1](https://huggingface.co/moonshotai/Kimi-K3.1-Pro), [MiniMax M3.1](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash), [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3-Flash), [Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B); serving em [vLLM](https://docs.vllm.ai/), [SGLang](https://docs.sglang.io/) e [llm-d](https://llm-d.ai/).

## 3. Aplicações e agentes

| Frente | O que muda na aplicação | Implementações/referências |
| --- | --- | --- |
| Engenharia de contexto | Selecionar documentos, histórico, ferramentas e memória conforme a tarefa | [LangChain/Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) |
| Compaction | Reduzir histórico mantendo informação necessária para continuar | [OpenAI](https://developers.openai.com/api/docs/guides/compaction), [Claude](https://platform.claude.com/docs/en/build-with-claude/compaction) |
| Structured outputs e ferramentas estritas | Contratos de dados mais fortes entre modelo e aplicação | [OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs), [PydanticAI](https://ai.pydantic.dev/) |
| MCP e A2A | Integrações padronizadas com ferramentas e entre agentes | [MCP](https://modelcontextprotocol.io/docs/learn/architecture), [A2A](https://a2a-protocol.org/latest/) |
| Skills e instruções de projeto | Recursos reutilizáveis carregados conforme tarefa | [Claude Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview), [Claude Code](https://code.claude.com/docs/en/overview) |
| Computer/browser use | Agir sobre interfaces, além de chamar APIs | [Browser Use](https://github.com/browser-use/browser-use), [Playwright](https://playwright.dev/python/) |
| Agentes de código | Trabalhar com arquivos, comandos e testes | Codex, Claude Code, Gemini CLI e OpenCode, no [catálogo](ferramentas.md#7-programação-assistida-por-ia) |
| Execução durável e revisão | Persistir estado, pausar, aprovar e retomar | [LangGraph](https://docs.langchain.com/oss/python/langgraph/interrupts), [Temporal](https://docs.temporal.io/) |
| Sandboxes e permissões | Executar código/ferramentas com limites reais | [E2B](https://e2b.dev/docs), [segurança](seguranca.md) |

## 4. Dados, avaliação e treinamento

| Frente | O que incluir no projeto |
| --- | --- |
| Document AI | Conversão com layout e tabelas usando Docling/OCR; preservação de origem e evidências |
| RAG multimodal | Recuperação de páginas/imagens, não apenas chunks de texto; avaliação separada por modalidade |
| RAG com ACL | Filtrar documentos segundo a identidade do usuário antes de entregá-los ao modelo |
| Avaliação de agentes | Sucesso da tarefa, efeitos externos, recuperação de erros, custo e quantidade de passos |
| LLM-as-judge | Avaliação assistida com rubrica, calibração humana e controle de viés; não tratar juiz como verdade |
| Dados sintéticos | Proveniência, deduplicação, licença e avaliação independente do gerador |
| Pós-treinamento | SFT, LoRA/QLoRA, DPO, GRPO e destilação conforme dados e objetivo |
| Observabilidade | Trace da recuperação até a ferramenta e resposta; custos por tarefa e dados sensíveis minimizados |

Referências: [Docling](https://docling-project.github.io/docling/), [Ragas](https://docs.ragas.io/), [Phoenix](https://arize.com/docs/phoenix), [TRL](https://huggingface.co/docs/trl/index), [OpenTelemetry](https://opentelemetry.io/docs/).

## 5. Como manter o mapa útil

Uma tecnologia entra com definição, caso de uso, limitação e fonte. Um modelo entra com checkpoint/ID, modalidade, licença quando disponível e data da consulta. Parâmetros entram com escopo explícito de API/engine/modelo. Benchmarks entram apenas com método e condições suficientes para interpretar a comparação.

Veja [parâmetros](parametros.md) e [metodologia](fontes.md). O mapa não promete cobrir todo lançamento: prioriza tecnologias verificáveis que ajudam a construir, avaliar ou operar aplicações.
