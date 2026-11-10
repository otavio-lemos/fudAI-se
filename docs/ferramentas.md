# Ferramentas e frameworks

[← Voltar ao canivete](../README.md) · Revisão: 10/11/2026

Cada nome leva ao projeto ou à documentação oficial. A seleção mapeia funções, não estabelece um vencedor. Confira manutenção, licença e recursos da edição escolhida.

## 1. Aplicações e agentes

| Ferramenta | Papel | Quando usar |
| --- | --- | --- |
| [LangChain](https://docs.langchain.com/oss/python/langchain/overview) | Componentes e abstrações para aplicações com modelos | Integrações, ferramentas e aplicações com LLM |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | Grafos com estado, checkpoints e interrupções | Fluxos com ramificações, retomada e revisão humana |
| [LlamaIndex](https://docs.llamaindex.ai/) | Ingestão, indexação e recuperação | Aplicações centradas em dados e documentos |
| [PydanticAI](https://ai.pydantic.dev/) | Agentes com tipos e validação | Integração Python com contratos de entrada/saída |
| [CrewAI](https://docs.crewai.com/) | Agentes, equipes e flows | Divisão de trabalho com responsabilidades explícitas |
| [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) | Agentes e workflows em Python/.NET | Projetos no ecossistema Microsoft e migração de AutoGen |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel) | Integração de IA com aplicações | Sistemas existentes; confira os guias de migração do ecossistema Microsoft |
| [Google ADK](https://google.github.io/adk-docs/) | Kit de desenvolvimento de agentes | Agentes, ferramentas e avaliação no ecossistema Google |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) | Runtime de agentes, handoffs e tracing | Orquestração programática com ferramentas |
| [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview) | SDK para agentes com ferramentas e contexto | Fluxos programáticos baseados no ambiente do Claude Code |
| [smolagents](https://huggingface.co/docs/smolagents/index) | Agentes leves, incluindo execução de código | Experimentos com execução isolada e limites de ferramentas |
| [DSPy](https://dspy.ai/) | Programação e otimização de pipelines de linguagem | Otimizar com dataset, métrica e orçamento definidos |
| [Haystack](https://docs.haystack.deepset.ai/docs/intro) | Pipelines de busca e aplicações de IA | Recuperação, componentes e avaliação |
| [Vercel AI SDK](https://ai-sdk.dev/docs/introduction) | Integração de modelos em JavaScript/TypeScript | Interfaces, streaming e ferramentas em apps web |
| [Instructor](https://python.useinstructor.com/) / [Outlines](https://dottxt-ai.github.io/outlines/) | Extração validada / geração controlada | Schemas e restrições de saída, conforme o backend |
| [FastAPI](https://fastapi.tiangolo.com/) | Framework de API, não framework de agentes | Expor serviços HTTP/WebSocket |

**Mudança de manutenção:** o [AutoGen](https://github.com/microsoft/autogen) foi arquivado; o projeto orienta novos usuários ao Microsoft Agent Framework. Não o apresente como escolha atual para novos sistemas sem considerar essa condição.

## 2. Protocolos, contexto e execução de ferramentas

| Tecnologia | Papel | Distinção importante |
| --- | --- | --- |
| [MCP](https://modelcontextprotocol.io/docs/learn/architecture) | Conectar hosts a servidores com ferramentas, recursos e prompts | Transporte padronizado não torna ferramenta confiável |
| [A2A](https://a2a-protocol.org/latest/) | Comunicação entre agentes | Complementa integrações de ferramentas; não é outro nome para MCP |
| [Skills do Claude](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) | Instruções e recursos reutilizáveis carregados conforme a tarefa | Formato e execução dependem do host; leia permissões dos scripts |
| [Playwright](https://playwright.dev/python/) | Automação de navegador com código | Preferível para passos conhecidos e verificáveis |
| [Browser Use](https://github.com/browser-use/browser-use) | Navegação orientada por agentes | Útil para interfaces variáveis; exige limites e validação de resultado |
| [E2B](https://e2b.dev/docs) | Sandboxes para execução de código | Isolamento é parte da aplicação, não uma propriedade do prompt |
| [OpenTelemetry](https://opentelemetry.io/docs/) | Instrumentação de traces, métricas e logs | Ajuda a correlacionar chamadas de modelo, recuperação e ferramentas |

## 3. Inferência local e serving

| Engine/ferramenta | Papel | Quando usar |
| --- | --- | --- |
| [Ollama](https://docs.ollama.com/) | Execução local e API | Primeiro laboratório com modelos suportados |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Inferência C/C++, GGUF e múltiplos backends | CPU, GPU e quantizações locais |
| [MLX LM](https://github.com/ml-explore/mlx-lm) | Inferência e ajuste no Apple Silicon | Macs compatíveis; memória unificada é compartilhada com o sistema |
| [LM Studio](https://lmstudio.ai/docs/app/basics) | Interface e servidor local | Experimentar modelos sem montar uma aplicação própria |
| [vLLM](https://docs.vllm.ai/) | Serving e processamento em GPU | Concorrência, batching e APIs compatíveis, conforme modelo |
| [SGLang](https://docs.sglang.io/) | Serving de modelos generativos | Inferência com otimizações e suporte a arquiteturas específicas |
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) | Otimização de inferência NVIDIA | Ambiente com GPUs e modelos compatíveis |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | Servidor de inferência para diferentes backends | Operação de modelos de tipos distintos |
| [BentoML](https://docs.bentoml.com/) / [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) | Empacotamento e serviços de inferência | APIs, composição e escala |
| [ONNX Runtime](https://onnxruntime.ai/docs/) | Runtime de modelos ONNX | Portabilidade e aceleração nos providers suportados |
| [Ultralytics YOLO](https://docs.ultralytics.com/) / [Detectron2](https://github.com/facebookresearch/detectron2) | Detecção e segmentação | Visão computacional especializada; confira licença e manutenção |

**TGI:** [Text Generation Inference está em manutenção](https://huggingface.co/docs/inference-endpoints/engines/tgi). A Hugging Face indica vLLM e SGLang como alternativas nos Inference Endpoints. Migrar exige validar modelo, quantização e comportamento da API.

## 4. Dados, documentos e ML

| Ferramentas | Uso |
| --- | --- |
| [PyTorch](https://pytorch.org/docs/stable/index.html), [JAX](https://docs.jax.dev/), [TensorFlow](https://www.tensorflow.org/) | Computação numérica, treinamento e inferência |
| [Transformers](https://huggingface.co/docs/transformers/index), [Diffusers](https://huggingface.co/docs/diffusers/index) | Modelos de linguagem/multimodais e difusão |
| [scikit-learn](https://scikit-learn.org/stable/), [spaCy](https://spacy.io/), [OpenCV](https://docs.opencv.org/) | ML clássico, NLP e visão |
| [pandas](https://pandas.pydata.org/docs/), [Polars](https://docs.pola.rs/), [NumPy](https://numpy.org/doc/) | Manipulação de dados e arrays |
| [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/python/) | Gráficos e exploração |
| [Docling](https://docling-project.github.io/docling/), [Unstructured](https://docs.unstructured.io/), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Conversão de documentos, layout, tabelas e OCR |
| [Airbyte](https://docs.airbyte.com/), [dbt](https://docs.getdbt.com/), [Label Studio](https://labelstud.io/guide/) | Integração, transformação e anotação de dados |

## 5. Fine-tuning e otimização

| Ferramenta | Papel |
| --- | --- |
| [Transformers Trainer](https://huggingface.co/docs/transformers/trainer) | Loop de treinamento de propósito geral |
| [PEFT](https://huggingface.co/docs/peft/index) | LoRA e outras formas de adaptação eficiente |
| [TRL](https://huggingface.co/docs/trl/index) | SFT, DPO, GRPO e outros métodos de pós-treinamento |
| [Unsloth](https://docs.unsloth.ai/) | Ajuste e inferência com otimizações específicas |
| [Axolotl](https://docs.axolotl.ai/) | Configuração e pipelines de treinamento |
| [torchtune](https://github.com/pytorch/torchtune), [Ludwig](https://github.com/ludwig-ai/ludwig) | Receitas PyTorch / configuração declarativa; confira atividade e compatibilidade antes de adotar |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | Compressão e quantização para modelos compatíveis |

GRPO é uma técnica de otimização de política com comparação relativa em grupos; não é substituto universal de SFT ou DPO. Escolha método conforme dados, recompensa, tarefa e custo. Referência: [TRL/GRPO](https://huggingface.co/docs/trl/grpo_trainer).

## 6. Construção visual e automação

| Categoria | Ferramentas | Uso |
| --- | --- | --- |
| Apps de IA | [Dify](https://docs.dify.ai/), [Langflow](https://docs.langflow.org/), [Flowise](https://docs.flowiseai.com/) | Workflows, recuperação e integrações visuais |
| Workflows de prompts | [Prompt flow](https://microsoft.github.io/promptflow/) | Desenvolvimento e avaliação de fluxos |
| Agentes gerenciados | [Vertex AI Agent Builder](https://cloud.google.com/products/agent-builder) | Integrações e operação no Google Cloud |
| Automação de apps | [n8n](https://docs.n8n.io/), [Make](https://www.make.com/), [Zapier](https://zapier.com/) | Eventos, conectores e regras de negócio |
| Conversação visual | [Typebot](https://docs.typebot.io/) | Fluxos de chatbot e formulários |
| Apps web/mobile | [Bubble](https://manual.bubble.io/), [FlutterFlow](https://docs.flutterflow.io/) | Construção visual de aplicações |
| Ferramentas internas | [Retool](https://docs.retool.com/), [Appsmith](https://docs.appsmith.com/), [Budibase](https://docs.budibase.com/) | Painéis, formulários e administração |
| Sites | [Webflow](https://webflow.com/), [Framer](https://www.framer.com/) | Publicação visual de sites |

Self-hosted, código visível e open source são propriedades distintas. Leia licenças, limites de uso, custos dos conectores e recursos da edição escolhida.

## 7. Programação assistida por IA

| Ferramenta | Interface/papel |
| --- | --- |
| [Codex](https://developers.openai.com/learn/codex) | Agente de programação com interfaces de aplicativo, terminal e editor |
| [Claude Code](https://code.claude.com/docs/en/overview) | Agente de programação no terminal e integrações |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Agente de terminal no ecossistema Gemini |
| [OpenCode](https://opencode.ai/docs/) | Agente de terminal com configuração de providers |
| [Aider](https://aider.chat/docs/) | Edição de código no terminal com integração Git |
| [Cursor](https://cursor.com/docs), [Windsurf](https://docs.windsurf.com/), [Trae](https://www.trae.ai/) | Editores com IA e fluxos de agentes |
| [GitHub Copilot](https://docs.github.com/en/copilot), [JetBrains AI](https://www.jetbrains.com/ai/) | Assistência e agentes no ecossistema de desenvolvimento |
| [Continue](https://docs.continue.dev/), [Zed](https://zed.dev/docs/ai/overview), [Tabnine](https://docs.tabnine.com/) | Integrações e fluxos de IA em editores |
| [Amazon Q Developer](https://aws.amazon.com/q/developer/) | Assistência ao desenvolvimento no ecossistema AWS; substitui a referência antiga a CodeWhisperer |
| [Bolt](https://support.bolt.new/), [v0](https://v0.app/docs), [Lovable](https://docs.lovable.dev/), [Replit Agent](https://docs.replit.com/) | Criação de aplicações em ambientes gerenciados |

Verifique contexto enviado ao fornecedor, execução de comandos, escopo de arquivos, configuração do projeto e revisão do diff. Não há multiplicador universal de produtividade. **Sourcegraph Cody**, antes listado aqui, deve ser avaliado a partir do [catálogo atual da Sourcegraph](https://sourcegraph.com/), sem presumir disponibilidade dos planos antigos.

## 8. Infraestrutura e execução durável

| Ferramentas | Papel |
| --- | --- |
| [Docker](https://docs.docker.com/), [Podman](https://podman.io/docs) | Containers e ambientes reproduzíveis |
| [Kubernetes](https://kubernetes.io/docs/), [KEDA](https://keda.sh/docs/), [Knative](https://knative.dev/docs/) | Orquestração, escala por eventos e serving serverless |
| [KServe](https://kserve.github.io/website/), [llm-d](https://llm-d.ai/) | Serving de modelos e inferência distribuída em Kubernetes |
| [Temporal](https://docs.temporal.io/), [Dapr](https://docs.dapr.io/) | Execução durável / blocos de construção para sistemas distribuídos |
| [Airflow](https://airflow.apache.org/docs/), [Prefect](https://docs.prefect.io/), [Kestra](https://kestra.io/docs/) | Orquestração de pipelines |
| [Ray](https://docs.ray.io/), [Modal](https://modal.com/docs), [Runpod](https://docs.runpod.io/) | Computação distribuída e workloads em GPU |
| [Vertex AI](https://cloud.google.com/vertex-ai/docs), [SageMaker](https://docs.aws.amazon.com/sagemaker/), [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/) | Plataformas gerenciadas de ML |
| [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/), [Microsoft Foundry](https://learn.microsoft.com/en-us/azure/ai-foundry/) | Acesso gerenciado a modelos e recursos de aplicações de IA |

Retentativas não garantem que um workflow nunca falhe. Ações externas precisam de idempotência, prazos, compensação e visibilidade de falhas.

## 9. Avaliação, observabilidade e custos

| Ferramentas | Papel |
| --- | --- |
| [Ragas](https://docs.ragas.io/), [DeepEval](https://deepeval.com/docs/getting-started), [Promptfoo](https://www.promptfoo.dev/docs/intro/) | Avaliação de respostas, RAG e regressões |
| [Braintrust](https://www.braintrust.dev/docs), [LangSmith](https://docs.langchain.com/langsmith/home) | Datasets, experimentos e tracing |
| [Langfuse](https://langfuse.com/docs), [Phoenix](https://arize.com/docs/phoenix), [Opik](https://www.comet.com/docs/opik/) | Observabilidade e avaliações |
| [MLflow](https://mlflow.org/docs/latest/), [Weights & Biases](https://docs.wandb.ai/), [DVC](https://dvc.org/doc) | Experimentos, artefatos e versionamento |
| [Datadog LLM Observability](https://docs.datadoghq.com/llm_observability/), [OpenTelemetry](https://opentelemetry.io/docs/) | Monitoramento e instrumentação |
| [Helicone AI Gateway](https://docs.helicone.ai/gateway/overview), [LiteLLM](https://docs.litellm.ai/) | Gateway, custos, limites e roteamento conforme configuração |

A integração antiga de proxy OpenAI-compatible do Helicone é mantida, mas o fornecedor direciona novas integrações ao AI Gateway. A entrada **Lithic** do catálogo anterior foi removida da categoria de custos de LLM por não haver sustentação adequada para aquela classificação.

## 10. Peças complementares para produtos

| Ferramentas | Uso |
| --- | --- |
| [Supabase](https://supabase.com/docs), [Logto](https://docs.logto.io/) | Backend/autenticação e controle de identidade |
| [PostHog](https://posthog.com/docs) | Analytics de produto, experimentos e feature flags |
| [AppFlowy](https://docs.appflowy.io/), [Penpot](https://help.penpot.app/) | Organização de conhecimento e design |
| [Coolify](https://coolify.io/docs/) | Deploy em infraestrutura própria |

Licenças e recursos variam entre componentes e edições. Esta seção trata de apoio ao produto; essas ferramentas não são modelos de IA.
