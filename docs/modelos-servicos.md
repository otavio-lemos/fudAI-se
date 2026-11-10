# Modelos, bancos, APIs e serviços

[← Voltar ao canivete](../README.md) · Levantamento: 10/11/2026

Este é um retrato datado, com exemplos identificáveis e links oficiais. Não é uma lista exaustiva nem uma classificação de qualidade. Um modelo disponível em chat pode não estar disponível na API, no mesmo país ou no mesmo plano. IDs de API, nomes de checkpoints e nomes de produtos são coisas diferentes.

## 1. Modelos via API

| Família / exemplos do catálogo consultado | IDs ou referência | Papel e observações |
| --- | --- | --- |
| GPT-6 Astra, GPT-6.1 Sol, GPT-6.1 Terra, GPT-6 Luna | `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra`, `gpt-6-luna` · [OpenAI](https://developers.openai.com/api/docs/models) | Diferentes perfis de capacidade, custo e latência; confira esforço de raciocínio e ferramentas por modelo |
| Claude Fable 5.2, Opus 5.5, Sonnet 5.5, Haiku 5.5 | `claude-fable-5-2`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` · [Anthropic](https://platform.claude.com/docs/en/models/overview) | Thinking adaptativo e tool use conforme modelo/plataforma; IDs em clouds podem diferir |
| Gemini 3.8 Flash e família Gemini 3 | `gemini-3.8-flash` · [Google](https://ai.google.dev/gemini-api/docs/models?authuser=002) | Texto, multimodalidade e ferramentas conforme variante; Live, imagem e TTS têm endpoints próprios |
| Gemini 4 Argon | `gemini-4-argon` · [tabela oficial de preços](https://ai.google.dev/gemini-api/docs/pricing?frmapp=yes) | Presente na página de preços consultada; disponibilidade e limites devem ser conferidos na conta/API antes de migrar |
| Grok 4.7 | [Catálogo xAI](https://docs.x.ai/developers/models) | Raciocínio e ferramentas; acesso a busca na web/X depende das ferramentas habilitadas |
| Mistral Large 4 e especialistas | `mistral-large-4` · [Mistral](https://docs.mistral.ai/) | Generalistas e modelos especializados em código, OCR e áudio conforme catálogo |
| Command e modelos Cohere | [Cohere](https://docs.cohere.com/docs/models) | Geração, ferramentas, busca e documentos; modelos de embedding/rerank são produtos distintos |

**Situação das fontes Google:** as páginas consultadas em idiomas/URLs diferentes não estavam sincronizadas. O catálogo em inglês mostrava 3.8 Flash, enquanto a tradução em português ainda destacava 3.7; Argon aparecia na página de preços. Por isso o guia não infere disponibilidade universal nem reproduz limites de uma página na outra. Confira o endpoint escolhido antes do uso.

A edição anterior tratava GPT-4o, Claude 3.5, Gemini 1.5/2.0 e Grok 2 como retrato atual. Agora a referência é o catálogo datado e o ciclo de vida do fornecedor. Isso não significa que todos os modelos antigos tenham o mesmo status de aposentadoria.

## 2. Pesos disponíveis: famílias e checkpoints recentes

**B = bilhões; T = trilhões de parâmetros.** Valores abaixo são os declarados nas model cards, não estimativas de qualidade ou memória. Parâmetros ativos por token não representam o total de pesos que precisa ser armazenado.

| Família / checkpoint oficial | Parâmetros / arquitetura declarados | O que acrescenta ao mapa |
| --- | --- | --- |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B no componente de linguagem; encoder visual separado; atenção híbrida | Texto/visão e modos thinking/instruct; o tamanho total exibido no Hub pode incluir outros componentes |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Backbone MoE 552B; 8B ativos no prefill e 16B no decode, segundo a card | Arquitetura causal encoder-decoder, compressão de KV e esforço de raciocínio configurável |
| [Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) | MoE 2.8T totais / 104B ativos | Multimodalidade, atenção híbrida e tarefas de agentes; exige infraestrutura muito além de um laptop |
| [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) | Aproximadamente 428B totais / 23B ativos | Entradas texto, imagem e vídeo; atenção esparsa |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 320B totais / 18B ativos | Multimodalidade, atenção esparsa/linear e parâmetros próprios de thinking |
| [Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) | Cerca de 1T total / 52B ativos incluindo embeddings e saída | Modelo multimodal, instruct e raciocínio; a card distingue 49B ativos sem esses componentes |
| [Granite 4.1 8B](https://huggingface.co/ibm-granite/granite-4.1-8b) | 8B, denso | Opção menor com documentação de ferramentas; catálogo inclui outros tamanhos e especialistas |
| [Gemma 4](https://ai.google.dev/gemma/docs) | Família com variantes; conferir card de cada tamanho | Modelos locais e variantes especializadas, como EmbeddingGemma e FunctionGemma |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) / [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | 21B/3.6B ativos e 117B/5.1B ativos, respectivamente | Pesos disponíveis, raciocínio configurável e formato Harmony obrigatório |
| [Llama](https://huggingface.co/meta-llama), [Phi](https://huggingface.co/microsoft), [Nemotron](https://huggingface.co/nvidia), [Command](https://huggingface.co/CohereForAI) | Catálogos com tamanhos, modalidades e licenças diferentes | Famílias continuam no mapa; selecione checkpoint e revisão concretos para deploy |

Não misture “open weights”, código aberto e licença comercial irrestrita. Confira a licença do checkpoint e dos derivados, restrições de redistribuição e compatibilidade do runtime. Variantes quantizadas da comunidade podem usar templates, revisões ou conversões diferentes das oficiais.

**Referências históricas:** Llama 3.x, DeepSeek V3/R1, Qwen2.5/CodeQwen/QwQ, Mixtral, Phi-3.5/4, Gemma 2, Yi-1.5, Mathstral, Code Llama e StarCoder2 ajudam a entender gerações anteriores. Não devem ser apresentados automaticamente como as versões mais recentes ou melhores. As famílias atuais acima e seus catálogos são o ponto de partida para novas comparações.

## 3. Embeddings e rerankers

| Família / serviço | Uso | O que conferir |
| --- | --- | --- |
| [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings) | Vetorização de texto | Modelo, dimensões suportadas e limites de entrada |
| [Cohere Embed 5 Pro/Fast](https://docs.cohere.com/changelog/embed-v5) | Texto, imagem e entradas mistas | IDs `embed-v5.0-pro` / `embed-v5.0-fast`; dimensões e tipos de saída |
| [Cohere Rerank 5](https://docs.cohere.com/docs/models) | Reordenação de documentos | `rerank-v5.0-pro` / `rerank-v5.0-fast`; schema, idiomas e truncamento |
| [Voyage 4](https://docs.voyageai.com/docs/embeddings) | Recuperação geral e multilíngue | `voyage-4-large`, `voyage-4`, `voyage-4-lite`; compatibilidade declarada dentro da série |
| [Qwen embeddings/rerankers](https://huggingface.co/Qwen) | Recuperação com pesos disponíveis | Checkpoint específico, instruções de consulta e licença |
| [BGE](https://huggingface.co/BAAI), [Nomic](https://huggingface.co/nomic-ai), [EmbeddingGemma](https://ai.google.dev/gemma/docs) | Embeddings locais e variantes multimodais | Idiomas, tarefa, pooling e normalização |
| [Sentence Transformers](https://sbert.net/) | Embeddings e cross-encoders | Dataset de treinamento, tarefa e uso correto de query/document |

Dimensão maior não prova qualidade superior. Não compare vetores de modelos distintos como se compartilhassem um espaço, salvo compatibilidade explicitamente declarada. Avalie recuperação em português, nomes próprios, siglas e documentos da sua área.

## 4. Código, visão e documentos

| Necessidade | Famílias/ferramentas de referência | Critério de escolha |
| --- | --- | --- |
| Código e agentes de software | Modelos generalistas atuais, famílias Qwen/DeepSeek/GLM/Kimi/MiniMax e [modelos Mistral](https://docs.mistral.ai/) | Resolver tarefas no repositório com testes, limites de ferramentas e custo medido |
| Visão e documentos | Modelos multimodais acima, [Qwen](https://huggingface.co/Qwen), [Gemma](https://ai.google.dev/gemma/docs), [Docling](https://docling-project.github.io/docling/) | OCR, tabelas, gráficos, localização e ordem de leitura |
| Visão clássica | [YOLO](https://docs.ultralytics.com/), [OpenCV](https://docs.opencv.org/), [Detectron2](https://github.com/facebookresearch/detectron2) | Detecção/segmentação especializada, com métricas próprias |
| OCR gerenciado | [Mistral OCR](https://docs.mistral.ai/), [Google Document AI](https://cloud.google.com/document-ai/docs), [Amazon Textract](https://docs.aws.amazon.com/textract/) | Qualidade por tipo de documento, custos e tratamento dos dados |

Uma URL de PDF não é uma imagem. Use o mecanismo de arquivos/PDF da API ou converta páginas em imagens compatíveis. Validação de schema não substitui conferência de números, campos ausentes e evidências de origem.

## 5. Imagem e vídeo

| Família / plataforma | Papel | Verificação necessária |
| --- | --- | --- |
| [GPT Image](https://developers.openai.com/api/docs/models) | Catálogo inclui GPT-Image-2.6 Sunburst/Flare para geração e edição | ID, modalidade e parâmetros da API de imagem |
| [FLUX 3](https://docs.bfl.ai/quick_start/introduction) | Família visual atual destacada pela BFL; geração/edição conforme endpoint | Diferencie API, pesos disponíveis e licenças por variante |
| [Stable Diffusion](https://huggingface.co/stabilityai), [Qwen Image](https://huggingface.co/Qwen) | Ecossistemas com modelos e pesos disponíveis | Licença, VRAM, pipeline e qualidade na tarefa |
| [Midjourney](https://docs.midjourney.com/), [Ideogram](https://docs.ideogram.ai/) | Geração visual como serviço | Interface, automação oficialmente suportada e direitos de uso |
| [Wan](https://huggingface.co/Wan-AI), [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo), [LTX-2](https://huggingface.co/Lightricks/LTX-2) | Vídeo com modelos/pipelines disponíveis | Condicionamento, duração, resolução, FPS, memória e licença |
| [Veo/Gemini vídeo](https://ai.google.dev/gemini-api/docs/video), [Kling](https://app.klingai.com/global/), [Runway](https://docs.dev.runwayml.com/) | Vídeo gerenciado | Região, acesso à API, áudio, limites de clipe e custo |
| [ComfyUI](https://docs.comfy.org/), [Diffusers](https://huggingface.co/docs/diffusers/index) | Workflows e pipelines | Versões dos nós, modelos auxiliares e dependências |

DALL-E 3, FLUX.1 e Ideogram 2.0 eram as referências antigas do README; não representam sozinhas o catálogo atual. Para produção audiovisual, mantenha parâmetros de resolução, proporção, duração, FPS, seed e condicionamento junto de cada geração. O mesmo seed não garante identidade entre versões, hardwares ou pipelines.

## 6. Áudio: transcrição, síntese e tempo real

| Categoria | Referências | O que avaliar |
| --- | --- | --- |
| ASR local | [Whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Granite Speech](https://huggingface.co/ibm-granite) | WER/CER em pt-BR, memória, timestamp e ruído |
| ASR gerenciado | [OpenAI](https://developers.openai.com/api/docs/models), [Google](https://cloud.google.com/speech-to-text/docs), [Deepgram](https://developers.deepgram.com/docs/models-languages-overview) | Modelo/idioma, diarização, vocabulário e latência; não presuma que Nova-2 seja a geração atual |
| TTS local | [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice), [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [F5-TTS](https://github.com/SWivid/F5-TTS) | Idioma suportado pelo checkpoint, vozes, licença dos pesos e pronúncia |
| TTS gerenciado | [ElevenLabs](https://elevenlabs.io/docs/overview), [OpenAI áudio](https://developers.openai.com/api/docs/guides/audio), [Gemini speech](https://ai.google.dev/gemini-api/docs/speech-generation), [Cartesia](https://docs.cartesia.ai/) | Expressividade, estabilidade, tempo até primeiro áudio e direitos sobre a voz |
| Conversação em tempo real | [OpenAI Realtime](https://developers.openai.com/api/docs/guides/realtime), [Gemini Live](https://ai.google.dev/gemini-api/docs/live) | VAD, interrupções, codecs, turnos e integração de ferramentas |

Bark e PlayHT eram referências do guia anterior; avalie disponibilidade e manutenção antes de reutilizar exemplos antigos. A licença do código de TTS pode diferir da licença dos pesos e do dataset. Uma adaptação pt-BR de F5-TTS exige avaliação própria; acentuar uma palavra no texto não garante a pronúncia desejada.

## 7. Bancos e busca

| Tecnologia | Papel | Quando considerar |
| --- | --- | --- |
| [PostgreSQL + pgvector](https://github.com/pgvector/pgvector) | Dados relacionais e vetores | Aproveitar infraestrutura SQL existente e controles de acesso |
| [Qdrant](https://qdrant.tech/documentation/), [Weaviate](https://docs.weaviate.io/), [Milvus](https://milvus.io/docs) | Bancos com recursos de recuperação vetorial | Filtros, escala, operações e busca híbrida conforme produto |
| [Pinecone](https://docs.pinecone.io/), [Chroma](https://docs.trychroma.com/) | Recuperação gerenciada / desenvolvimento e deploy conforme edição | Consistência, isolamento, custo e perfil de carga |
| [Elasticsearch](https://www.elastic.co/docs), [OpenSearch](https://docs.opensearch.org/) | Busca lexical, vetorial e híbrida | Bases com necessidade de termos exatos, filtros e agregações |
| [Vespa](https://docs.vespa.ai/), [LanceDB](https://docs.lancedb.com/) | Busca/ranking e armazenamento vetorial | Ranking customizado ou integração com dados multimodais |
| [Neo4j](https://neo4j.com/docs/), [ArangoDB](https://docs.arangodb.com/) | Grafos / multimodelo | Relações e consultas de grafo; confira licença e edição |

Não há banco “mais usado” ou “mais barato” universal. Meça recall, latência sob concorrência, cardinalidade dos filtros, atualizações e custo de operação com seu dataset.

## 8. APIs, gateways e hospedagem

| Categoria | Serviços | Observação |
| --- | --- | --- |
| Busca para agentes | [Tavily](https://docs.tavily.com/), [Exa](https://docs.exa.ai/), [Brave Search](https://api-dashboard.search.brave.com/documentation), [Serper](https://serper.dev/), [Perplexity](https://docs.perplexity.ai/) | Diferencie resultados de busca, extração de páginas e respostas geradas com citações |
| Conteúdo web | [Firecrawl](https://docs.firecrawl.dev/) | Extração/rastreamento; qualidade e acesso ao conteúdo variam |
| Gateways/roteamento | [OpenRouter](https://openrouter.ai/docs/quickstart), [LiteLLM](https://docs.litellm.ai/), [Helicone](https://docs.helicone.ai/gateway/overview) | Uma interface comum não garante recursos idênticos nem políticas de dados iguais |
| Inferência gerenciada | [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index), [Together AI](https://docs.together.ai/), [Fireworks](https://docs.fireworks.ai/), [Replicate](https://replicate.com/docs) | Catálogo, hardware e cobrança por endpoint/modelo |
| Hardware/serviços especializados | [Groq](https://console.groq.com/docs/overview), [Cerebras](https://inference-docs.cerebras.ai/), [SambaNova](https://docs.sambanova.ai/) | Latência e throughput dependem do modelo e da carga |
| Demos e distribuição | [Hugging Face Hub/Spaces](https://huggingface.co/docs/hub/index), [Ollama library](https://ollama.com/library) | Hub distribui artefatos; Spaces hospeda apps; nenhum implica GPU gratuita ilimitada |
| Fine-tuning gerenciado | [Together](https://docs.together.ai/docs/fine-tuning-overview), [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/tuning/overview), [OpenAI](https://developers.openai.com/api/docs/guides/supervised-fine-tuning) | Modelos elegíveis e disponibilidade devem ser verificados; não extrapole suporte entre fornecedores |

A entrada genérica “Anthropic fine-tuning” do guia antigo não deve ser interpretada como uma API pública de ajuste para qualquer modelo Claude. Só inclua uma oferta de ajuste com documentação explícita da plataforma e do modelo elegível.

## 9. Como escolher sem depender de um ranking

1. Defina tarefa, modalidade, privacidade e critérios de sucesso.
2. Separe casos de teste representativos, incluindo erros, perguntas sem resposta e pt-BR.
3. Compare poucos candidatos com o mesmo harness, ferramentas e orçamento.
4. Meça qualidade, custo por tarefa concluída, TTFT, latência total e falhas.
5. Verifique licenças, deprecações, região, retenção e suporte operacional.
6. Fixe a revisão em produção e repita a avaliação antes de migrar.

Consulte [parâmetros](parametros.md) para configurar as chamadas e [fontes](fontes.md) para os critérios de atualização.
