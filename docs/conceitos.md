# Conceitos fundamentais de IA

[← Voltar ao canivete](../README.md) · Revisão: 10/11/2026

## 1. Arquiteturas e tipos de modelos

| Conceito | O que é | O que observar |
| --- | --- | --- |
| LLM | Modelo de linguagem que estima e gera sequências de tokens | Não há faixa universal de parâmetros; qualidade depende também de dados, treinamento e inferência |
| SLM | Termo relativo para modelos de linguagem menores | Não existe corte único; memória e latência dependem de precisão, contexto e hardware |
| VLM | Modelo que combina visão e linguagem | Suporte a imagens não implica suporte nativo a vídeo ou áudio |
| Multimodal | Processa ou gera mais de uma modalidade | Verifique separadamente entradas e saídas: texto, imagem, áudio e vídeo |
| Transformer | Arquitetura baseada em mecanismos de atenção | Base de muitos LLMs; não é sinônimo de todo modelo generativo |
| MoE | Mixture of Experts: roteamento para subconjuntos de especialistas | Parâmetros totais e ativos são diferentes; pesos inativos também podem ocupar memória |
| Difusão | Família de processos generativos baseados em remoção de ruído ou formulações relacionadas | Comum em imagem, áudio e vídeo; custo varia com resolução e amostragem |
| SSM e híbridos | Modelos de espaço de estados e combinações com atenção | Mamba e RWKV são referências de arquiteturas alternativas; confira o suporte do runtime |
| Modelo base | Modelo antes do ajuste específico para interação | Pode exigir template, fine-tuning ou configuração própria |
| Instruct/chat | Modelo ajustado para seguir instruções/conversar | Use o chat template do modelo; papéis e tokens especiais importam |
| Modelo de raciocínio | Modelo treinado para tarefas que se beneficiam de computação adicional na inferência | Pode trocar latência e custo por qualidade; não presuma acesso ao raciocínio interno |

Referências: [Transformers](https://huggingface.co/docs/transformers/index), [model cards](https://huggingface.co/docs/hub/model-cards), [Mamba](https://github.com/state-spaces/mamba), [RWKV](https://github.com/BlinkDL/RWKV-LM).

## 2. Recuperação, busca e RAG

| Conceito | O que faz | Limitação ou escolha prática |
| --- | --- | --- |
| RAG | Recupera conteúdo externo e o inclui no contexto da geração | Não atualiza os pesos nem elimina alucinações; conteúdo recuperado pode estar errado |
| RAG em duas etapas | Recupera antes de gerar | Fluxo previsível; baseline para documentação e FAQ |
| Agentic RAG | Um agente decide quando e como consultar ferramentas de busca | Pode repetir buscas e mudar estratégia; limite passos, tempo e custo |
| Self-RAG | Abordagem de pesquisa com recuperação e autorreflexão aprendidas | Não é apenas pedir ao chat para revisar a própria resposta |
| Corrective RAG | Abordagem de pesquisa que avalia e corrige a recuperação | Não é garantia de correção factual; exige implementação e avaliação |
| Busca lexical | Recuperação por termos, por exemplo BM25 | Útil para identificadores, códigos e nomes exatos |
| Busca vetorial | Busca por proximidade entre embeddings | Depende do modelo, métrica, filtros e distribuição dos dados |
| Busca semântica | Busca orientada a significado | Pode usar vetores e outros métodos; não é uma terceira modalidade independente por definição |
| Busca híbrida | Combina sinais, frequentemente lexical + vetorial | Normalize scores ou use fusão, como reciprocal rank fusion (RRF) |
| GraphRAG | Recuperação/geração que aproveita estruturas de grafo | A implementação da Microsoft inclui extração e resumos de comunidades; não se limita a grafo + vetores |
| Chunking | Divide documentos em unidades recuperáveis | Ajuste tamanho e sobreposição ao conteúdo; preserve título, origem e versão |
| Embedding | Representação numérica de texto ou outra entrada | Dimensão e limites dependem do modelo; trocar modelo geralmente exige reindexação |
| Reranking | Reordena candidatos com modelo ou função adicional | Adiciona custo e latência; meça ganho na sua base |
| Query rewriting | Reformula ou expande a consulta | Pode ajudar ou distorcer a intenção; preserve a consulta original |
| Knowledge graph | Representa entidades e relações | Relações extraídas por LLM precisam de evidência e controle de qualidade |

Referências: [arquiteturas RAG](https://docs.langchain.com/oss/python/deepagents/retrieval), [GraphRAG](https://microsoft.github.io/graphrag/), [Self-RAG](https://arxiv.org/abs/2310.11511), [CRAG](https://arxiv.org/abs/2401.15884), [busca híbrida no Qdrant](https://qdrant.tech/documentation/concepts/hybrid-queries/).

## 3. Treinamento e adaptação

| Técnica | O que muda | Quando considerar |
| --- | --- | --- |
| Transfer learning | Reutiliza conhecimento de um modelo pré-treinado | Base para adaptar modelos a tarefas específicas |
| SFT/fine-tuning supervisionado | Ajusta pesos com exemplos de entradas e saídas | Formato, estilo, classificação e comportamento; não substitui uma base de fatos atualizada |
| LoRA | Treina matrizes de baixa ordem em camadas selecionadas | Reduz parâmetros treináveis; economia depende de módulos, rank e otimizador |
| QLoRA | Combina base quantizada com treinamento de adaptadores | Reduz memória em configurações compatíveis; não equivale a treinar todos os pesos em 4 bits |
| RLHF | Usa feedback humano em processos de alinhamento, frequentemente com recompensa e RL | Exige dados e avaliação de preferências |
| DPO | Otimiza preferências a partir de pares de respostas | Não exige o mesmo loop de RL do RLHF clássico |
| Destilação | Treina um modelo com sinais de outro | Confira licença, qualidade e custo dos dados |
| Quantização | Representa pesos e/ou ativações com menor precisão | Memória, velocidade e perda de qualidade variam; não há aceleração universal |

Separe treino, validação e teste; remova duplicatas e vazamento de respostas. Compare o ajuste com o modelo base no mesmo conjunto de avaliação. Não existe duração de treino, número de exemplos ou rank LoRA universal.

Referências: [PEFT/LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora), [QLoRA](https://arxiv.org/abs/2305.14314), [TRL/SFT](https://huggingface.co/docs/trl/sft_trainer), [TRL/DPO](https://huggingface.co/docs/trl/dpo_trainer).

## 4. Prompts, contexto e saída

| Conceito | Definição | Cuidado |
| --- | --- | --- |
| Prompt engineering | Especificar tarefa, contexto, restrições e exemplos | Avalie mudanças; texto mais longo não é necessariamente melhor |
| Zero-shot / few-shot | Executar sem exemplos / incluir exemplos no contexto | Few-shot não altera pesos; quantidade depende da tarefa |
| Chain-of-thought (CoT) | Técnica de pesquisa com etapas intermediárias em tarefas de raciocínio | Explicação gerada não prova o processo interno nem garante acerto |
| Context engineering | Selecionar e organizar instruções, histórico, documentos e ferramentas | Considere prioridade, tamanho, atualidade e autorização |
| Temperature | Ajusta a distribuição de amostragem quando suportada | Zero não garante determinismo; alguns modelos restringem o parâmetro |
| Top-k / top-p | Restringem candidatos por quantidade / massa de probabilidade | Suporte e interação variam entre APIs |
| Structured outputs | Geração com schema, por exemplo JSON Schema | Conformidade estrutural não prova que os valores são verdadeiros |
| Tool/function calling | Modelo propõe chamada estruturada para a aplicação executar | A aplicação valida argumentos e permissões; o modelo não concede acesso |
| Context window | Orçamento de tokens conforme modelo/API | Não é memória infinita; saída, ferramentas e raciocínio podem consumir parte do limite |
| Memória de aplicação | Estado mantido fora do modelo | Defina isolamento por usuário, expiração, edição e exclusão |
| MCP | Protocolo para integrar aplicações de IA a ferramentas, recursos e prompts | Não substitui autorização nem framework de agentes |
| A2A | Protocolo de comunicação entre agentes | Diferente de MCP; útil quando agentes precisam interoperar |

Referências: [CoT](https://arxiv.org/abs/2201.11903), [saídas estruturadas](https://developers.openai.com/api/docs/guides/structured-outputs), [MCP](https://modelcontextprotocol.io/docs/learn/architecture), [A2A](https://a2a-protocol.org/latest/).

## 5. Inferência, infraestrutura e métricas

| Conceito | O que medir ou entender |
| --- | --- |
| Token | Unidade do tokenizer; não há conversão fixa entre caracteres, palavras e tokens em todos os idiomas |
| TTFT | Tempo até o primeiro token; registre se inclui rede, fila e recuperação |
| TPS | Tokens por segundo; diferencie taxa por requisição e throughput agregado |
| Latência p50/p95/p99 | Distribuição do tempo de resposta sob uma carga definida |
| Prefill / decode | Processamento inicial do contexto / geração dos tokens seguintes |
| KV cache | Cache de chaves e valores da atenção; cresce com contexto, arquitetura e concorrência |
| Prefix/prompt caching | Reutiliza processamento de prefixos compatíveis; retenção e cobrança variam |
| Continuous batching | Inclui e remove requisições de lotes durante a execução |
| PagedAttention | Gerencia memória do KV cache em blocos; não é uma versão universal “v2” |
| FlashAttention | Família de algoritmos de atenção eficientes no acesso à memória; depende da implementação e GPU |
| Speculative decoding | Usa propostas de tokens e verificação para acelerar a decodificação em configurações compatíveis |
| Streaming | Entrega saída incremental; não elimina recuperação nem garante latência inferior a um segundo |
| Batch assíncrono | Processa lotes com prazo de conclusão; descontos dependem do provedor |
| Alucinação | Saída incorreta ou sem sustentação apresentada como fato; avalie evidência, abstenção e tipos de erro |

Para benchmarks, registre modelo e revisão, hardware, engine, quantização, comprimentos de entrada/saída, concorrência, warm-up, qualidade e custo por tarefa concluída. Compare condições equivalentes.

Referências: [vLLM](https://docs.vllm.ai/), [PagedAttention](https://arxiv.org/abs/2309.06180), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [speculative decoding](https://arxiv.org/abs/2211.17192).

## 6. Outras tarefas

| Área | Uso |
| --- | --- |
| NLP | Classificação, extração de entidades, análise textual e tradução |
| Visão computacional | Detecção, segmentação, classificação e análise visual |
| OCR | Extração de texto de imagens/documentos; confira ordem de leitura e tabelas |
| ASR/STT | Transcrição de fala; avalie sotaques, ruído, pontuação e diarização |
| TTS | Síntese de fala; avalie inteligibilidade, pronúncia, latência e direitos sobre vozes |
| Text-to-image / image-to-text | Geração visual / descrição e interpretação de imagens |

Veja as ferramentas e catálogos em [modelos e serviços](modelos-servicos.md).
