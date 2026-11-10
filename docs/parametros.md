# Parâmetros: modelo, geração, agentes e servidor

[← Voltar ao canivete](../README.md) · Revisão: 10/11/2026

“Parâmetro” pode significar **peso aprendido**, **opção de uma chamada** ou **configuração do servidor**. Este guia separa os três sentidos. Nomes parecidos entre APIs não implicam comportamento equivalente.

## 1. Tamanho do modelo e memória

| Termo | Significado | Consequência prática |
| --- | --- | --- |
| Parâmetros totais | Quantidade de pesos, conforme a convenção da model card | Afeta armazenamento; MoE não carrega apenas os especialistas ativos em todas as implementações |
| Parâmetros ativos | Pesos usados para processar um token ou estágio | Depende da arquitetura; prefill e decode podem ter contagens diferentes |
| Dense / MoE | Rede densa / roteamento para especialistas | Compare qualidade e custo real, não apenas o sufixo B/A do nome |
| BF16 / FP16 | Representações de 16 bits | Aproximadamente 2 bytes por peso, antes de outros custos |
| FP8 / INT8 | Representações de 8 bits | Exigem suporte e calibração/formato compatíveis |
| INT4 / FP4 / MXFP4 / NVFP4 | Famílias de formatos de baixa precisão | Não são intercambiáveis; escalas, agrupamento, kernels e hardware importam |
| GGUF / Safetensors | Formatos de armazenamento | GGUF não é um nível único de quantização; Safetensors também não determina a precisão sozinho |
| KV cache | Estado de atenção durante inferência | Contexto e concorrência podem consumir muita memória além dos pesos |
| Adaptador LoRA | Pesos adicionais de ajuste | Rank, módulos e precisão afetam tamanho e custo |

Estimativa inicial de memória dos pesos: **quantidade de pesos × bits por peso ÷ 8**. Um modelo hipotético com 8 bilhões de pesos em 4 bits usa cerca de 4 GB decimais só para esses valores. Escalas de quantização, buffers, ativações, KV cache e runtime aumentam o total; isso não é uma recomendação de RAM/VRAM.

Fontes: [quantização no Transformers](https://huggingface.co/docs/transformers/quantization/overview), [GGUF/llama.cpp](https://github.com/ggml-org/llama.cpp), [PEFT](https://huggingface.co/docs/peft/index). Para números concretos de modelos, veja [modelos e serviços](modelos-servicos.md#2-pesos-disponíveis-famílias-e-checkpoints-recentes).

## 2. Geração e amostragem

| Controle | Efeito | Atenção |
| --- | --- | --- |
| `temperature` | Reescala a distribuição antes da amostragem | Nem todo modelo aceita; zero não garante reprodutibilidade entre ambientes |
| `top_p` | Mantém candidatos até uma massa de probabilidade | Não é percentual de acerto |
| `top_k` | Limita a quantidade de tokens candidatos | Pode existir no engine local e faltar na API hospedada |
| `min_p` | Filtra tokens por probabilidade relativa ao candidato mais provável | Não equivale a `top_p` |
| `presence_penalty` | Penaliza presença prévia de tokens | Pode alterar repetição e aderência; suporte varia |
| `frequency_penalty` | Penaliza conforme frequência de tokens | Não tem a mesma fórmula de `repetition_penalty` |
| `repetition_penalty` | Penalidade multiplicativa conforme engine | Pode prejudicar código e nomes próprios; confira o valor neutro |
| `seed` | Inicializa aleatoriedade quando suportado | Não fixa modelo, kernels, infraestrutura ou resultados entre versões |
| `stop` / stop token IDs | Condições de término | Não corte tokens do chat template ou estruturas necessárias |
| `logprobs` | Retorna probabilidades de tokens quando disponível | Não é confiança calibrada na veracidade da resposta |
| Limite de saída | Limita tokens gerados conforme API | Pode incluir raciocínio e tokens não visíveis; truncamento exige tratamento |

Nomes comuns para limites: `max_output_tokens` em Responses, `max_completion_tokens` em Chat Completions compatível, `max_tokens` em outras APIs e `max_new_tokens` em geração Transformers. Não os substitua mecanicamente.

Fonte de referência para sampling local: [vLLM SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/). As opções descritas não formam um contrato comum a todos os provedores.

## 3. Raciocínio e verbosidade por ecossistema

| Ecossistema / escopo | Controle documentado | Como interpretar |
| --- | --- | --- |
| OpenAI Responses, modelos compatíveis | `reasoning.effort` | Orçamento/esforço de raciocínio; níveis aceitos dependem do modelo |
| OpenAI, saída textual compatível | `text.verbosity` | Extensão/detalhe da resposta visível; não é sinônimo de esforço de raciocínio |
| Claude Messages, modelos compatíveis | `thinking` e `output_config.effort` | Modo de thinking e esforço são controles distintos; `adaptive` é um modo, não nível de esforço |
| Gemini, conforme geração | `thinkingConfig`, com `thinkingLevel` ou `thinkingBudget` conforme modelo | O SDK pode usar nomes em snake_case; confira a documentação do endpoint e modelo |
| Qwen3.8-27B | Opção de thinking no chat template e configurações de sampling da model card | O modo thinking e o instruct usam recomendações diferentes |
| GLM-5.3-Flash | `reasoning_effort` e `clear_thinking` no template, conforme runtime | A card lista `low`, `high`, `max`; não presuma os níveis de outro fornecedor |
| DeepSeek-V4.1-Flash | Controle de esforço descrito na model card | O modelo declara escala inteira 1–100; confira como o runtime/API a expõe |
| gpt-oss | Esforço `low`, `medium`, `high`, com template Harmony | Não copie os níveis do GPT hospedado para esse checkpoint |

Fontes: [OpenAI/GPT-6](https://developers.openai.com/api/docs/guides/latest-model), [Claude effort](https://platform.claude.com/docs/en/build-with-claude/effort), [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking), [Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B), [GLM](https://huggingface.co/zai-org/GLM-5.3-Flash), [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [gpt-oss](https://huggingface.co/openai/gpt-oss-20b).

### Exemplo de presets específicos, não universais

A model card do **Qwen3.8-27B** recomenda os seguintes pontos de partida:

| Modo | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repetition penalty |
| --- | --- | --- | --- | --- | --- | --- |
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Instruct/non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

São recomendações desse checkpoint na data da consulta, não resultados de benchmark deste repositório. Confirme suporte do engine e avalie na sua tarefa. Não há preset universal “criativo”, “preciso” ou “sem alucinação”.

## 4. Contexto, cache e persistência

| Controle/recurso | Para que serve | O que registrar |
| --- | --- | --- |
| Contexto máximo | Limite aceito pelo modelo/engine | Entrada, saída reservada, ferramentas e modalidades |
| Prompt/prefix caching | Reutilização de prefixos elegíveis | Tokens em cache, políticas de escrita/leitura, retenção e preço |
| Chave de cache | Roteamento/contabilização conforme provedor | Não use segredo ou PII como identificador; não a trate como autorização |
| Compaction | Redução do histórico preservado no contexto | O que foi perdido, estado das ferramentas e qualidade após compactação |
| `store` e retenção | Persistência de objetos de API conforme contrato | `store=False` não substitui a política de retenção de dados do fornecedor |
| Conversation/thread ID | Retomada de histórico/estado | Autenticação, titularidade e isolamento entre usuários |

Memória da aplicação, KV cache e cache de prompts têm finalidades diferentes. Um hit de cache não significa que fatos estão atualizados. Não assuma desconto fixo ou caching gratuito: política e tarifação variam por modelo.

Fontes: [OpenAI cache](https://developers.openai.com/api/docs/guides/prompt-caching), [Claude cache](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence).

## 5. Ferramentas, estrutura e orçamento de agentes

| Controle | Uso | Validação necessária |
| --- | --- | --- |
| Schema de saída / strict mode | Contrato estrutural | Campos ausentes, recusa, resposta incompleta e regras de negócio |
| `tool_choice` ou equivalente | Permitir, exigir ou selecionar ferramenta | Regras e nomes aceitos pelo modelo/API |
| Chamadas paralelas | Ferramentas independentes no mesmo passo | Conflitos, ordenação e efeitos colaterais |
| Máximo de passos/chamadas | Evitar loops e gastos sem limite | Orçamento total, timeout e condição de parada |
| Timeout e retries | Lidar com falhas transitórias | Backoff, idempotência e prevenção de ações duplicadas |
| Aprovação humana | Revisar ação antes do efeito externo | Identidade do aprovador, payload aprovado e prazo |
| Output/tool result limits | Controlar volume de resultados | Não truncar evidências essenciais; preserve origem e erros |
| Streaming | Eventos incrementais | Falhas intermediárias, recusas e evento de conclusão |

Fontes: [saídas estruturadas](https://developers.openai.com/api/docs/guides/structured-outputs), [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts), [MCP](https://modelcontextprotocol.io/docs/learn/architecture).

## 6. Configuração do servidor de inferência

| Opção/recurso, conforme engine | Impacto | Trade-off |
| --- | --- | --- |
| Quantização de pesos e KV | Memória e kernels utilizados | Compatibilidade e qualidade |
| Contexto máximo do engine | Capacidade por sequência | Mais contexto pode reduzir concorrência possível |
| Tensor/pipeline/expert parallelism | Distribuição entre aceleradores | Comunicação, topologia e memória |
| Limite de sequências/tokens em lote | Throughput e pressão de memória | Latência individual e risco de falta de memória |
| Prefix caching | Reutilização de prefixos | Taxa de reutilização e memória ocupada |
| Chunked prefill | Divide processamento de entradas longas | Equilíbrio entre prefill e decode |
| Speculative decoding/MTP | Propostas de múltiplos tokens | Taxa de aceitação, suporte do modelo e custo do draft |
| Prefill/decode disaggregation | Separa estágios em grupos de recursos | Transferência de KV, rede e complexidade operacional |
| Cache-aware routing | Roteia conforme afinidade de cache | Balanceamento de carga e isolamento de tenants |
| CPU offload | Move parte do estado/pesos para CPU | Capacidade em troca de comunicação/latência |

As flags de CLI não são estáveis entre engines e versões. Use a documentação da **versão instalada**, não um comando copiado de outro modelo. Referências: [vLLM](https://docs.vllm.ai/), [SGLang](https://docs.sglang.io/), [llm-d](https://llm-d.ai/).

## 7. Imagem, vídeo e voz

| Modalidade | Parâmetros a mapear | Não confundir |
| --- | --- | --- |
| Imagem | Resolução, proporção, qualidade, seed, steps, guidance, imagem de referência e máscara | Steps/guidance de difusão local podem não existir numa API gerenciada |
| Vídeo | Duração, FPS, resolução, quadros-chave, referências, extensão e áudio | FPS final interpolado não é necessariamente FPS nativo gerado |
| ASR | Idioma, vocabulário, timestamps, diarização e formato | Detecção de idioma não garante transcrição correta |
| TTS | Voz, idioma, velocidade, referência e controles de expressividade | Clonar timbre não garante pronúncia ou autorização de uso |
| Realtime | Codec, sample rate, VAD, interrupção e turnos | Streaming de áudio não implica conversação bidirecional completa |

Consulte os [catálogos por modalidade](modelos-servicos.md). Registre versões do pipeline, modelo e pós-processamento para reproduzir resultados.
