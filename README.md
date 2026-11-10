# fudAI-se

## canivete suíço de IA — conceitos, ferramentas e recursos

Espero que esse canivete suíço de IA salve muitas pessoas como fui salvo pelo [canivete suíço do shell do Aurélio](https://aurelio.net/shell/canivete/). Grande Aurélio!

Um guia em português para entender IA, escolher ferramentas e sair da teoria com exemplos pequenos. A proposta é explicar **o que cada peça faz, quando usá-la e o que conferir antes de colocá-la em produção**.

**Revisão geral: 10 de novembro de 2026.** Catálogos e documentação oficial são a referência para versões, disponibilidade, preços e limites. Este guia não é um ranking de modelos.

## Explore o canivete

| Quero… | Onde ir |
| --- | --- |
| Explorar tecnologias e famílias que faltavam no guia | [Mapa ampliado](docs/mapa-atual.md) |
| Entender tamanhos, sampling, thinking, cache e serving | [Guia de parâmetros](docs/parametros.md) |
| Entender LLM, RAG, LoRA, embeddings e inferência | [Conceitos fundamentais](docs/conceitos.md) |
| Encontrar frameworks, editores, automação e infraestrutura | [Ferramentas e frameworks](docs/ferramentas.md) |
| Escolher modelos, bancos, APIs e serviços | [Modelos e serviços](docs/modelos-servicos.md) |
| Ver os 11 cenários de aplicação e executar exemplos | [Exemplos e arquiteturas](docs/exemplos.md) |
| Proteger dados, ferramentas e aplicações | [Segurança e governança](docs/seguranca.md) |
| Consultar fontes e entender os critérios de atualização | [Metodologia e fontes](docs/fontes.md) |
| Ver o que mudou nesta revisão | [Histórico de alterações](CHANGELOG.md) |

## Qual caminho seguir?

| Objetivo | Comece por | Só acrescente quando necessário |
| --- | --- | --- |
| Consultar documentos próprios | Busca + RAG com fontes | Reranking, busca híbrida e avaliação de recuperação |
| Extrair campos de documentos | OCR/visão + schema + validação | Revisão humana e integração com ERP |
| Executar ações em sistemas | Chamadas de ferramentas com permissões limitadas | Agentes, MCP e aprovação de ações sensíveis |
| Rodar modelos no computador | Ollama, llama.cpp ou MLX no Apple Silicon | Servidor dedicado e benchmarks de concorrência |
| Publicar inferência em GPU | vLLM ou SGLang, conforme suporte do modelo | Autoscaling, filas, cache e observabilidade |
| Ajustar estilo ou comportamento | Prompt e conjunto de avaliação | Fine-tuning/LoRA se houver ganho medido |
| Automatizar apps | n8n, Make, Zapier ou código | Execução durável com Temporal para fluxos que exigem retomada |
| Programar com IA | Editor/agente com Git e testes | Acesso a ferramentas externas com escopo explícito |

## Execute o primeiro exemplo

Requer Python 3.11 ou superior. Na raiz do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r exemplos/requirements.txt
python -m exemplos.revisao_humana
python -m unittest discover -s tests -v
```

Esse exemplo de pausa, edição e retomada com LangGraph funciona **sem chave de API**. Os testes também não fazem chamadas externas. Para RAG, extração e streaming com um modelo real, siga a [configuração dos exemplos](docs/exemplos.md#configuração).

## Como usar este guia

- **Open weights não significa automaticamente open source:** leia a licença de cada modelo, seus pesos, código e dados disponíveis.
- **Mais parâmetros, contexto ou agentes não garantem uma solução melhor:** compare com um baseline na sua tarefa, em português e com seus dados.
- **RAG e JSON válido não garantem verdade:** confira fontes, cálculos e regras de negócio.
- **Desempenho depende do ambiente:** registre hardware, quantização, tamanho das entradas e concorrência antes de comparar números.
- **Catálogo não é recomendação universal:** planos gratuitos, recursos e termos comerciais precisam ser conferidos no fornecedor.

## Contribua

Encontrou erro ou ferramenta relevante? Abra uma issue ou PR com a descrição, link oficial, data da consulta e motivo da alteração. Para exemplos de código, inclua pré-requisitos e a forma de verificar o comportamento. Os critérios completos estão em [metodologia e fontes](docs/fontes.md).
