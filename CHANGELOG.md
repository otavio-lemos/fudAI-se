# Histórico de alterações

**Estrutura atual proposta neste PR:** guia completo e sequencial no `README.md`, com exemplos executáveis em `exemplos/`. O conteúdo das oito páginas de `docs/` foi reunido no README; a divisão em páginas separadas foi substituída.

## 2026-11-10 — auditoria e correções dos orquestradores

- Documenta OpenClaw/ACPX e Hermes, com receita entre OpenCode e Kilo e alternativa com CLIs externos.
- Corrige a omissão de agentes ACP externos no OpenHands e aponta seu exemplo oficial de refinamento iterativo, distinguindo os dois recursos.
- Esclarece seleção efetiva de modelos, diretório, permissões e limites do sandbox no OpenClaw; contexto e worktrees no Hermes; acrescenta o comando nativo `/review`.
- Registra a restrição dos modelos Free do OpenCode ao próprio cliente e confere custos de tarefas auxiliares.
- Corrige contagem dos casos práticos, endereço Milvus, URLs Google e inclui ACP no mapa de protocolos.
- Verifica 300 destinos externos, âncoras, arquivos locais, cinco blocos Python e os 14 testes offline; sem chamadas autenticadas aos modelos.

## 2026-11-10 — curadoria de orquestração por adoção e escopo

- Acrescenta Ruflo/Claude Flow e OpenHands, junto de CrewAI, LangGraph e Microsoft Agent Framework, com licença, estrelas consultadas pela API GitHub e distinção de função.
- Evita apresentar wrappers pequenos como base principal; separa adoção pública de compatibilidade efetivamente testada.

## 2026-11-10 — foco em ferramentas e modelos gratuitos

- Inclui agy, Kilo Code, Pi e OMP, com distinção entre harness aberto e acesso gratuito ao serviço.
- Mapeia ofertas gratuitas hospedadas no OpenCode, Kilo Gateway, OpenRouter e Antigravity.
- Reescreve os exemplos de programação e complementa o loop autor–revisor com combinações concretas dessas ferramentas.
- Corrige a distinção entre Pi básico e OMP e registra a evolução do modo Orchestrator do Kilo.
- Mantém integrações pagas como material complementar, sem exigir execução local.

## 2026-11-10 — restaura o guia único e sequencial

- Reúne o conteúdo no README, com os seis capítulos numerados do canivete: conceitos, ferramentas, recursos/APIs, exemplos, segurança e fontes.
- Converte a navegação entre oito páginas em índice e atalhos dentro do próprio README.
- Acrescenta o loop autor–revisor entre dois agentes de código, com executor de testes, contrato de feedback, limites, critérios de parada e ferramentas de orquestração.
- Explica o ciclo de agentes autônomos, harness, subagentes, orquestrador de código, delegação por função e retorno de evidências, com exemplo de integração.
- Integra parâmetros e arquiteturas recentes ao capítulo de conceitos e remove resumos duplicados do mapa.
- Mantém modelos, tecnologias e correções da revisão; disponibiliza os quatro códigos em blocos expansíveis no guia.
- Remove os oito documentos separados; preserva os scripts executáveis e seus testes como material de apoio.

## 2026-11-10 — revisão inicial e ampliação geral (estrutura substituída)

Esta entrada registra a revisão inicial. A organização em `docs/` foi substituída pela restauração do guia único, registrada acima; os tópicos foram incorporados ao README.

- Inicialmente dividiu conceitos, ferramentas, modelos/serviços, parâmetros, mapa ampliado, exemplos, segurança e fontes em oito páginas de `docs/`, com o README como entrada. Essa divisão não corresponde à estrutura atual proposta.
- Acrescenta famílias/checkpoints atuais e referências oficiais: GPT-6/6.1, Claude Fable/5.5, Gemini 3/4, Grok, Qwen3.8, DeepSeek V4.1, Kimi K3.1, MiniMax M3.1, GLM-5.3, Mistral Large 4, Gemma 4, Granite 4.1 e gpt-oss.
- Expande recuperação, embeddings/rerankers, imagem, vídeo, áudio, agentes, MCP/A2A, sandboxes, documentos, execução local e serving distribuído.
- Distingue parâmetros totais/ativos, quantização, sampling, thinking, cache, orçamento de agentes e configuração de engines; adiciona preset específico de Qwen com fonte.
- Remove promessas sem evidência de precisão, tempo de treino, latência, economia e produtividade; corrige noções de contexto infinito, determinismo e open source.
- Registra manutenção/arquivamento de AutoGen, TGI e integração legada do Helicone; substitui a classificação incorreta de Lithic.
- Mantém os 11 cenários práticos, distinguindo receitas de código executável. Substitui o caso clínico sem validação por classificação de chamados.
- Acrescenta quatro exemplos Python com pré-requisitos, configuração de modelo explícita, tratamento de falhas e testes offline.
- Acrescenta fontes, data de consulta, critérios de contribuição e limitações da verificação.

O conteúdo anterior permanece acessível no histórico Git.
