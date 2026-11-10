# Segurança e governança

[← Voltar ao canivete](../README.md) · Revisão: 10/11/2026

Controles devem existir na aplicação e na infraestrutura. Instruir um modelo a “não revelar dados” não substitui autenticação, autorização ou isolamento.

| Risco/tema | Controle concreto | Como verificar |
| --- | --- | --- |
| Prompt injection em documentos e páginas | Tratar conteúdo como dado não confiável; separar instruções; limitar ferramentas e dados acessíveis | Testar documentos maliciosos e tentativas de desviar a tarefa |
| Vazamento entre usuários no RAG | Aplicar ACL por identidade na recuperação; isolar memória e cache conforme arquitetura | Usuário sem permissão não recupera conteúdo restrito nem por pergunta indireta |
| Ações indevidas de agentes | Allowlist de ferramentas, validação de argumentos, menor privilégio e aprovação conforme impacto | Revisar o payload exato e registrar identidade/decisão |
| Execução de código e browser use | Sandbox, limites de rede/arquivos/tempo e controle de sessão | Testar tentativas de sair do escopo e limpar sessões |
| Servidor MCP malicioso | Verificar origem, código, permissões e dados expostos; autenticar conexões | Revisar ferramentas e mudanças de descrições; testar autorização por recurso |
| Segredos em código/logs | Secret manager/ambiente local, rotação, mascaramento e revisão de diffs | Secret scanning e teste de traces sem credenciais |
| PII e documentos sensíveis | Minimização, classificação, mascaramento e retenção definida | Conferir dados de entrada, saída, backup e observabilidade |
| Saída estruturada incorreta | Schema + regras de negócio + origem dos campos | Testar ausências, valores impossíveis e cálculos divergentes |
| Ferramentas duplicadas por retry | Idempotência, chave por operação e reconciliação | Simular timeout após o efeito externo e repetir a chamada |
| Dependências e pesos adulterados | Fixar revisões, verificar origem e revisar código de carregamento | Evitar confiança automática em `trust_remote_code`; registrar hashes/revisões |
| Indisponibilidade/custo | Timeout, orçamento, rate limits, fallback e circuit breaker conforme necessidade | Testar falha, esgotamento e loop de agente |
| Viés e avaliação | Casos representativos, métricas por grupo e revisão humana | Inspecionar erros por idioma, contexto e público |
| Backup e recuperação | Versionamento, restauração testada e política de exclusão | Recuperar estado/índice e verificar consistência de permissões |

Ferramentas possíveis: [Presidio](https://microsoft.github.io/presidio/) para detecção de PII, [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/) e [Guardrails AI](https://github.com/guardrails-ai/guardrails) para controles de aplicação, [Promptfoo](https://www.promptfoo.dev/docs/intro/) para testes e [OpenTelemetry](https://opentelemetry.io/docs/) para instrumentação. Nenhuma ferramenta garante segurança completa ou conformidade por si só.

**Explicabilidade:** SHAP e LIME têm aplicações específicas em interpretação de modelos, mas não explicam automaticamente decisões de um agente generativo. Atenção visualizada e texto de raciocínio não devem ser tratados como prova causal de uma resposta.

**Governança:** registre finalidade, responsáveis, origem/licença dos dados e modelos, versões, critérios de avaliação, retenção e resposta a incidentes. Obrigações de LGPD e outras normas dependem do caso e precisam de análise própria; este documento não afirma conformidade jurídica de uma arquitetura.

Referências primárias: [OWASP GenAI Security Project](https://genai.owasp.org/), [segurança MCP](https://modelcontextprotocol.io/specification/latest/basic/security_best_practices), [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework), [ANPD](https://www.gov.br/anpd/pt-br).
