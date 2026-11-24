# Observabilidade

```bash
docker compose -f docker-compose.observability.yml up -d
# UI em http://localhost:3000 — crie projeto e copie as chaves
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
pip install -e ".[tracing]"
```

Sem as variáveis, o tracer é no-op e nada quebra.

## O que é rastreado

| Span | Atributos |
|---|---|
| `retrieve` | question, mode, top_score, n |
| `generate` | abstained, output, citation_validity |
| `judge.*` (só em eval) | metric, value |

Scores enviados por query: `citation_validity`, `unsupported_ratio`. Por item de eval:
`faithfulness`, `answer_relevance`, `context_precision`.

## Privacidade

input/output completos vão para o Langfuse. Para corpus sensível, defina
`LANGFUSE_MASK` e envolva `span.update` com a função de máscara — o hook está em
`telemetry._Span.update`.

## Nota de medição

O Langfuse v3 exige Postgres, ClickHouse, Redis e MinIO; a máquina usada para medir
`docs/RESULTS.md` não tinha Docker, então o tracing rodou no caminho no-op (`NullTracer`)
e essa parte não foi exercitada contra um servidor real. A fiação está coberta por teste
unitário (`tests/test_telemetry.py`), não por um trace observado.
