# Relatório — baseline

Execução: `2026-11-25T15:43:43.073345+00:00`  
Itens: 60

| Métrica | Valor |
|---|---|
| recall_at_k | 0.5300 |
| ndcg_at_k | 0.3223 |
| mrr | 0.2582 |
| context_precision | 0.3575 |
| context_recall | 0.7150 |
| faithfulness | 0.7553 |
| answer_relevance | 0.4831 |
| citation_validity | 1.0000 |
| abstention_accuracy | 0.0000 |

## Piores casos (faithfulness < 0.6)

- `f005` (factual) faith=0.44
  > Você pode usar Query com min_length dentro de Annotated no parâmetro, por exemplo: q: Annotated[str | None, Query(min_length=3)] = None [[482ba6b5b007ce00]] [[1184dbbccb78eb53]]. Exemplo (sintetizado do contexto):
from typing import Annotat
- `f007` (factual) faith=0.33
  > Você passa o parâmetro tags com uma lista de strings ao decorador da path operation, por exemplo tags=["meu_tag"] para que elas sejam adicionadas ao esquema OpenAPI e usadas pela documentação automática [[4fe64c9e2ed4c472]]. Exemplo relevan
- `f017` (factual) faith=0.00
  > Não encontrado na documentação.
- `f030` (factual) faith=0.25
  > Use Pydantic's Field() dentro do modelo para declarar validação e metadata nos atributos do modelo [[1fe5beb8bf84abec]]. Exemplo:
from pydantic import BaseModel, Field class Item(BaseModel): name: str = Field(..., min_length=3, max_length=5
- `m005` (multi_hop) faith=0.00
  > Não encontrado na documentação.
- `m006` (multi_hop) faith=0.00
  > Não encontrado na documentação.
- `m007` (multi_hop) faith=0.00
  > Não encontrado na documentação.
- `m020` (multi_hop) faith=0.00
  > Não encontrado na documentação.
