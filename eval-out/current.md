# Relatório — docsage

Execução: `2026-11-25T15:37:33.295876+00:00`  
Itens: 60

| Métrica | Valor |
|---|---|
| recall_at_k | 0.5100 |
| ndcg_at_k | 0.3435 |
| mrr | 0.3024 |
| context_precision | 0.3550 |
| context_recall | 0.6717 |
| faithfulness | 0.7316 |
| answer_relevance | 0.4541 |
| citation_validity | 1.0000 |
| abstention_accuracy | 0.3000 |

## Piores casos (faithfulness < 0.6)

- `f006` (factual) faith=0.57
  > Use Path com o parâmetro de validação ge=1, por exemplo declarando o parâmetro como int e usando Path(ge=1) para exigir que seja maior ou igual a 1 [[03faa935f4e3c1f2]][8196a0571a1d8bfb]]. Exemplo:
from fastapi import Path @app.get("/items/
- `f017` (factual) faith=0.00
  > Não encontrado na documentação.
- `f019` (factual) faith=0.57
  > Use um dependency de tipo OAuth2 que extraia o token do cabeçalho Authorization com o esquema "Bearer", por exemplo usando OAuth2PasswordBearer como dependency na rota/na dependência que precisa do token [[b98fa983a91274b5]] [[ff95e994b61cd
- `f025` (factual) faith=0.00
- `f030` (factual) faith=0.33
  > Você usa o Field() do Pydantic para declarar validação e metadados dentro do modelo Pydantic [[1fe5beb8bf84abec]]. Exemplo (síntese do uso mencionado): from pydantic import BaseModel, Field class Item(BaseModel): name: str = Field(..., titl
- `m005` (multi_hop) faith=0.00
  > Não encontrado na documentação.
- `m007` (multi_hop) faith=0.00
  > Não encontrado na documentação.
- `m009` (multi_hop) faith=0.00
  > Não encontrado na documentação.
- `m011` (multi_hop) faith=0.00
- `m017` (multi_hop) faith=0.00
  > Não encontrado na documentação.
