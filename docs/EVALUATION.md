# Metodologia de avaliação

## Golden set

60 itens, três categorias: 30 factuais (resposta em 1 ou 2 chunks), 20 multi-hop (exige dois
ou mais documentos) e 10 não-respondíveis (a resposta **não** está no corpus, mas a pergunta é
do domínio). Anotados à mão sobre a documentação do FastAPI no commit fixo, com os ids de
chunk que sustentam cada resposta. Formato em `eval/golden.jsonl`:

```json
{"id": "f001", "question": "...", "answer": "...", "gold_chunk_ids": ["..."], "category": "factual", "answerable": true}
```

O carregamento valida que todo `gold_chunk_ids` existe no `data/chunks.jsonl` atual: anotação
órfã falha alto em vez de virar zero silencioso.

## Definições

- **Recall@k** — fração dos gold chunks presentes no top-k. Mede se o material necessário foi
  recuperado.
- **nDCG@k** — ganho descontado por posição, normalizado pelo ideal. Penaliza gold no fundo do
  ranking.
- **MRR** — inverso do rank do primeiro gold. Só é informativo em single-hop.
- **Context precision** — fração dos chunks de contexto que a resposta realmente usou.
- **Context recall** — fração das afirmações do gabarito cobertas pelo contexto recuperado.
- **Faithfulness** — fração das afirmações da resposta sustentadas pelo contexto (o juiz
  decompõe em claims e classifica cada uma).
- **Answer relevance** — o juiz gera perguntas a partir da resposta e mede a similaridade com a
  pergunta original.
- **Abstenção correta** — % dos itens não-respondíveis em que o sistema se absteve.
- **Abstenção indevida** — % dos itens respondíveis em que o sistema se absteve.
- **Validade de citação** — % dos marcadores que apontam para um chunk realmente no contexto.

## Abstenção calibrada

O limiar de cosseno não é escolhido no olho. `python -m docsage.calibrate --folds 5` roda
validação cruzada sobre os 10 itens não-respondíveis mais os respondíveis, escolhe o limiar
que maximiza a acurácia balanceada em cada fold (com restrição de precisão mínima) e adota a
**mediana** dos limiares dos folds, reportando o desvio entre folds. Desvio alto significa
limiar instável, e o relatório diz isso em vez de esconder.

O sinal usado é o **cosseno do melhor chunk** depois do rerank. A primeira versão usava o logit
do cross-encoder e ficou em 0,5 de acurácia: logit é degenerado para chunk fora de distribuição
(todo chunk irrelevante recebe valor muito negativo), então o máximo quase não separa.

## A/B pareado

`python -m docsage.eval.ab` roda os dois braços nas mesmas 60 perguntas:

- braço A: `RETRIEVAL_MODE=vector_only`, sem rerank, sem abstenção;
- braço B: híbrido + rerank + abstenção calibrada.

A unidade de reamostragem é o **item**, não a métrica: o bootstrap (10.000 reamostragens)
respeita o pareamento, o que controla a variância de dificuldade da pergunta, que é maior que
a do juiz. Sobre as 9 métricas aplica-se Benjamini-Hochberg; uma regressão só é declarada
quando o IC95 fica inteiro abaixo da tolerância **e** o p ajustado passa de 0,05. O relatório
também publica o **efeito mínimo detectável** (MDE a 80% de poder) — se o MDE é maior que a
tolerância, o experimento não tem resolução para o efeito que se quer detectar, e isso fica
escrito.

## Antifraude

- O juiz roda em modelo diferente do gerador.
- O cálculo é por pergunta e depois macro-média: uma pergunta com 40 claims não domina o score.
- Itens não-respondíveis impedem que "responder sempre" ganhe.
- Abster em item respondível **não** conta como faithfulness 1,0; conta como resposta faltando.
- O braço baseline roda o mesmo código, só com as variáveis de ambiente trocadas. Nada de
  baseline feito à mão.
- Falha de um item não derruba a execução: o item é marcado com `error`, sai das médias e
  aparece em `n_errors`. Um zero silencioso seria mentira.

## Reprodutibilidade

`docs/RESULTS.md` fixa: commit do corpus, sha256 do `chunks.jsonl`, versões dos modelos, limiar
calibrado, hardware, data e custo. `make gate` compara a execução atual com
`eval/baseline.json` e falha se qualquer métrica piorar mais que a tolerância configurada em
`configs/eval.yaml` (3 pontos percentuais).
