# Resultados — execução medida

Tudo aqui saiu de `eval-out/current.json`, `eval-out/baseline.json`, `eval-out/ab.json` e
`configs/calibrated.json`, produzidos por `scripts/measure.sh` nesta máquina. Nenhum número foi
estimado à mão.

| Campo | Valor |
|---|---|
| Data | 2026-11-25 |
| Corpus | `fastapi/fastapi` no commit `05e6df7e906c3decba377c37548019973f40bf15` (versão 0.165.0) |
| Documentos | 155 arquivos Markdown |
| Chunks | 2039, sha256 `1eea3e8336027b40c994c80b5edab97787315e73fbb5a091a1e4bc5cec743852` |
| Golden set | 60 itens: 30 factuais, 20 multi-hop, 10 não-respondíveis |
| Gerador | `gpt-5-mini`, temperatura 0,0, `reasoning_effort=minimal` |
| Juiz | `gemini-3-flash-preview`, temperatura 0,0 |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensões |
| Reranker | `BAAI/bge-reranker-v2-m3`, `max_length=192` |
| Limiar de abstenção | 0,4705 (mediana de 5 folds, σ = 0,0291) |
| Hardware | 6 vCPU, 20 GB, sem GPU |
| Custo (eval + baseline) | US$ 0,0782 |

## Os dois braços

| Métrica | Baseline (vetorial puro) | docsage (híbrido + rerank) |
|---|---|---|
| recall@10 | **0,530** | 0,510 |
| nDCG@10 | 0,322 | **0,343** |
| MRR | 0,258 | **0,302** |
| context precision | 0,358 | 0,355 |
| context recall | **0,715** | 0,672 |
| faithfulness | **0,755** | 0,732 |
| answer relevance | **0,483** | 0,454 |
| validade de citação | 1,000 | 1,000 |
| acurácia de abstenção | 0,000 | **0,300** |
| abstenção indevida | 0,000 | 0,060 |
| latência p50 | 5,3 s | **4,0 s** |
| latência p95 | **10,8 s** | 63,4 s |
| custo por pergunta | US$ 0,00100 | **US$ 0,00060** |
| erros de execução | 0 | 0 |

A leitura honesta, que não é a que o projeto gostaria de ter:
- **As duas execuções descritivas não concordam com o A/B.** Rodando os braços separadamente,
  nDCG sobe 0,322 → 0,343 e MRR 0,258 → 0,302, mas o A/B pareado (mesmo processo, mesma ordem de
  perguntas) dá Δ nDCG −3,6% e Δ MRR −2,1%, com IC cruzando zero. O pareado é o experimento
  correto, porque cada par vê a mesma pergunta; a diferença entre os dois runs isolados é
  ruído de execução. Conclusão: **não há evidência de ganho de recuperação**.
- **Recall@10 cai nos dois desenhos** (0,530 → 0,510 no descritivo, −8,0 p.p. no pareado). Com a
  busca densa já trazendo 50 candidatos e o braço devolvendo 8, a fusão com BM25 e o corte do
  rerank às vezes descartam um chunk que a busca densa tinha achado. O rerank reordena o que
  fica, não aumenta a cobertura.
- **Faithfulness e relevância não melhoram.** 0,755 → 0,732 e 0,483 → 0,454 no descritivo;
  Δ faithfulness +1,4% no pareado, IC [−0,094, 0,119]. Contexto mais curto barateia a consulta
  e reduz a chance de o modelo encontrar a frase que sustenta a resposta.
- **Custo por pergunta cai 40%.** US$ 0,00100 → 0,00060: contexto menor, menos tokens de
  entrada. É o ganho mais claro do rerank aqui, e ele é real porque o número de tokens de
  entrada não depende de ruído de amostragem.
- **A cauda de latência piorou muito.** p50 melhorou (5,3 s → 4,0 s) mas p95 foi de 10,8 s para
  63,4 s. O cross-encoder é serializado de propósito (paralelizar 6 reranks em 6 vCPU dava 6×
  menos vazão por item), então quem chega na fila espera. Em produção isso pede GPU ou um
  reranker menor, não um ajuste de timeout.
## Abstenção: o ponto fraco

O limiar calibrado (0,4705) abstém em 3 dos 10 itens sem resposta e em 3 dos 50 respondíveis.
A precisão é 0,5 (3 acertos em 6 abstenções), o recall é 0,3.

O diagnóstico está nos dados: os scores dos 10 itens não-respondíveis vão de 0,409 a 0,688, e
vários respondíveis ficam na mesma faixa. O cosseno do melhor chunk **separa mal** esse corpus,
e com 10 itens negativos o limiar é instável (σ = 0,029 entre folds, com folds escolhendo entre
0,4347 e 0,5135). A calibração está correta; o sinal é que é fraco. A correção de verdade é
aumentar o número de itens não-respondíveis e testar um classificador de abstenção com features
de retrieval, não um limiar único.

## A/B pareado

50 pares respondíveis, 10 não-respondíveis, 10.000 reamostragens de bootstrap, correção de
Benjamini-Hochberg sobre as 9 métricas, α = 0,05.

| Métrica | Δ (docsage − baseline) | IC95 | p ajustado | Decisão |
|---|---|---|---|---|
| recall@10 | -8,0% | [-0,180, 0,010] | 0,9681 | inconclusivo |
| nDCG@10 | -3,6% | [-0,092, 0,016] | 0,9681 | inconclusivo |
| MRR | -2,1% | [-0,068, 0,025] | 0,9681 | inconclusivo |
| faithfulness | 1,4% | [-0,094, 0,119] | 0,9681 | inconclusivo |
| answer relevance | -0,7% | [-0,047, 0,034] | 0,9681 | inconclusivo |
| abstenção | 30,0% | [0,000, 0,600] | 0,2583 | passa |

Veredito geral: **inconclusivo**. Efeito mínimo detectável em recall@10: 0,019, abaixo da
tolerância de 0,030 do gate. Em abstenção o MDE é 0,128, acima da tolerância: com 10 itens
negativos o experimento não distingue nem um efeito do tamanho do que o limiar produz, e por isso
a decisão de abstenção aparece como "passa" apenas por não haver regressão detectável, não por
haver ganho demonstrado.

## Por categoria

| Categoria | Baseline recall / faithfulness | docsage recall / faithfulness |
|---|---|---|
| Factual (n=30) | 0,633 / 0,778 | 0,600 / 0,792 |
| Multi-hop (n=20) | 0,375 / 0,721 | 0,375 / 0,641 |
| Não-respondível (n=10) | — / 0,100 | — / 0,400 |

Multi-hop não melhorou nada em recall: 0,375 nos dois braços. É o caso em que a fusão deveria
ajudar mais, e não ajudou, porque a pergunta multi-hop precisa de dois chunks e o corte em 8
depois do rerank mantém o melhor deles e perde o segundo.

## Reproduzir

```bash
make install
make corpus index          # baixa o commit fixo e indexa 2039 chunks
make calibrate             # 5 folds, grava configs/calibrated.env
make eval                  # braço híbrido
make eval-baseline         # braço vetorial puro
make ab                    # A/B pareado com bootstrap
make plots                 # regenera docs/img dos JSONs
```

`scripts/measure.sh` roda a cadeia inteira em sequência. A calibração é a etapa mais cara em
CPU (60 retrievals com rerank, ~17 s por item em 6 vCPU); eval e baseline são dominados pelas
chamadas de LLM.

## Limitações

- **Uma execução por braço.** Temperatura 0, mas o modelo não é determinístico; a incerteza
  reportada é a do bootstrap pareado sobre 60 itens, não a variância entre execuções.
- **Golden set pequeno e de um domínio.** 60 perguntas sobre a documentação do FastAPI. Os
  números não se transferem para código, contratos ou artigos.
- **Juiz como medida.** Faithfulness e relevância vêm de um LLM com rubrica; não há calibração
  contra anotação humana.
- **Abstenção binária**, com o sinal fraco descrito acima.
- **`recall@10` com 8 chunks devolvidos**: as duas pontas usam `TOP_K_RERANK=8`, então a métrica
  mede a cobertura desse corte, não de 10.
