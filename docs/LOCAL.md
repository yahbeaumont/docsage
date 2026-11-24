# Rodando 100% local

Nenhuma chamada externa: LLM no Ollama, embeddings e reranker em CPU via
`sentence-transformers`, vetores no Qdrant local.

```bash
cp .env.local.example .env.local
docker compose -f docker-compose.local.yml up -d ollama qdrant
./scripts/pull_models.sh
make corpus index
docker compose -f docker-compose.local.yml up api
```

## Sem Docker

O índice também roda embutido: basta apontar `QDRANT_PATH` para um diretório.

```bash
export QDRANT_PATH=data/qdrant_local
python -m docsage.cli index build --chunks data/chunks.jsonl
```

Foi assim que a medição publicada em `docs/RESULTS.md` rodou (máquina sem Docker),
com o LLM servido por um endpoint compatível com OpenAI em vez do Ollama.

## Expectativa de qualidade

**Não medido.** A máquina usada para os números de `docs/RESULTS.md` não tinha Ollama nem GPU,
então o caminho local é suportado pelo código, mas não foi exercitado aqui. Preferi não
publicar uma tabela de "o que se perde com um 8B" sem tê-la rodado.

O que muda na prática, e é observável sem benchmark: recuperação (embeddings e reranker são os
mesmos, então recall e nDCG tendem a ficar estáveis) contra obediência de formato e calibração
de abstenção, que dependem do modelo gerador. Modelo pequeno cita menos e precisa de limiar
mais baixo, porque o score do rerank também cai.

Para medir na sua máquina: `make eval` com o mesmo golden set e comparar contra
`eval/baseline.json` com `make gate`.

## Ajustes necessários em modelo pequeno

- `ABSTAIN_THRESHOLD` para 0.20–0.24: o reranker pontua mais baixo e o limiar padrão abstém demais.
- `MAX_CONTEXT_TOKENS` para 4000: contexto longo degrada modelo de 8B.
- Rode `python -m docsage.eval.run_eval --limit 20` antes de confiar em qualquer mudança.

## vLLM

Para GPU, troque o serviço por `vllm/vllm-openai:latest` com
`--model Qwen/Qwen2.5-7B-Instruct` e `LLM_PROVIDER=vllm` (suporta `response_format`).
