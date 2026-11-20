SYSTEM = """Você responde perguntas sobre a documentação do FastAPI usando
EXCLUSIVAMENTE o contexto fornecido.
Regras:
1. Toda frase factual termina com um ou mais marcadores de citação no formato [[chunk_id]].
2. O chunk_id é o valor hexadecimal que aparece em `chunk_id=` no cabeçalho de cada bloco.
   Nunca cite o número do bloco, o caminho do arquivo ou um id inventado.
3. Se o contexto não responder, escreva exatamente: "Não encontrado na documentação." e não cite.
4. Prefira o vocabulário da documentação a sinônimos.
5. Não repita o contexto literalmente; sintetize.
6. Inclua exemplo de código quando o contexto tiver um e ele for pertinente.
"""

USER = """Contexto:
{context}

Pergunta: {question}

Resposta com citações:"""


def build_context(candidates) -> str:
    """Cada bloco mostra apenas um identificador citável: o chunk_id."""
    blocks = []
    for cand in candidates:
        heading = " > ".join(cand.chunk.heading_path) or "(raiz)"
        blocks.append(
            f"chunk_id={cand.chunk.chunk_id} | doc={cand.chunk.path} | secao={heading}\n"
            f"{cand.chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)
