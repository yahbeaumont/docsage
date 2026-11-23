import pytest

from docsage.models import Chunk


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id=f"c{i:015x}", doc_id="d1", path=f"docs/x{i}.md",
              heading_path=("H",), text=f"conteudo {i} sobre Depends", n_tokens=10, ordinal=i)
        for i in range(5)
    ]
