from docsage.models import Chunk
from docsage.retrieve.hybrid import Candidate, _rrf


def _c(cid: str) -> Candidate:
    return Candidate(Chunk(chunk_id=cid, doc_id="d", path="p", heading_path=(), text="t",
                           n_tokens=1, ordinal=0), 1.0, "dense")


def test_rrf_rewards_agreement():
    a = [_c("x"), _c("y")]
    b = [_c("y"), _c("z")]
    fused = _rrf([a, b], k=60)
    assert fused[0].chunk.chunk_id == "y"
    assert len(fused) == 3


def test_rrf_dedupes():
    fused = _rrf([[_c("x")], [_c("x")]], k=60)
    assert len(fused) == 1
