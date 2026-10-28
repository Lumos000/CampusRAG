from campus_rag.ingest.chunker import CharacterTokenizer, TokenChunker


def test_chunker_never_exceeds_hard_limit():
    chunker = TokenChunker(
        CharacterTokenizer.encode, CharacterTokenizer.decode,
        target_min=20, target_max=30, hard_max=35, overlap=5,
    )
    sections = [("标题", "第一段。" * 30 + "\n\n" + "第二段。" * 20)]
    chunks = chunker.split_sections(sections)
    assert len(chunks) > 2
    assert all(count <= 35 for _, _, count in chunks)
    assert all(heading == "标题" for heading, _, _ in chunks)

