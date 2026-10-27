from __future__ import annotations

from campus_rag.types import Evidence


def select_evidence(ids,aliases):
    """Resolve only supplied IDs. This checks identity, not semantic support."""
    if not isinstance(ids,list) or any(not isinstance(i,str) or i not in aliases for i in ids):
        raise ValueError('引用ID不在本次提供的证据中')
    return list({(aliases[i].evidence_id,aliases[i].text):aliases[i] for i in ids}.values())


def build_references(evidence: list[Evidence]) -> list[str]:
    references: list[str] = []
    seen: set[tuple[int, str]] = set()
    for item in evidence:
        key = (item.doc_id, item.source_url)
        if key in seen:
            continue
        seen.add(key)
        parts = [item.title]
        if item.department:
            parts.append(item.department)
        if item.published_at:
            parts.append(item.published_at)
        if item.source_url:
            parts.append(item.source_url)
        references.append("｜".join(parts))
    return references

