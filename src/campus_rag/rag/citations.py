from __future__ import annotations

from campus_rag.types import Evidence


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

