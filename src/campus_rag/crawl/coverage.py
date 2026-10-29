"""Title-based coverage evidence; presence is not semantic or currency approval."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def coverage_report(clean_dir, config):
    clean_dir = Path(clean_dir)
    documents = list(read_jsonl(clean_dir / 'clean_documents.jsonl'))
    decisions = list(read_jsonl(clean_dir / 'clean_decisions.jsonl'))

    def evidence(words, host=None):
        hits = []
        for item in documents:
            source = item['document']['source']
            # Use aliases as well, so deduplication cannot hide a college source.
            sources = item.get('sources') or [dict(url=source['source_url'], title=source['title'])]
            matched = [s for s in sources if (not host or urlparse(s['url']).hostname == host)
                       and any(word in (s.get('title') or '') for word in words)]
            if matched:
                hits.append({'title': matched[0]['title'], 'url': matched[0]['url'],
                             'published_at': matched[0].get('published_at')})
        reviews = [r for r in decisions if r['decision'] == 'review'
                   and (not host or urlparse(r['url']).hostname == host)
                   and any(word in (r.get('title') or '') for word in words)]
        return {'accepted_documents': len(hits), 'review_candidates': len(reviews),
                'status': 'present_needs_verification' if hits else 'missing', 'examples': hits[:3]}

    topics = {topic: {name: evidence(words) for name, words in groups.items()}
              for topic, groups in config['coverage'].items()}
    colleges = {host: {'department': config['departments'][host],
                      'coverage': {name: evidence(words, host) for name, words in config['college_coverage'].items()}}
                for host in config['college_hosts']}
    return {'method': '标题关键词匹配；仅统计清洗接收的去重正文，不能替代内容完整性和现行有效性核验。',
            'accepted_unique_documents': len(documents),
            'decisions': dict(Counter(r['decision'] for r in decisions)),
            'missing_date_documents': sum(not r['document']['source'].get('published_at') for r in documents),
            'topics': topics, 'colleges': colleges}
