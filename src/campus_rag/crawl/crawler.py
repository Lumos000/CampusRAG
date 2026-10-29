from __future__ import annotations

import hashlib
import json
import time
import re
from collections import Counter
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


class CampusCrawler:
    def __init__(self, config_path: str | Path, output_dir: str | Path, max_pages: int | None = None, max_depth: int | None = None, refresh: bool = False):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_domains = tuple(self.config["allowed_domains"])
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "CampusRAG-course-project/0.1 (public-data crawler)"
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self._robots: dict[str, RobotFileParser] = {}
        self.max_pages = max_pages if max_pages is not None else self.config.get('max_pages', 200)
        self.max_depth = max_depth if max_depth is not None else self.config.get('max_depth', 3)
        self.previous = {}
        self.refresh = refresh
        self.state_path = self.output_dir / 'crawl_state.json'
        self.signature = hashlib.sha256(json.dumps([self.config, self.max_depth], sort_keys=True).encode()).hexdigest()
        if self.manifest_path.exists():
            for line in self.manifest_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get('status') == 'ok':
                        self.previous[record['url']] = record

    def _get(self, url, headers=None):
        for attempt in range(self.config.get('retries', 2) + 1):
            try:
                for _ in range(6):
                    time.sleep(float(self.config.get('request_interval_seconds', 0.8)))
                    response = self.session.get(url, headers=headers or {}, stream=True,
                        timeout=self.config.get('timeout_seconds', 20), allow_redirects=False)
                    if response.status_code not in {301, 302, 303, 307, 308}:
                        response.raise_for_status()
                        return response
                    target = self._canonical(urljoin(url, response.headers.get('Location', '')))
                    response.close()
                    if not self._allowed(target) or not self._can_fetch(target):
                        raise ValueError('redirect_not_allowed')
                    url = target
                raise ValueError('too_many_redirects')
            except requests.RequestException:
                if attempt == self.config.get('retries', 2):
                    raise
                time.sleep(2 ** attempt)

    def run(self) -> int:
        queue = deque((url, 0, {}) for url in self.config['seed_urls'])
        queued = set(self.config['seed_urls'])
        seen: set[str] = set()
        if not self.refresh and self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding='utf-8'))
            if state.get('signature') == self.signature:
                seen = set(state['seen'])
                queue = deque((url, depth, parent) for url, depth, parent in state['queue'])
                queued = seen | {item[0] for item in queue}
                print(f'[RESUME] visited={len(seen)} pending={len(queue)}', flush=True)
        starting_seen = len(seen)
        failures = []
        downloaded = 0
        counts, departments = Counter(), Counter()
        while queue and len(seen) - starting_seen < self.max_pages:
            queue = deque(sorted(queue, key=lambda item: (
                -10000 if item[1] == 0 else
                -(100 if Path(urlparse(item[0]).path).suffix.lower() in {'.pdf', '.docx', '.xlsx'} else 0)
                - sum(5 for word in self.config.get('priority_keywords', [])
                      if word in item[2].get('link_text', '') + item[2].get('title', ''))
            )))
            raw_url, depth, parent = queue.popleft()
            url = self._canonical(raw_url)
            if not url or url in seen or not self._allowed(url):
                continue
            seen.add(url)
            if not self._can_fetch(url):
                self._write_manifest({'url': url, 'status': 'robots_denied'})
                failures.append((url, depth, parent))
                counts['robots_denied'] += 1
                print(f'[SKIP robots] {url}', flush=True)
                continue
            old = self.previous.get(url, {})
            response = None
            headers = {}
            if Path(old.get('local_path', '__missing__')).is_file():
                if old.get('etag'):
                    headers['If-None-Match'] = old['etag']
                if old.get('last_modified'):
                    headers['If-Modified-Since'] = old['last_modified']
            try:
                response = self._get(url, headers)
                if response.status_code == 304:
                    content = Path(old['local_path']).read_bytes()
                else:
                    parts, size = [], 0
                    for part in response.iter_content(65536):
                        size += len(part)
                        if size > self.config.get('max_file_mb', 40) * 1024 * 1024:
                            raise ValueError('file_too_large')
                        parts.append(part)
                    content = b''.join(parts)
            except (requests.RequestException, ValueError, OSError) as exc:
                if response is not None:
                    response.close()
                self._write_manifest({'url': url, 'status': 'error', 'error_type': type(exc).__name__})
                failures.append((url, depth, parent))
                counts['error'] += 1
                print(f'[ERROR {type(exc).__name__}] {url}', flush=True)
                continue
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if response.status_code == 304:
                content_type = old['content_type']
            suffix = self._suffix(url, content_type)
            if suffix == '.bin':
                response.close()
                counts['unsupported'] += 1
                continue
            digest = hashlib.sha256(content).hexdigest()
            local_path = self.output_dir / f"{digest}{suffix}"
            if not local_path.exists():
                local_path.write_bytes(content)
            record = {
                "url": url,
                "final_url": response.url,
                "local_path": str(local_path),
                "content_type": content_type,
                "sha256": digest,
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                "status": "ok",
                'etag': response.headers.get('ETag', old.get('etag', '')),
                'last_modified': response.headers.get('Last-Modified', old.get('last_modified', '')),
                'title': parent.get('link_text') or Path(urlparse(url).path).name,
                'parent_url': parent.get('final_url', ''),
                'parent_title': parent.get('title', ''),
                'department': self._department(url),
                'published_at': None, 'date_source': 'unknown', 'depth': depth,
                'unchanged': old.get('sha256') == digest,
            }
            response.close()
            if suffix in {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.xlsm', '.txt'}:
                if record['title'] == '正文附件':
                    record['title'] = parent.get('title') or record['title']
                record.update(is_attachment=True, published_at=parent.get('published_at'),
                    date_source='parent_page' if parent.get('published_at') else 'unknown')
                counts['attachments'] += 1
            if content_type in {"text/html", "application/xhtml+xml"}:
                soup = BeautifulSoup(content, "lxml")
                record.update(page_metadata(soup))
                if is_error_page(record['title']):
                    record['status'] = 'invalid_content'
                    self._write_manifest(record)
                    counts['invalid_content'] += 1
                    continue
                links = []
                for href, label in discover_links(soup):
                    link = self._canonical(urljoin(response.url, href.strip()))
                    if link and self._allowed(link) and link not in queued and depth < self.max_depth and self._follow_link(link, label, record['title']):
                        priority = sum(1 for word in self.config.get('priority_keywords', []) if word in label)
                        if Path(urlparse(link).path).suffix.lower() in {'.pdf', '.docx', '.xlsx'}:
                            priority += 10
                        links.append((priority, link, dict(record, link_text=label)))
                        queued.add(link)
                # Follow useful links first without starving the configured seed sites.
                ordered = sorted(links, key=lambda item: -item[0])
                insert_at = sum(item[1] == 0 for item in queue)
                for priority, link, metadata in ordered:
                    if priority > 0:
                        queue.insert(insert_at, (link, depth + 1, metadata))
                        insert_at += 1
                    else:
                        queue.append((link, depth + 1, metadata))
            self._write_manifest(record)
            self.previous[url] = record
            downloaded += 1
            counts['ok'] += 1
            counts['unchanged'] += record['unchanged']
            departments[record['department']] += 1
            print(f'[{len(seen) - starting_seen}/{self.max_pages} OK] {record["title"]} {url}', flush=True)
            self._checkpoint(queue, seen)
        for url, depth, parent in failures:
            seen.discard(url)
            queue.append((url, depth, parent))
        self._checkpoint(queue, seen)
        report = {'attempted': sum(counts.get(key, 0) for key in ['ok','error','robots_denied','unsupported','invalid_content']),
                  'visited_total': len(seen), 'results': dict(counts), 'departments': dict(departments),
                  'pending_urls': len(queue), 'finished_at': datetime.now(timezone.utc).isoformat()}
        (self.output_dir / 'crawl_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        self.session.close()
        return downloaded

    def _checkpoint(self, queue, seen):
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'signature': self.signature, 'seen': sorted(seen),
            'queue': list(queue)}, ensure_ascii=False), encoding='utf-8')
        temporary.replace(self.state_path)

    def _department(self, url):
        host = urlparse(url).hostname or ''
        for domain, name in sorted(self.config.get('departments', {}).items(), key=lambda item: -len(item[0])):
            if host == domain or host.endswith('.' + domain):
                return name
        return '南京邮电大学'

    def _follow_link(self, url, label, parent_title=''):
        keywords = self.config.get('follow_keywords', [])
        if not keywords:
            return True
        # Unlabelled attachments need a relevant parent, not just an official host.
        if Path(urlparse(url).path).suffix.lower() in {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.xlsm', '.txt'}:
            return any(word in label or word in parent_title for word in keywords)
        if label.strip() in {'下一页', '下页', '上一页', '上页', '首页', '尾页'}:
            return True
        return any(word in label for word in keywords)

    def _allowed(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        parsed = urlparse(url)
        return (parsed.scheme in {'http', 'https'} and not parsed.username and not parsed.password
            and parsed.port in {None, 80, 443}
            and any(host == domain or host.endswith('.' + domain) for domain in self.allowed_domains)
            and Path(parsed.path).suffix.lower() not in {'.jpg', '.png', '.gif', '.svg', '.css', '.js', '.mp4', '.zip', '.rar'}
            and not re.search(r'/(login|logout|search)(/|\.|$)', parsed.path, re.I))

    def _can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            robots = RobotFileParser(urljoin(origin, "/robots.txt"))
            try:
                with self.session.get(robots.url, timeout=self.config.get('timeout_seconds', 20), allow_redirects=False) as response:
                    if response.status_code == 200:
                        robots.parse(response.text.splitlines())
                    elif response.status_code in {404, 410}:
                        robots.parse([])
                    else:
                        robots.disallow_all = True
            except requests.RequestException:
                robots.disallow_all = True
            self._robots[origin] = robots
        return self._robots[origin].can_fetch(self.session.headers["User-Agent"], url)

    @staticmethod
    def _canonical(url: str) -> str:
        clean, _ = urldefrag(url.strip())
        parsed = urlparse(clean)
        return clean if parsed.scheme in {"http", "https"} else ""

    @staticmethod
    def _suffix(url: str, content_type: str) -> str:
        mime_map = {'application/pdf': '.pdf', 'application/msword': '.doc',
                    'application/vnd.ms-excel': '.xls',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx'}
        if content_type in mime_map:
            return mime_map[content_type]
        if content_type in {'text/html', 'application/xhtml+xml'}:
            return '.html'
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix in {".html", ".htm", ".pdf", ".docx", ".xlsx", ".xlsm", ".txt"}:
            return suffix
        return '.bin'

    def _write_manifest(self, record: dict) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def page_metadata(soup):
    node = soup.select_one('.Article_Title, .arti_title, h1') or soup.title
    title = node.get_text(' ', strip=True) if node else ''
    date = None
    for selector in ['meta[name="Article_PublishDate"]', 'meta[property="article:published_time"]',
                     '.Article_PublishDate', '.arti_update', '.publish-date', '.arti_metas', '.info', 'time']:
        node = soup.select_one(selector)
        if node:
            candidate = node.get('content') or node.get('datetime') or node.get_text(' ', strip=True)
            match = re.search(r'(20\d{2})[-/年.](\d{1,2})[-/月.](\d{1,2})', candidate)
            if match:
                try:
                    date = datetime(*(int(part) for part in match.groups())).date().isoformat()
                except ValueError:
                    continue
                break
    return {'title': title, 'published_at': date, 'date_source': 'page_metadata' if date else 'unknown'}


def discover_links(soup):
    for node in soup.select('a[href]'):
        yield node['href'], (node.get('title') or node.get_text(' ', strip=True))[:300]
    # WebPlus embeds downloadable PDFs in custom attributes rather than anchors.
    for node in soup.select('[pdfsrc], iframe[src], embed[src], object[data]'):
        href = node.get('pdfsrc') or node.get('src') or node.get('data')
        if href and Path(urlparse(href).path).suffix.lower() in {'.pdf', '.docx', '.xlsx'}:
            yield href, node.get('title') or '正文附件'


def is_error_page(title):
    return title.strip().lower() in {'提示信息', '错误提示', '页面不存在', '404 not found', 'access denied', '403 forbidden'}
