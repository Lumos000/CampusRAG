import json

from bs4 import BeautifulSoup

from campus_rag.crawl.crawler import CampusCrawler, discover_links, page_metadata


def test_embedded_pdf_and_article_date():
    soup = BeautifulSoup('''<h1>质量报告</h1><span class="arti_update">发布时间：2025-12-29</span>
        <div pdfsrc="/_upload/report.pdf"></div><a href="/a.htm#x">正文</a>''', 'lxml')
    assert page_metadata(soup)['published_at'] == '2025-12-29'
    assert ('/_upload/report.pdf', '正文附件') in list(discover_links(soup))


class Response:
    status_code = 200
    headers = {'content-type': 'text/html', 'ETag': 'v1'}
    url = 'https://jwc.njupt.edu.cn/article.htm'

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def close(self):
        pass

    def iter_content(self, _):
        yield b'<h1>Report</h1><div pdfsrc="/report.pdf"></div>'


def test_crawl_provenance_and_repeat(tmp_path, monkeypatch):
    config = tmp_path / 'sites.json'
    config.write_text(json.dumps({'allowed_domains': ['njupt.edu.cn'],
        'seed_urls': [Response.url], 'departments': {'jwc.njupt.edu.cn': '本科生院'}}), encoding='utf-8')
    crawler = CampusCrawler(config, tmp_path / 'raw', max_pages=2)
    monkeypatch.setattr(crawler, '_can_fetch', lambda _: True)
    def get(url, headers):
        response = Response()
        if url.endswith('.pdf'):
            response.url = url
            response.headers = {'content-type': 'application/pdf'}
        return response
    monkeypatch.setattr(crawler, '_get', get)
    assert crawler.run() == 2
    records = [json.loads(line) for line in crawler.manifest_path.read_text(encoding='utf-8').splitlines()]
    assert records[1]['parent_url'] == Response.url
    assert records[1]['is_attachment']
    assert records[1]['department'] == '本科生院'
    again = CampusCrawler(config, tmp_path / 'raw')
    assert len(again.previous) == 2
    assert not again._allowed('https://evilnjupt.edu.cn/a')
    assert not again._allowed('https://njupt.edu.cn.evil.com/a')
    assert not again._allowed('https://jwc.njupt.edu.cn/login.htm')


def test_resume_does_not_repeat_completed_seed(tmp_path, monkeypatch):
    config = tmp_path / 'sites.json'
    config.write_text(json.dumps({'allowed_domains': ['njupt.edu.cn'],
        'seed_urls': [Response.url]}), encoding='utf-8')
    crawler = CampusCrawler(config, tmp_path / 'raw', max_pages=1)
    monkeypatch.setattr(crawler, '_can_fetch', lambda _: True)
    monkeypatch.setattr(crawler, '_get', lambda *_: Response())
    assert crawler.run() == 1
    again = CampusCrawler(config, tmp_path / 'raw', max_pages=1)
    monkeypatch.setattr(again, '_can_fetch', lambda _: True)
    requested = []
    def get(url, headers):
        requested.append(url)
        response = Response()
        response.url = url
        response.headers = {'content-type': 'application/pdf'}
        return response
    monkeypatch.setattr(again, '_get', get)
    assert again.run() == 1
    assert requested == ['https://jwc.njupt.edu.cn/report.pdf']
