"""Conservative detection of short link menus, not general non-numeric tables."""
import re


LABELS = {'图书馆简介', '入馆须知', '部门设置', '服务流程', '馆藏分布', '图书馆FAQ',
          '图书馆FAQ', '借阅规则', '电子资源', '代查代检', '科技查新', '核心期刊目录',
          '新读者向导', '读者服务', '常见问题', '开放时间', '联系我们', '首页', '工具下载'}


def navigation_values(values, link_fraction=None):
    values = [re.sub(r'\s+', '', str(v)) for v in values if str(v).strip()]
    if len(values) < 4:
        return False
    if any(re.search(r'\d', value) for value in values):
        return False
    named = sum(value in LABELS for value in values)
    if named >= 4 and named / len(values) >= 0.6:
        return True
    return link_fraction is not None and link_fraction >= 0.8 and max(map(len, values)) <= 30


def navigation_chunk(text):
    values = []
    for line in text.split('\n'):
        if '=' in line:
            values.extend(cell.split('=', 1)[1] for cell in line.split(' | ') if '=' in cell)
    return navigation_values(values)


def navigation_table_ids(connection):
    values = {}
    for table_id, raw in connection.execute('''SELECT r.table_id,c.raw_value FROM table_cells c
        JOIN table_rows r ON r.id=c.row_id JOIN tables t ON t.id=r.table_id
        JOIN documents d ON d.id=t.doc_id WHERE d.content_type IN ('html','htm')'''):
        values.setdefault(table_id, []).append(raw)
    return {table_id for table_id, cells in values.items() if navigation_values(cells)}
