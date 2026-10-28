"""Query-independent, conservative table normalization. Never infer missing values."""
import json
import re
from decimal import Decimal

from campus_rag.tools.temporal import question_periods


VERSION = 1
UNITS = {'㎡':'平方米', 'm²':'平方米', '%':'ratio', '％':'ratio',
         '万元':'万元', '亿元':'亿元', '元':'元', '万人':'万人', '千人':'千人',
         '人':'人', '台':'台', '本':'本', '项':'项', '个':'个', '年':'年',
         '平方米':'平方米', '小时':'小时', '天':'天', '次':'次', '件':'件'}


def compact(value):
    return re.sub(r'\s+', '', str(value)).replace('（','(').replace('）',')').replace('％','%').lower()


def field_label(header):
    """Use the leaf of a flattened multi-level header, not its table title."""
    return re.split(r'\s+/\s+',str(header))[-1]


def year_header(header):
    label = re.sub(r'\([^)]*\)','',compact(field_label(header)))
    return bool(re.search(r'(?:年份|年度|学年)$',label))


def header_unit(header):
    text = compact(header)
    for level in reversed(re.split(r'\s+/\s+',str(header))):
        units = {UNITS[u] for u in re.findall(r'\(([^()]+)\)',compact(level)) if u in UNITS}
        if len(units)>1:
            return ''
        if units:
            return units.pop()
    if '人数' in text:
        return '人'
    if text.endswith('百分比'):
        return 'ratio'
    return ''


def quantity(raw, default_unit=''):
    """Only whole-cell exact quantities. Ranges, approximations and prose stay null."""
    text = compact(raw).replace(',','')
    match = re.fullmatch(r'([-+]?\d+(?:\.\d+)?)([^\d]*)',text)
    if not match:
        return None, ''
    suffix = match.group(2)
    if suffix and suffix not in UNITS:
        return None, ''
    unit = UNITS.get(suffix,default_unit)
    value = Decimal(match.group(1))
    if suffix in {'%','％'} or (not suffix and unit == 'ratio'):
        value /= 100
    scale = {'万元':('元',10000),'亿元':('元',100000000),'万人':('人',10000),'千人':('人',1000)}
    if unit in scale:
        unit, multiplier = scale[unit]
        value *= multiplier
    return value, unit


def normalize_table(connection, table_id):
    table = connection.execute('SELECT t.*,d.title,d.published_at FROM tables t JOIN documents d ON d.id=t.doc_id WHERE t.id=?',(table_id,)).fetchone()
    rows = connection.execute('SELECT * FROM table_rows WHERE table_id=? ORDER BY row_index',(table_id,)).fetchall()
    columns = connection.execute('SELECT * FROM table_columns WHERE table_id=? ORDER BY column_index',(table_id,)).fetchall()
    row_metadata = {}
    for row in rows:
        values = [c[0] for c in connection.execute('SELECT raw_value FROM table_cells WHERE row_id=? ORDER BY column_id',(row['id'],))]
        labels = [compact(v) for v in values]
        total = any(re.fullmatch(r'合计|总计|小计|总额|总数|总面积',v) for v in labels)
        year_columns = {c['id'] for c in columns if year_header(c['original_name'])}
        year_values = [c['raw_value'] for c in connection.execute('SELECT column_id,raw_value FROM table_cells WHERE row_id=?',(row['id'],)) if c['column_id'] in year_columns]
        periods = list(dict.fromkeys(p for v in year_values for p in question_periods(v)))
        row_metadata[str(row['id'])] = {'is_total':total,'periods':periods}
    for col in columns:
        cells = connection.execute('SELECT c.* FROM table_cells c JOIN table_rows r ON r.id=c.row_id WHERE r.table_id=? AND c.column_id=?',(table_id,col['id'])).fetchall()
        default = header_unit(col['original_name']) or table['unit']
        parsed = [quantity(c['raw_value'],default) for c in cells]
        nonempty = sum(bool(c['raw_value'].strip()) for c in cells)
        numeric = sum(v is not None for v,_ in parsed)
        is_year = year_header(col['original_name'])
        dtype = 'year' if is_year else 'number' if numeric and numeric >= max(1,nonempty/2) else 'text'
        units = {u for v,u in parsed if v is not None and u}
        unit = next(iter(units)) if len(units)==1 else default if not units else ''
        connection.execute('UPDATE table_columns SET normalized_name=?,data_type=?,unit=? WHERE id=?',
            (compact(col['original_name']),dtype,unit if dtype=='number' else '',col['id']))
        for cell,(value,cell_unit) in zip(cells,parsed):
            connection.execute('UPDATE table_cells SET normalized_value=?,numeric_value=?,unit=? WHERE id=?',
                (compact(cell['raw_value']),str(value) if value is not None and dtype=='number' else None,
                 cell_unit if value is not None and dtype=='number' else '',cell['id']))
    periods = question_periods(table['title']+' '+table['name'])
    if not periods and table['published_at']:
        periods = [table['published_at'][:4]]
    connection.execute('INSERT OR REPLACE INTO table_metadata VALUES (?,?,?,?)',
        (table_id,VERSION,json.dumps(periods),json.dumps(row_metadata)))


def normalize_existing_tables(connection):
    for row in connection.execute('SELECT id FROM tables ORDER BY id').fetchall():
        normalize_table(connection,row['id'])
