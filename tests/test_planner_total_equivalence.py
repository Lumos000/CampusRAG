import importlib.util
import json
from pathlib import Path

from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import ParsedTable,SourceDocument

spec=importlib.util.spec_from_file_location('canonical_audit',Path(__file__).resolve().parents[1]/'experiments/planner_lora_v9/canonical_score.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_only_verified_complete_detail_sum_is_equivalent_to_total(tmp_path):
    with Repository(tmp_path/'db.sqlite') as repo:
        repo.initialize();did=repo.add_document(SourceDocument('课程规定','https://example.edu/course'))
        table=ParsedTable('明细',['课程名称','学分'],[['甲','2'],['乙','3'],['合计','5']])
        tid=repo.add_table(did,table,[[normalize_cell(v) for v in row] for row in table.rows])
        cols={r['original_name']:r['id'] for r in repo.connection.execute('SELECT * FROM table_columns WHERE table_id=?',(tid,))}
        engine=StructuredQueryEngine(repo.connection)
        def run(step,question='学分合计多少学分？'):
            result=engine.execute_steps(parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'result','field_id':cols['学分'],**step}]})),[tid],question)
            assert result.ok,result.error
            return audit.sources(engine,result.details,[])
        full=run({'op':'sum'});total=run({'op':'lookup','scope':'total'})
        assert full==total and len(full)==2
        partial=run({'op':'lookup','filters':[{'field_id':cols['课程名称'],'op':'eq','value':'甲'}]},'甲有多少学分？')
        assert partial!=total
        rid=repo.connection.execute('SELECT id FROM table_rows WHERE table_id=? AND row_label=?',(tid,'合计')).fetchone()[0]
        repo.connection.execute('UPDATE table_cells SET raw_value=?,normalized_value=?,numeric_value=? WHERE row_id=? AND column_id=?',('6','6','6',rid,cols['学分']))
        assert run({'op':'lookup','scope':'total'})!=full
