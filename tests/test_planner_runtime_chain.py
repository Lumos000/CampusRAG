from contextlib import nullcontext
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.numeric import NumericEngine
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence,ParsedTable,SourceDocument

directory=Path(__file__).resolve().parents[1]/'experiments/planner_lora'
sys.path.insert(0,str(directory))
spec=importlib.util.spec_from_file_location('runtime_chain_audit',directory/'runtime_eval.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)


def test_actual_two_error_repair_chain_is_scored_from_model_outputs(tmp_path):
    with Repository(tmp_path/'db.sqlite') as repo:
        repo.initialize();did=repo.add_document(SourceDocument('课程规定','https://example.edu/courses'))
        table=ParsedTable('明细',['课程名称','学分'],[['甲','2'],['乙','3'],['合计','5']])
        tid=repo.add_table(did,table,[[normalize_cell(v) for v in row] for row in table.rows])
        field=repo.connection.execute('SELECT id FROM table_columns WHERE table_id=? AND original_name=?',(tid,'学分')).fetchone()[0]
        e=Evidence('e1','课程总计5学分。',1,did,'课程规定','https://example.edu/courses',table_id=tid)
        question='按《课程规定》的课程表，总计5学分，已完成2学分，还差多少学分？'
        correct={'table_id':tid,'steps':[{'id':'total','op':'sum','field_id':field},{'id':'done','op':'lookup','literal':'2','unit':'学分'},{'id':'result','op':'subtract','inputs':['total','done']}]}
        facts=NumericEngine.text_facts([e],academic_units=True)
        case={'id':'two-errors','family':'remaining','question':question,'expected':'3','unit':'学分','plan':correct,
            'context':{'tables':StructuredQueryEngine(repo.connection).schema_context([tid],[],question),'facts':[{k:v for k,v in f.items() if k!='evidence'} for f in facts.values()]},
            'source_doc_ids':[did],'table_ids':[tid],'evidence_map':{'e1':asdict(e)},
            'reference':{'measure_field_ids':[field]}}
        broken=json.loads(json.dumps(correct));broken['steps'][0]['op']='lookup';broken['steps'][1]['literal']='5'
        class Model:
            def disable_adapter(self):return nullcontext()
        class Generator:
            model=Model()
            def __init__(self,last):self.outputs=[json.dumps(broken),json.dumps(last)];self.prompts=[]
            def generate(self,prompt,**kwargs):self.prompts.append(prompt);return self.outputs.pop(0)
        generator=Generator(correct)
        row=runtime.evaluate_case(generator,case,repo.connection,'adapter',tmp_path)
        assert not row['first_correct'] and row['correct'] and row['was_repaired']
        assert '已完成量2学分' in generator.prompts[1] and 'lookup匹配多个' in generator.prompts[1]
        partial=json.loads(json.dumps(broken));partial['steps'][0]['op']='sum'
        failed=runtime.evaluate_case(Generator(partial),case,repo.connection,'adapter',tmp_path)
        assert not failed['correct'] and not failed['execution_ok'] and failed['was_repaired']
        summary=runtime.summarize([row,failed])
        assert summary['repair_succeeded']==1 and not summary['repair_introduced_wrong']
