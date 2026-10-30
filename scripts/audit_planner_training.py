"""Check supervised planner coverage before migrating training to a new corpus."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def summarize(rows, required_units, require_composition=False):
    coverage, operations = Counter(), Counter()
    templates = defaultdict(set)
    composed = 0
    for row in rows:
        context = row.get('context', {})
        facts = {fact['fact_id']: fact for fact in context.get('facts', [])}
        fields = {column['field_id']: column for table in context.get('tables', []) for column in table['columns']}
        steps = row.get('plan', {}).get('steps', [])
        by_id = {step['id']: step for step in steps}
        used_units = set()
        for step in steps:
            operations[step['op']] += 1
            source = facts.get(step.get('fact_id')) or fields.get(step.get('field_id')) or step
            if source.get('unit'):
                used_units.add(source['unit'])
            used_units.update(value['unit'] for value in step.get('inputs', [])
                              if isinstance(value, dict) and value.get('unit'))
        coverage.update(used_units)
        def contains_multiply(step_id, seen):
            if step_id in seen or step_id not in by_id:
                return False
            seen.add(step_id)
            step = by_id[step_id]
            return step['op'] == 'multiply' or any(contains_multiply(child, seen) for child in step.get('inputs', []) if isinstance(child, str))
        if any(step['op'] == 'subtract' and any(contains_multiply(child, set()) for child in step.get('inputs', []) if isinstance(child, str)) for step in steps):
            composed += 1
        if row.get('template_family'):
            templates[row.get('kind', row.get('family', 'unknown'))].add(row['template_family'])
    missing = sorted(set(required_units) - set(coverage))
    return {'examples': len(rows), 'supervised_unit_examples': dict(coverage),
        'missing_required_units': missing, 'operation_steps': dict(operations),
        'multiply_then_subtract_examples': composed,
        'template_families_per_kind': {key: len(value) for key, value in sorted(templates.items())},
        'passed': bool(rows) and not missing and (not require_composition or composed > 0),
        'scope': 'Supervised operands only; distractor facts do not count. Reference execution and held-out evaluation remain required.'}


def main():
    parser = argparse.ArgumentParser(description='训练迁移前检查新增单位与组合运算覆盖；不训练、不修改数据库')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--required-units', nargs='+', default=['学分', '学时'])
    parser.add_argument('--require-composition', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.train.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = summarize(rows, args.required_units, args.require_composition)
    if args.output:
        if args.output.exists():
            parser.error('输出已存在，请使用新的审查文件名')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
