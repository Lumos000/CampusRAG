from contextlib import contextmanager
from pathlib import Path
import json
import csv
import importlib.util
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from campus_rag.config import ROOT, load_settings
from campus_rag.rag.generator import TransformersGenerator


def test_adapter_configuration_resolves_from_project_root(tmp_path):
    config = json.loads((ROOT / 'config/settings.json').read_text(encoding='utf-8'))
    config['planner_adapter_path'] = 'models/adapters/planner_lora/step_300'
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    assert load_settings(path).planner_adapter_path == ROOT / config['planner_adapter_path']
    config.pop('planner_adapter_path')
    path.write_text(json.dumps(config), encoding='utf-8')
    assert load_settings(path).planner_adapter_path is None


def test_adapter_is_used_only_for_planning_and_restored_after_text_errors():
    class Model:
        enabled = True

        @contextmanager
        def disable_adapter(self):
            self.enabled = False
            try:
                yield
            finally:
                self.enabled = True

    generator = TransformersGenerator.__new__(TransformersGenerator)
    generator.model = Model()
    generator.planner_adapter_path = Path('adapter')

    def generate(prompt, *args, **kwargs):
        if prompt == 'error':
            raise ValueError('text generation failed')
        return generator.model.enabled

    generator._generate = generate
    assert generator.generate('text') is False
    assert generator.planning_generate('plan') is True
    assert generator.planning_generate('repair') is True
    with pytest.raises(ValueError):
        generator.generate('error')
    assert generator.planning_generate('after error') is True


def test_disabled_adapter_needs_no_peft_model_wrapper():
    generator = TransformersGenerator.__new__(TransformersGenerator)
    generator.planner_adapter_path = None
    generator.model = object()
    generator._generate = lambda *args, **kwargs: 'base'
    assert generator.generate('text') == generator.planning_generate('plan') == 'base'


@pytest.mark.parametrize('entry', ['app.py', 'scripts/batch_answer.py'])
@pytest.mark.parametrize('disabled', [False, True])
def test_official_entry_points_forward_adapter_switch(tmp_path, monkeypatch, entry, disabled):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('official_entry', ROOT / entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = load_settings()
    database = tmp_path / 'db.sqlite'
    sqlite3.connect(database).close()
    from dataclasses import replace
    settings = replace(settings, database_path=database,
                       planner_adapter_path=tmp_path / 'selected_adapter')
    monkeypatch.setattr(module, 'load_settings', lambda path: settings)
    monkeypatch.setattr(module, 'HybridIndex', SimpleNamespace(load=lambda *args: object()))
    captured = []

    def generator(*args, **kwargs):
        captured.append(kwargs['planner_adapter_path'])
        return object()

    monkeypatch.setattr(module, 'TransformersGenerator', generator)
    monkeypatch.setattr(module, 'RAGService', lambda *args: SimpleNamespace(
        answer=lambda question: SimpleNamespace(answer='answer', references=[])))
    argv = [entry]
    if entry == 'app.py':
        monkeypatch.setattr('builtins.input', lambda prompt: 'exit')
    else:
        source = tmp_path / 'questions.csv'
        with source.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerows([['问题'], ['question']])
        argv += ['--input', str(source), '--output', str(tmp_path / 'answers.csv')]
    if disabled:
        argv.append('--no-adapter')
    monkeypatch.setattr(sys, 'argv', argv)
    module.main()
    assert captured == [None if disabled else settings.planner_adapter_path]
