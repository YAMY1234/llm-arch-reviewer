"""Model-independent lifecycle regressions; existing revision-7 math stays strict."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import jsonschema
import pytest
import yaml

from llm_arch_v2.compiler import CatalogError, compile_catalog
from llm_arch_v2.validation_evidence import validate_validation_evidence
from test_validation_evidence import _toy_contract, _write_yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def semantic_catalog(tmp_path):
    # Independently authored scale module with exact edge-owned ports.
    dest = tmp_path / 'generic_semantic_fixture'
    dest.mkdir()
    def node(name, op, shape='elem', **extra):
        return {'id': name, 'label': name, 'semantic_op': op, 'shape': shape, **extra}
    def edge(a, b, identity):
        return {'from': a, 'to': b, 'identity': identity, 'shape': '[H]',
                'dtype': 'float32', 'layout': 'feature', 'state': 'invocation'}
    model = {
        'schema_version': 'model-ir.v2', 'model_id': dest.name,
        'model_label': 'Generic scale fixture', 'ir_version': '1', 'semantic_revision': 7,
        'default_view': 'top', 'dimensions': {'H': 4},
        'dimension_symbols': {'H': {'meaning': 'Feature width', 'value_class': 'model_constant',
                                   'value': 4, 'provenance': 'Fixture specification'}},
        'semantic_coverage': {**{key: 'complete' for key in (
            'parameter_closure', 'state_closure', 'layer_variant_closure',
            'operator_dataflow_closure', 'detail_view_closure')},
            'config_field_disposition': {key: [] for key in (
                'model_ir', 'execution_ir', 'binding_profile', 'excluded')}},
        'semantic_contract': {'version': 1, 'operations': {
            'identity': {'kind': 'boundary', 'equation': 'Pass the boundary tensor unchanged.'},
            'scale_module': {'kind': 'module', 'equation': 'y = 2 * x'},
            'scale': {'kind': 'elementwise', 'equation': 'y[h] = 2 * x[h]'}}},
        'views': {
            'top': {'nodes': [node('input', 'identity', 'io'),
                             node('module', 'scale_module', 'block', drill='scale'),
                             node('output', 'identity', 'io')],
                    'edges': [edge('input', 'module', 'x'), edge('module', 'output', 'y')]},
            'scale': {'nodes': [node('input', 'identity', 'io', boundary_direction='input'),
                               node('multiply', 'scale'),
                               node('output', 'identity', 'io', boundary_direction='output')],
                      'edges': [edge('input', 'multiply', 'x'), edge('multiply', 'output', 'y')]}},
        'boundary_contracts': [{'parent_node': 'top.module', 'child_view': 'scale',
                               'boundary_mode': 'exact_node', 'input_shape': '[H]', 'output_shape': '[H]'}]}
    pipeline = {'lifecycle': 'model_only', 'acceptance': {'semantic_release_contract': {
        'expected_revision': 7, 'required_views': ['scale'],
        'required_drills': {'top.module': 'scale'},
        'required_nodes': {'scale': ['input', 'multiply', 'output']}}}}
    _write_yaml(dest / 'model_ir.yaml', model)
    _write_yaml(dest / 'pipeline.yaml', pipeline)
    shutil.copy(ROOT / 'catalog' / 'semantic-policy.yaml', tmp_path / 'semantic-policy.yaml')
    return dest


def test_generic_model_only_zero_runtime_and_deterministic(semantic_catalog):
    bundle = compile_catalog(semantic_catalog)
    assert bundle == compile_catalog(semantic_catalog)
    assert bundle['meta']['lifecycle'] == 'model_only'
    assert bundle['model_ir']['semantic_revision'] == 7
    for field in ('execution_variants', 'implementations', 'profiles', 'comparison_contracts', 'sol_profiles'):
        assert bundle[field] == {}
    assert bundle['views'] == bundle['model_ir']['views']
    assert any(n['semantics']['kind'] in {'linear', 'elementwise', 'normalization', 'attention'}
               for v in bundle['views'].values() for n in v['nodes'])
    assert all(not v['nodes_profile'] for v in bundle['enriched'].values())


def test_model_only_rejects_unreachable_template_view(semantic_catalog):
    path = semantic_catalog / 'model_ir.yaml'
    model = yaml.safe_load(path.read_text())
    model['views']['orphan_template'] = {'title': 'Unreachable template', 'nodes': [], 'edges': []}
    _write_yaml(path, model)
    with pytest.raises(CatalogError, match='unreachable model_only views'):
        compile_catalog(semantic_catalog)


@pytest.mark.parametrize('field', ['ms_per_iter', 'active_gpu_ms', 'fusion_owner', 'included_in', 'duration_ms', 'ideal_ms', 'avg_us'])
def test_model_only_rejects_fake_runtime(semantic_catalog, field):
    path = semantic_catalog / 'model_ir.yaml'
    model = yaml.safe_load(path.read_text())
    next(iter(model['views'].values()))['nodes'][0][field] = 0 if 'ms' in field else 'top.fake'
    _write_yaml(path, model)
    with pytest.raises(CatalogError, match='runtime evidence is forbidden'):
        compile_catalog(semantic_catalog)


def test_profiled_without_plans_still_fails(semantic_catalog):
    path = semantic_catalog / 'pipeline.yaml'
    pipeline = yaml.safe_load(path.read_text())
    pipeline['lifecycle'] = 'profiled'
    _write_yaml(path, pipeline)
    with pytest.raises(CatalogError, match='no execution plans'):
        compile_catalog(semantic_catalog)


def test_model_only_cannot_attach_runtime_files(semantic_catalog):
    _write_yaml(semantic_catalog / 'bindings' / 'fake.yaml', {})
    with pytest.raises(CatalogError, match='cannot contain bindings'):
        compile_catalog(semantic_catalog)


def model_only_evidence():
    contract = _toy_contract()
    contract['lifecycle'] = 'model_only'
    contract['authorities'] = contract['authorities'][:1]
    for name in contract['gates']:
        if name != 'semantic_ir':
            contract['gates'][name] = {'status': 'out_of_scope', 'reason': 'No runtime evidence requested.'}
    return contract


def test_model_only_evidence_schema_and_runtime_gate_bypass(tmp_path):
    schema = json.loads((ROOT / 'schema/v2/validation-evidence.schema.json').read_text())
    contract = model_only_evidence()
    jsonschema.validate(contract, schema)
    root = tmp_path / 'toy'
    _write_yaml(root / 'pipeline.yaml', {'lifecycle': 'model_only'})
    _write_yaml(root / 'model_ir.yaml', {'facts': {'layers': 1}})
    _write_yaml(root / 'validation_evidence.yaml', contract)
    report = validate_validation_evidence(root)
    assert report['status'] == 'pass', report['errors']
    assert report['production_release_ready'] is False
    contract['lifecycle'] = 'profiled'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(contract, schema)
    _write_yaml(root / 'pipeline.yaml', {'lifecycle': 'profiled'})
    _write_yaml(root / 'validation_evidence.yaml', contract)
    assert validate_validation_evidence(root)['status'] == 'fail'


def test_semantic_gate_cannot_be_out_of_scope():
    schema = json.loads((ROOT / 'schema/v2/validation-evidence.schema.json').read_text())
    contract = model_only_evidence()
    contract['gates']['semantic_ir'] = {'status': 'out_of_scope', 'reason': 'skip'}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(contract, schema)


def test_shared_audit_never_reports_production_acceptance(semantic_catalog, tmp_path, monkeypatch):
    from scripts import audit_model_only as audit
    bundle = compile_catalog(semantic_catalog)
    bundle_path = tmp_path / 'arch_data.json'
    bundle_path.write_text(json.dumps(bundle))
    monkeypatch.setattr(audit, 'validate_validation_evidence', lambda _: {'status': 'pass'})
    monkeypatch.setattr(audit, 'audit_semantic_closure', lambda **_: {'status': 'complete'})
    report = audit.audit_model_only(semantic_catalog, tmp_path, bundle_path)
    assert report['status'] == 'pass'
    assert report['production_release_ready'] is False
    assert report['browser_verified'] is False
    bundle['views'] = {}
    bundle_path.write_text(json.dumps(bundle))
    assert audit.audit_model_only(semantic_catalog, tmp_path, bundle_path)['status'] == 'fail'


@pytest.mark.parametrize('weaken', ['remove_contract', 'disable_equations'])
def test_model_only_requires_explicit_semantic_contract(semantic_catalog, weaken):
    path = semantic_catalog / 'model_ir.yaml'
    model = yaml.safe_load(path.read_text())
    if weaken == 'remove_contract':
        model.pop('semantic_contract')
    else:
        model['semantic_contract']['require_explicit_equations'] = False
    _write_yaml(path, model)
    with pytest.raises(CatalogError):
        compile_catalog(semantic_catalog)


@pytest.mark.parametrize('gate', ['execution_contract', 'binding_reconciliation', 'production_evidence'])
def test_out_of_scope_requires_exact_fields(tmp_path, gate):
    root = tmp_path / 'toy'
    _write_yaml(root / 'pipeline.yaml', {'lifecycle': 'model_only'})
    _write_yaml(root / 'model_ir.yaml', {'facts': {'layers': 1}})
    contract = model_only_evidence()
    contract['gates'][gate]['authority_refs'] = ['official']
    _write_yaml(root / 'validation_evidence.yaml', contract)
    report = validate_validation_evidence(root)
    assert report['status'] == 'fail'
    assert any('out_of_scope and reason only' in e for e in report['errors'])


def test_evidence_validator_enforces_schema(tmp_path):
    root = tmp_path / 'toy'
    _write_yaml(root / 'pipeline.yaml', {'lifecycle': 'model_only'})
    _write_yaml(root / 'model_ir.yaml', {'facts': {'layers': 1}})
    contract = model_only_evidence()
    contract.pop('contract_revision')
    _write_yaml(root / 'validation_evidence.yaml', contract)
    report = validate_validation_evidence(root)
    assert report['status'] == 'fail'
    assert any('contract_revision' in e for e in report['errors'])


@pytest.mark.parametrize('field,value', [
    ('identity', 'wrong_state'), ('shape', '[2,H]'), ('dtype', 'float16'),
    ('layout', 'transposed'), ('state', 'persistent')])
def test_exact_model_only_boundary_mutations_fail(semantic_catalog, field, value):
    path = semantic_catalog / 'model_ir.yaml'
    model = yaml.safe_load(path.read_text())
    model['views']['scale']['edges'][0][field] = value
    # Old containment validation accepted a union containing both actual shapes.
    model['boundary_contracts'][0]['input_shape'] = '[H] | [2,H]'
    _write_yaml(path, model)
    with pytest.raises(CatalogError, match='exact input tensor boundary mismatch'):
        compile_catalog(semantic_catalog)


def test_identical_boundary_fanout_is_one_logical_port(semantic_catalog):
    path = semantic_catalog / 'model_ir.yaml'
    model = yaml.safe_load(path.read_text())
    model['views']['scale']['edges'].append(deepcopy(model['views']['scale']['edges'][0]))
    _write_yaml(path, model)
    assert compile_catalog(semantic_catalog)['meta']['lifecycle'] == 'model_only'


def test_explicit_port_rename_preserves_all_other_fields(semantic_catalog):
    path = semantic_catalog / 'model_ir.yaml'
    model = yaml.safe_load(path.read_text())
    model['views']['scale']['edges'][0]['identity'] = 'local_x'
    model['boundary_contracts'][0]['port_bindings'] = {'inputs': {'local_x': 'x'}}
    _write_yaml(path, model)
    assert compile_catalog(semantic_catalog)['meta']['lifecycle'] == 'model_only'
    model['views']['scale']['edges'][0]['state'] = 'persistent'
    _write_yaml(path, model)
    with pytest.raises(CatalogError, match='exact input tensor boundary mismatch'):
        compile_catalog(semantic_catalog)


def test_explicit_port_binding_requires_real_endpoints(semantic_catalog):
    path = semantic_catalog / 'model_ir.yaml'
    model = yaml.safe_load(path.read_text())
    model['boundary_contracts'][0]['port_bindings'] = {'inputs': {'fake': 'x'}}
    _write_yaml(path, model)
    with pytest.raises(CatalogError, match='port binding references absent'):
        compile_catalog(semantic_catalog)


def test_port_binding_cannot_collapse_distinct_inputs(semantic_catalog):
    path = semantic_catalog / 'model_ir.yaml'
    model = yaml.safe_load(path.read_text())
    extra = deepcopy(model['views']['scale']['edges'][0])
    extra['identity'] = 'other_x'
    model['views']['scale']['edges'].append(extra)
    model['boundary_contracts'][0]['port_bindings'] = {'inputs': {'other_x': 'x'}}
    _write_yaml(path, model)
    with pytest.raises(CatalogError, match='cannot merge distinct tensor identities'):
        compile_catalog(semantic_catalog)


def test_lifecycle_handoff_must_be_an_actual_internal_tensor():
    from llm_arch_v2.compiler import _validate_model_only_boundaries
    def edge(a, b, name):
        return {'from': a, 'to': b, 'identity': name, 'shape': '[H]',
                'layout': 'feature', 'dtype': 'float32', 'state': 'invocation'}
    model = {'views': {
        'top': {'edges': [edge('input', 'pre', 'x'), edge('pre', 'post', 'handoff'),
                          edge('post', 'output', 'y')]},
        'child': {'nodes': [{'id': 'i', 'boundary_direction': 'input'},
                            {'id': 'h', 'boundary_direction': 'handoff'},
                            {'id': 'o', 'boundary_direction': 'output'}],
                  'edges': [edge('i', 'collapse', 'x'), edge('h', 'combine', 'handoff'),
                            edge('combine', 'o', 'y')]}},
        'boundary_contracts': [{'parent_node': 'top.pre', 'child_view': 'child',
                               'boundary_mode': 'exact_lifecycle',
                               'scope_nodes': ['top.pre', 'top.post']}]}
    _validate_model_only_boundaries(model, source=Path('fixture'))
    model['views']['child']['edges'][1]['dtype'] = 'float16'
    with pytest.raises(CatalogError, match='handoffs must match actual scoped internal edges'):
        _validate_model_only_boundaries(model, source=Path('fixture'))
