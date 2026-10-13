"""CPU control-flow checks of real runtime functions, without accelerator imports.

Run with ``python -m pytest --noconftest tests/unit_tests/test_lora_layer_integration.py``.
AST extraction removes only unrelated top-level imports/definitions; tested function
bodies execute unchanged. Hardware classes are stubs, so these are not numerical or
distributed training tests.
"""

import ast
import builtins
import importlib.util
import symtable
import sys
from argparse import ArgumentParser
from copy import deepcopy
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SELECTION_MODULE = 'mindspeed_llm.tasks.posttrain.lora.layer_selection'


def load_functions(path, names, dependencies):
    """Execute the actual source definitions in an explicit dependency namespace."""
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    # Stubs must not hide missing imports in production. Check real global
    # bindings, including references in branches the current case will not take.
    module_symbols = symtable.symtable(source, str(path), 'exec')
    bound_globals = {
        symbol.get_name() for symbol in module_symbols.get_symbols()
        if symbol.is_imported() or symbol.is_assigned() or symbol.is_namespace()
    } | set(vars(builtins))
    # symtable synthesizes an implicit __class__ reference for explicit super().
    source_references = {
        reference.id for node in nodes for reference in ast.walk(node)
        if isinstance(reference, ast.Name) and isinstance(reference.ctx, ast.Load)
    }
    pending = [table for table in module_symbols.get_children() if table.get_name() in names]
    missing = set()
    while pending:
        scope = pending.pop()
        missing.update(
            symbol.get_name() for symbol in scope.get_symbols()
            if symbol.is_referenced() and symbol.is_global()
            and symbol.get_name() in source_references and symbol.get_name() not in bound_globals
        )
        pending.extend(scope.get_children())
    assert not missing, f'{path}: production functions reference unbound globals: {sorted(missing)}'
    namespace = dict(dependencies)
    # Only the named definitions from the repository under test are executed.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)  # noqa: S102
    return SimpleNamespace(**{name: namespace[name] for name in names})


def test_source_loader_rejects_stubs_that_hide_missing_runtime_imports(tmp_path):
    source = tmp_path / 'missing_import.py'
    source.write_text('def factory():\n    def inner():\n        return MissingDependency()\n    return inner\n')
    with pytest.raises(AssertionError, match='unbound globals.*MissingDependency'):
        load_functions(source, ['factory'], {'MissingDependency': lambda: 'would hide the bug'})


def test_source_loader_accepts_real_import_bindings_and_builtins(tmp_path):
    source = tmp_path / 'real_import.py'
    source.write_text(
        'from accelerator import Dependency\n'
        'def factory():\n    def inner():\n        return Dependency(), len([])\n    return inner\n'
    )
    functions = load_functions(source, ['factory'], {'Dependency': lambda: 'stubbed accelerator'})
    assert functions.factory()() == ('stubbed accelerator', 0)


def install_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture(params=['mindspeed_llm', 'tests/mindspeed_llm'])
def runtime(request):
    """Both maintained runtime copies must execute the same behavior."""
    return ROOT / request.param


@pytest.fixture
def selection(runtime, monkeypatch):
    path = runtime / 'tasks/posttrain/lora/layer_selection.py'
    spec = importlib.util.spec_from_file_location(SELECTION_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, SELECTION_MODULE, module)
    return module


class NoopLayer:
    def __init__(self, layer_number):
        self.layer_number = layer_number


def block_harness(runtime, monkeypatch, *, selected=(34, 42), offset=33, count=11, layout=None):
    args = SimpleNamespace(
        transformer_impl='local', num_experts=256, first_k_dense_replace=-1,
        moe_layer_freq=1, moe_grouped_gemm=True, full_attention_interval=0,
        noop_layers={43}, lora_ffn_layers=selected, mtp_num_layers=None,
        enable_mhc=True, recompute_norm=False,
    )
    shared_spec = SimpleNamespace(submodules=SimpleNamespace(
        mlp=SimpleNamespace(params={'existing': 'preserved'}), self_attention=object(),
    ))
    block = SimpleNamespace(
        config=SimpleNamespace(pipeline_model_parallel_layout=layout, num_layer_list=None),
        submodules=SimpleNamespace(layer_specs=[shared_spec] * count, layer_norm=None),
        post_layer_norm=False, post_process=False, pg_collection=None,
    )
    calls = []
    identity = object()

    def build_module(spec, **kwargs):
        if spec is identity:
            return SimpleNamespace()
        calls.append((spec, kwargs))
        return SimpleNamespace(layer_number=kwargs['layer_number'])

    install_module(
        monkeypatch, 'mindspeed_llm.tasks.models.spec.qwen3_next_spec',
        linear_attention_spec=object(), full_attention_spec=object(),
    )
    function = load_functions(runtime / 'core/transformer/transformer_block.py',
                              ['_transformer_block_build_layers'], {
        'get_args': lambda: args, 'deepcopy': deepcopy,
        '_get_layer_offset': lambda *a: offset,
        '_get_mlp_module_spec': lambda **kw: shared_spec.submodules.mlp,
        'parallel_state': SimpleNamespace(get_pipeline_model_parallel_rank=lambda: 3),
        'NoopTransformerLayer': NoopLayer, 'build_module': build_module,
        'IdentityOp': identity, 'torch': SimpleNamespace(nn=SimpleNamespace(ModuleList=list)),
    })._transformer_block_build_layers
    return function, block, shared_spec, calls


def test_block_passes_global_indices_without_polluting_shared_specs(runtime, monkeypatch):
    build, block, original_spec, calls = block_harness(runtime, monkeypatch)
    build(block)
    assert len(block.layers) == 11
    assert len(calls) == 10  # Global index 43 is a no-op and must never construct an FFN.
    assert [layer._lora_ffn_layer_index for layer in block.layers[:-1]] == list(range(33, 43))
    assert isinstance(block.layers[-1], NoopLayer)
    assert block.layers[-1].layer_number == 44
    assert [kwargs['layer_number'] for _, kwargs in calls] == list(range(1, 11))
    assert [spec.submodules.mlp.params['lora_ffn_layer_index'] for spec, _ in calls] == list(range(33, 43))
    assert original_spec.submodules.mlp.params == {'existing': 'preserved'}
    assert len({id(spec.submodules.mlp.params) for spec, _ in calls}) == 10
    assert all(spec is not original_spec for spec, _ in calls)
    assert all(spec.submodules.mlp.params['existing'] == 'preserved' for spec, _ in calls)


def test_explicit_pipeline_layout_supplies_global_offset(runtime, monkeypatch):
    layout = SimpleNamespace(get_layer_offset=lambda **kw: 11)
    build, block, _, calls = block_harness(runtime, monkeypatch, count=2, layout=layout)
    build(block)
    assert [layer._lora_ffn_layer_index for layer in block.layers] == [11, 12]
    assert [spec.submodules.mlp.params['lora_ffn_layer_index'] for spec, _ in calls] == [11, 12]


def test_block_default_does_not_change_module_spec_or_constructor_arguments(runtime, monkeypatch):
    build, block, original_spec, calls = block_harness(runtime, monkeypatch, selected=None, count=2)
    build(block)
    assert all(spec is original_spec for spec, _ in calls)
    assert original_spec.submodules.mlp.params == {'existing': 'preserved'}
    assert all(not hasattr(layer, '_lora_ffn_layer_index') for layer in block.layers)


class ExpertStub:
    def __init__(self, count, config, lora_config=None):
        self.count = count
        self.config = config
        self.lora_config = lora_config


class PlainExperts(ExpertStub):
    pass


class LoraExperts(ExpertStub):
    pass


class MoeBase:
    def __init__(self, config):
        self.expert_parallel_size = 4


class MegatronMoeStub(MoeBase):
    pass


def moe_harness(runtime, monkeypatch, selected):
    args = SimpleNamespace(
        lora_ffn_layers=selected, lora_r=16, lora_alpha=32,
        lora_target_modules=['linear_q', 'linear_fc1', 'linear_fc2'],
        moe_alltoall_overlap_comm=False,
    )
    config = SimpleNamespace(
        moe_layer_recompute=False, moe_tp_extend_ep=False, num_moe_experts=256,
        moe_shared_expert_intermediate_size=2048, moe_shared_expert_overlap=False,
        moe_grouped_gemm=True,
    )
    submodules = SimpleNamespace(experts=SimpleNamespace(),
                                 shared_experts=SimpleNamespace(submodules=SimpleNamespace()))
    shared_calls = []

    def build_module(spec, *module_args, **kwargs):
        if spec is submodules.experts:
            experts = PlainExperts(*module_args, **kwargs)
            experts.source_spec = spec
            return experts
        shared_calls.append((spec, kwargs))
        return SimpleNamespace()

    install_module(monkeypatch, 'mindspeed.core.transformer.moe.moe_feature',
                   TopKRouter=lambda **kwargs: SimpleNamespace(**kwargs), build_module=build_module)
    install_module(monkeypatch, 'mindspeed.core.transformer.moe.moe_feature.adaptor',
                   MindSpeedMOEAlltoAllSeqOverLapDispatcherAdaptor=lambda *a, **kw: object())
    install_module(monkeypatch, 'peft', LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs))
    install_module(monkeypatch, 'mindspeed_llm.tasks.posttrain.lora.moe.experts',
                   LoraParallelGroupedMLP=LoraExperts)
    install_module(monkeypatch, 'megatron.core.tensor_parallel',
                   ColumnParallelLinear=object(), RowParallelLinear=object())
    dependencies = {
        'get_args': lambda: args, 'MegatronMoELayer': MegatronMoeStub,
        'is_enable_lora': lambda: True,
        'parallel_state': SimpleNamespace(get_expert_model_parallel_rank=lambda: 2),
    }
    if runtime.parent == ROOT / 'tests':
        dependencies['MoEAlltoAllTokenDispatcher'] = lambda *a, **kw: SimpleNamespace(kind='alltoall')
    else:
        dependencies['GroupedMLP'] = PlainExperts
        dependencies['MoEAlltoAllSEQTokenDispatcher'] = lambda *a, **kw: SimpleNamespace(kind='legacy_seq')
    init = load_functions(runtime / 'core/transformer/moe/moe_layer.py',
                          ['lora_moe_layer_init'], dependencies).lora_moe_layer_init
    return init, config, submodules, shared_calls


@pytest.mark.parametrize('selected,index,expected', [
    ((34, 42), 34, LoraExperts), ((34, 42), 33, PlainExperts),
    ((), 34, PlainExperts), (None, None, LoraExperts),
])
def test_moe_constructs_only_selected_routed_adapters(runtime, monkeypatch, selected, index, expected):
    init, config, submodules, shared_calls = moe_harness(runtime, monkeypatch, selected)
    layer = MegatronMoeStub.__new__(MegatronMoeStub)
    init(layer, config, submodules, layer_number=1, lora_ffn_layer_index=index)
    assert type(layer.experts) is expected
    assert layer.experts.count == 64
    assert layer.local_expert_indices == list(range(128, 192))
    assert layer.token_dispatcher.kind == ('alltoall' if runtime.parent == ROOT / 'tests' else 'legacy_seq')
    if expected is PlainExperts and runtime.parent == ROOT / 'tests':
        assert layer.experts.source_spec is submodules.experts
    # Shared FFNs always participate in forward; PEFT applies their selection later.
    assert len(shared_calls) == 1
    assert layer.shared_experts is not None
    if expected is LoraExperts:
        assert layer.experts.lora_config.r == 16
        assert layer.experts.lora_config.lora_alpha == 32


def test_moe_selection_rejects_missing_global_index(runtime, monkeypatch):
    init, config, submodules, _ = moe_harness(runtime, monkeypatch, (34,))
    layer = MegatronMoeStub.__new__(MegatronMoeStub)
    with pytest.raises(ValueError, match='global layer index'):
        init(layer, config, submodules, layer_number=1)


def checkpoint_harness(runtime, selection, args):
    return load_functions(runtime / 'tasks/posttrain/lora/utils.py',
                          ['modify_keys_with_dict', 'merge_dicts'], {'get_args': lambda: args})


ATTENTION = 'decoder.layers.0.self_attention.linear_q'
SHARED = 'decoder.layers.0.mlp.shared_experts.linear_fc1'
SHARED_2 = 'decoder.layers.1.mlp.shared_experts.linear_fc1'
REPLACEMENTS = {'weight': 'base_layer.weight', 'bias': 'base_layer.bias'}
EXCLUSIONS = ['base_layer', 'lora_', 'norm']


def checkpoint_parameters():
    return {
        ATTENTION + '.weight': 'attention', SHARED + '.weight': 'shared0',
        SHARED + '.bias': 'bias0', SHARED_2 + '.weight': 'shared1',
        SHARED + '.lora_A.default.weight': 'adapter',
        SHARED + '.base_layer.weight': 'already_wrapped',
        SHARED + '_extra.weight': 'similar_name',
        'decoder.layers.0.mlp.experts.weight1': 'grouped_base',
        'decoder.layers.0.mlp.experts.lora_A.weight1': 'grouped_adapter',
    }


def test_checkpoint_renaming_uses_actual_targets_for_single_model(runtime, selection):
    args = SimpleNamespace(lora_ffn_layers=(33,), lora_target_modules=['linear_fc1', 'linear_q'],
                           _lora_target_modules_by_model={'model': (ATTENTION, SHARED_2)})
    modify = checkpoint_harness(runtime, selection, args).modify_keys_with_dict
    original = {'model': {'nested': checkpoint_parameters()}, 'iteration': 10}
    before = deepcopy(original)
    renamed = modify(original, REPLACEMENTS, EXCLUSIONS)
    params = renamed['model']['nested']
    assert params[ATTENTION + '.base_layer.weight'] == 'attention'
    assert params[SHARED + '.weight'] == 'shared0'
    assert params[SHARED + '.bias'] == 'bias0'
    assert params[SHARED_2 + '.base_layer.weight'] == 'shared1'
    assert params[SHARED + '.lora_A.default.weight'] == 'adapter'
    assert params[SHARED + '.base_layer.weight'] == 'already_wrapped'
    assert params[SHARED + '_extra.weight'] == 'similar_name'
    assert params['decoder.layers.0.mlp.experts.weight1'] == 'grouped_base'
    assert params['decoder.layers.0.mlp.experts.lora_A.weight1'] == 'grouped_adapter'
    assert ATTENTION + '.weight' not in params
    assert SHARED_2 + '.weight' not in params
    assert original == before
    assert renamed['iteration'] == 10


def test_virtual_chunks_with_same_local_layer_name_use_independent_targets(runtime, selection):
    args = SimpleNamespace(lora_ffn_layers=(0,), lora_target_modules=['linear_fc1', 'linear_q'],
                           _lora_target_modules_by_model={
                               'model0': (ATTENTION, SHARED), 'model1': (ATTENTION,),
                           })
    modify = checkpoint_harness(runtime, selection, args).modify_keys_with_dict
    original = {'model0': {SHARED + '.weight': 10, SHARED + '.bias': 20, ATTENTION + '.weight': 30},
                'model1': {SHARED + '.weight': 40, SHARED + '.bias': 50, ATTENTION + '.weight': 60}}
    renamed = modify(original, REPLACEMENTS, EXCLUSIONS)
    assert renamed == {
        'model0': {SHARED + '.base_layer.weight': 10, SHARED + '.base_layer.bias': 20,
                   ATTENTION + '.base_layer.weight': 30},
        'model1': {SHARED + '.weight': 40, SHARED + '.bias': 50,
                   ATTENTION + '.base_layer.weight': 60},
    }
    assert modify(renamed, REPLACEMENTS, EXCLUSIONS) == renamed


def test_default_checkpoint_renaming_retains_suffix_target_behavior(runtime, selection):
    args = SimpleNamespace(lora_ffn_layers=None, lora_target_modules=['linear_fc1', 'linear_q'])
    modify = checkpoint_harness(runtime, selection, args).modify_keys_with_dict
    original = {'model': {SHARED + '.weight': 10, ATTENTION + '.weight': 20}}
    assert modify(original, REPLACEMENTS, EXCLUSIONS) == {
        'model': {SHARED + '.base_layer.weight': 10, ATTENTION + '.base_layer.weight': 20},
    }


@pytest.mark.parametrize('adapter_load', [False, True])
def test_checkpoint_loader_rejects_mismatched_adapter_layer_selection(runtime, selection, adapter_load):
    args = SimpleNamespace(lora_ffn_layers=(33,), num_layers=44, noop_layers={43},
                           lora_target_modules=['linear_q'], lora_load='adapter' if adapter_load else None,
                           _lora_target_modules_by_model={'model': (ATTENTION,)})
    utils = checkpoint_harness(runtime, selection, args)
    wrap = load_functions(runtime / 'training/checkpointing.py', ['_load_base_checkpoint_wrapper'], {
        'get_args': lambda: args, 'wraps': wraps, 'is_enable_lora': lambda: True,
        'modify_keys_with_dict': utils.modify_keys_with_dict, 'merge_dicts': utils.merge_dicts,
    })._load_base_checkpoint_wrapper
    saved_args = SimpleNamespace(lora_ffn_layers=(34,), num_layers=44, noop_layers={43},
                                 lora_target_modules=['linear_q'])
    calls = []

    def load(*load_args, **kwargs):
        calls.append(load_args)
        checkpoint_args = None if adapter_load and len(calls) == 1 else saved_args
        return {'args': checkpoint_args, 'model': {}}, 'checkpoint', False, 'torch'

    with pytest.raises(ValueError, match='does not match current selection'):
        wrap(load)('base', rank0=False)
    assert len(calls) == (2 if adapter_load else 1)


class ModelStub:
    def __init__(self, global_index):
        self.named_children = {
            'decoder.layers.0': SimpleNamespace(_lora_ffn_layer_index=global_index),
            ATTENTION: SimpleNamespace(), SHARED: SimpleNamespace(),
        }

    def named_modules(self):
        return iter([('', self), *self.named_children.items()])

    def modules(self):
        return iter([self])

    def parameters(self):
        return iter([])


class PeftStub:
    def __init__(self, model):
        self.base = model

    def get_base_model(self):
        return self.base

    def add_module(self, name, module):
        setattr(self, name, module)

    def modules(self):
        return iter([self, *self.base.modules()])

    def parameters(self):
        return iter([])

    def print_trainable_parameters(self):
        pass


def training_harness(runtime, selection, monkeypatch, *, selected=(33,), targets=None):
    args = SimpleNamespace(
        use_fused_mlp=False, lora_fusion=False, lu_lora_final_layer_index=None,
        lora_ffn_layers=selected, lora_target_modules=targets or ['linear_q', 'linear_fc1'],
        lora_r=16, lora_alpha=32, recompute_granularity=None,
        virtual_pipeline_model_parallel_size=2,
    )
    configurations = []

    def get_peft_model(model, config):
        configurations.append(config)
        return PeftStub(model)

    install_module(monkeypatch, 'peft', __version__='0.19.1',
                   LoraConfig=lambda **kw: SimpleNamespace(**kw), get_peft_model=get_peft_model,
                   PeftModel=PeftStub, LoraModel=PeftStub)
    install_module(monkeypatch, 'peft.tuners.lora', tp_layer=SimpleNamespace())
    install_module(monkeypatch, 'mindspeed_llm.tasks.posttrain.lora.lora_moe',
                   LoraParallelLinearMoE=object())
    functions = load_functions(runtime / 'training/training.py',
                               ['model_provider_func_wrapper', 'get_model_wrapper'], {
        'wraps': wraps, 'get_args': lambda: args, 'is_enable_lora': lambda: True,
        'core_transformer_config_from_args': lambda _: SimpleNamespace(sequence_parallel=False),
        'print_rank_0': lambda *a: None, 'torch': SimpleNamespace(nn=SimpleNamespace(Linear=type('Linear', (), {}))),
        'megatron': SimpleNamespace(training=SimpleNamespace(utils=SimpleNamespace(ALL_MODULE_WRAPPER_CLASSNAMES=()))),
        'ModelType': SimpleNamespace(encoder_or_decoder='encoder_or_decoder'),
    })
    return functions, args, configurations


def test_get_model_filters_actual_peft_targets_and_caches_each_virtual_chunk(runtime, selection, monkeypatch):
    functions, args, configurations = training_harness(runtime, selection, monkeypatch)
    original_targets = list(args.lora_target_modules)

    def get_models(provider, model_type, wrap_with_ddp, *extra_args):
        return [provider(33), provider(34)]

    models = functions.get_model_wrapper(get_models)(ModelStub)
    assert [config.target_modules for config in configurations] == [[ATTENTION, SHARED], [ATTENTION]]
    assert args._lora_target_modules_by_model == {
        'model0': (ATTENTION, SHARED), 'model1': (ATTENTION,),
    }
    assert args.lora_target_modules == original_targets
    assert models[0].base._lora_target_modules_for_checkpoint == (ATTENTION, SHARED)
    assert models[1].base._lora_target_modules_for_checkpoint == (ATTENTION,)


def test_provider_fails_if_selection_leaves_no_peft_targets(runtime, selection, monkeypatch):
    functions, _, configurations = training_harness(runtime, selection, monkeypatch,
                                                    selected=(), targets=['linear_fc1'])
    with pytest.raises(ValueError, match='without any PEFT target modules'):
        functions.model_provider_func_wrapper(ModelStub)(33)
    assert configurations == []


def test_default_provider_keeps_original_targets_and_no_selection_cache(runtime, selection, monkeypatch):
    functions, args, configurations = training_harness(runtime, selection, monkeypatch, selected=None)

    def get_models(provider, model_type, wrap_with_ddp, *extra_args):
        return [provider(33), provider(34)]

    models = functions.get_model_wrapper(get_models)(ModelStub)
    assert all(config.target_modules is args.lora_target_modules for config in configurations)
    assert not hasattr(args, '_lora_target_modules_by_model')
    assert all(not hasattr(model.base, '_lora_target_modules_for_checkpoint') for model in models)


class FeatureStub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def lora_feature(runtime, selection, monkeypatch):
    install_module(monkeypatch, 'mindspeed.features_manager.feature', MindSpeedFeature=FeatureStub)
    path = runtime / 'features_manager/finetune/lora.py'
    spec = importlib.util.spec_from_file_location('lora_feature_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LoraFeature()


def feature_args(feature, argv):
    parser = ArgumentParser()
    feature.register_args(parser)
    args = parser.parse_args(argv)
    args.__dict__.update(
        num_layers=44, noop_layers='43', num_experts=256,
        position_embedding_type='deepseek4', moe_grouped_gemm=True,
        first_k_dense_replace=-1, moe_layer_freq=1,
        moe_token_dispatcher_type='alltoall_seq', moe_tp_extend_ep=False,
    )
    return args


@pytest.mark.parametrize('expression,expected', [
    ('0,3,30-42:3', (0, 3, 30, 33, 36, 39, 42)),
    ('all', tuple(range(43))), ('none', ()),
])
def test_feature_parses_cli_to_global_layer_tuple(lora_feature, expression, expected):
    args = feature_args(lora_feature, [
        '--lora-ffn-layers', expression, '--lora-target-modules', 'linear_q', 'linear_fc1',
    ])
    assert args.lora_ffn_layers == expression
    lora_feature.validate_args(args)
    assert args.lora_ffn_layers == expected
    # Validation may be called more than once while features patch the parser.
    lora_feature.validate_args(args)
    assert args.lora_ffn_layers == expected


def test_feature_default_preserves_other_model_support(lora_feature):
    args = feature_args(lora_feature, ['--lora-target-modules', 'linear_fc1'])
    args.position_embedding_type = 'rope'
    args.first_k_dense_replace = 3
    assert args.lora_ffn_layers is None
    lora_feature.validate_args(args)
    assert args.lora_ffn_layers is None


@pytest.mark.parametrize('overrides', [
    {'lora_target_modules': []}, {'lora_target_modules': ['linear_fc1', 'linear_fc2']},
    {'position_embedding_type': 'rope'}, {'num_experts': None}, {'num_experts': -1},
    {'moe_grouped_gemm': False}, {'first_k_dense_replace': 3}, {'moe_layer_freq': 2},
    {'mtp_num_layers': 1}, {'lu_lora_final_layer_index': 0}, {'use_fused_mlp': True},
    {'moe_alltoall_overlap_comm': True}, {'moe_allgather_overlap_comm': True}, {'moe_fb_overlap': True},
])
def test_feature_rejects_unsupported_selection_paths(lora_feature, overrides):
    args = feature_args(lora_feature, ['--lora-ffn-layers', '30-42', '--lora-target-modules', 'linear_q'])
    args.__dict__.update(overrides)
    with pytest.raises(ValueError, match='--lora-ffn-layers'):
        lora_feature.validate_args(args)


@pytest.mark.parametrize('expression', ['43', '0-44', '1-2:0', '1,'])
def test_feature_rejects_invalid_or_noop_indices(lora_feature, expression):
    args = feature_args(lora_feature, ['--lora-ffn-layers', expression, '--lora-target-modules', 'linear_q'])
    with pytest.raises(ValueError):
        lora_feature.validate_args(args)
