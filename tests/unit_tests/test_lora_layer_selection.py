"""CPU-only checks: python -m pytest --noconftest tests/unit_tests/test_lora_layer_selection.py."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    'lora_layer_selection', _ROOT / 'mindspeed_llm/tasks/posttrain/lora/layer_selection.py'
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
parse_ffn_lora_layers = _MODULE.parse_ffn_lora_layers
select_lora_targets = _MODULE.select_lora_targets
cache_lora_targets_for_checkpoint = _MODULE.cache_lora_targets_for_checkpoint
checkpoint_key_is_lora_target = _MODULE.checkpoint_key_is_lora_target
validate_checkpoint_ffn_layers = _MODULE.validate_checkpoint_ffn_layers


@pytest.mark.parametrize('value, expected', [
    (None, None),
    ('none', ()),
    (' ALL ', tuple(range(43))),
    ('30-42', tuple(range(30, 43))),
    ('0-42:3', tuple(range(0, 43, 3))),
    ('0, 3, 6-10:2, 3, 42', (0, 3, 6, 8, 10, 42)),
    (' 0 - 4 : 2 ', (0, 2, 4)),
    ([42, 0, 42], (0, 42)),
    ((4, 2), (2, 4)),
    ([], ()),
])
def test_parse_layer_selection(value, expected):
    assert parse_ffn_lora_layers(value, 44, noop_layers='43') == expected


@pytest.mark.parametrize('noops', ['0,43', {0, 43}, (43, 0), [0, 43]])
def test_all_excludes_noop_layers(noops):
    assert parse_ffn_lora_layers('all', 44, noops) == tuple(range(1, 43))


@pytest.mark.parametrize('value', [
    '', ' ', ',', '0,', '-1', '4-1', '1-4:0', '1-4:-2', '1:3',
    'all,1', '1.5', '44', '0-44:100', '43', '0-43', [43], [44], [-1],
    [True], [1.0], ['1'], 3,
])
def test_invalid_layer_selection_fails(value):
    with pytest.raises(ValueError):
        parse_ffn_lora_layers(value, 44, noop_layers={43})


@pytest.mark.parametrize('num_layers', [0, -1, None, 2.5, True])
def test_invalid_model_layer_count_fails(num_layers):
    with pytest.raises(ValueError, match='num_layers'):
        parse_ffn_lora_layers('all', num_layers)


@pytest.mark.parametrize('noops', ['x', '0,', '-1', [44], [-1], [True], [0.5]])
def test_invalid_noop_indices_fail(noops):
    with pytest.raises(ValueError, match='noop_layers'):
        parse_ffn_lora_layers('all', 44, noops)


class FakeModel:
    def __init__(self, tag=True):
        self.modules = {'': self}
        for local_index, global_index in enumerate((33, 34)):
            block_path = f'decoder.layers.{local_index}'
            block = SimpleNamespace()
            if tag:
                block._lora_ffn_layer_index = global_index
            self.modules[block_path] = block
            for suffix in (
                'self_attention.linear_q', 'self_attention.linear_fc1',
                'mlp.linear_fc1', 'mlp.linear_fc2',
                'mlp.shared_experts.linear_fc1', 'mlp.shared_experts.linear_fc2',
                'mlp.experts.local_experts.0.linear_fc1',
                'mlp.experts.local_experts.0.linear_fc2',
                'mlp.experts.local_experts.1.linear_fc1',
                'mlp.linear_fc1_extra', 'mlp.mylinear_fc1',
                'other_mlp.linear_fc1',
            ):
                self.modules[f'{block_path}.{suffix}'] = SimpleNamespace()

    def named_modules(self):
        return self.modules.items()


def test_default_does_not_inspect_model_or_change_targets():
    targets = ['linear_fc1', 'linear_q']
    assert select_lora_targets(None, targets, None) is targets


def test_pipeline_local_layer_uses_global_index_and_filters_all_experts():
    model = FakeModel()
    targets = select_lora_targets(model, ['linear_q', 'linear_fc1', 'linear_fc2'], (33,))
    expected = [
        name for name in model.modules
        if name.endswith(('.linear_q', '.linear_fc1', '.linear_fc2'))
        and not name.startswith('decoder.layers.1.mlp.')
    ]
    assert targets == expected
    assert 'decoder.layers.0.mlp.shared_experts.linear_fc1' in targets
    assert 'decoder.layers.0.mlp.experts.local_experts.1.linear_fc1' in targets
    assert 'decoder.layers.1.self_attention.linear_fc1' in targets
    assert 'decoder.layers.1.other_mlp.linear_fc1' in targets
    assert not any('mylinear_fc1' in name or 'linear_fc1_extra' in name for name in targets)


def test_local_layer_zero_is_not_global_layer_zero():
    targets = select_lora_targets(FakeModel(), ['linear_fc1', 'linear_q'], (0,))
    assert not any('.mlp.' in name for name in targets)
    assert len([name for name in targets if '.self_attention.linear_q' in name]) == 2


def test_none_selection_keeps_attention_and_can_return_no_targets():
    targets = select_lora_targets(FakeModel(), ['linear_fc1', 'linear_q'], ())
    assert not any('.mlp.' in name for name in targets)
    assert select_lora_targets(FakeModel(), ['mlp.linear_fc2'], ()) == []


def test_exact_full_target_and_suffix_matching():
    target = 'decoder.layers.0.mlp.shared_experts.linear_fc1'
    assert select_lora_targets(FakeModel(), [target], (33,)) == [target]
    assert select_lora_targets(FakeModel(), [target], (34,)) == []


@pytest.mark.parametrize('selection', [(), (33,), (0,)])
def test_unmapped_ffn_never_silently_uses_local_layer_number(selection):
    with pytest.raises(ValueError, match='global _lora_ffn_layer_index'):
        select_lora_targets(FakeModel(tag=False), ['linear_fc1'], selection)


def test_unmapped_non_ffn_targets_need_no_global_tag():
    assert len(select_lora_targets(FakeModel(tag=False), ['linear_q'], (33,))) == 2


def test_regex_targets_fail_explicitly_for_layer_selection():
    with pytest.raises(TypeError, match='list'):
        select_lora_targets(FakeModel(), '.*linear_fc1', (33,))


class FakeChunk:
    def __init__(self, *children):
        self.children = children

    def modules(self):
        return iter((self, *self.children))


def test_checkpoint_cache_keeps_virtual_chunks_separate_and_deduplicates_wrappers():
    first_target = 'decoder.layers.0.mlp.linear_fc1'
    second_target = 'decoder.layers.0.self_attention.linear_q'
    first = SimpleNamespace(_lora_target_modules_for_checkpoint=(first_target,))
    second = SimpleNamespace(_lora_target_modules_for_checkpoint=(second_target,))
    args = SimpleNamespace(lora_ffn_layers=(0,), virtual_pipeline_model_parallel_size=2)
    cache_lora_targets_for_checkpoint([FakeChunk(first, first), FakeChunk(second)], args)
    assert args._lora_target_modules_by_model == {
        'model0': (first_target,), 'model1': (second_target,),
    }
    key = first_target + '.weight'
    assert checkpoint_key_is_lora_target(key, args._lora_target_modules_by_model['model0'])
    assert not checkpoint_key_is_lora_target(key, args._lora_target_modules_by_model['model1'])


def test_checkpoint_cache_records_empty_selection_on_a_fully_frozen_stage():
    base = SimpleNamespace(_lora_target_modules_for_checkpoint=())
    args = SimpleNamespace(lora_ffn_layers=())
    cache_lora_targets_for_checkpoint([FakeChunk(base)], args)
    assert args._lora_target_modules_by_model == {'model': ()}


def test_default_checkpoint_cache_leaves_legacy_behavior_untouched():
    args = SimpleNamespace(lora_ffn_layers=None)
    cache_lora_targets_for_checkpoint(None, args)
    assert not hasattr(args, '_lora_target_modules_by_model')


@pytest.mark.parametrize('marker_count', [0, 2])
def test_checkpoint_cache_fails_on_missing_or_ambiguous_marker(marker_count):
    markers = [SimpleNamespace(_lora_target_modules_for_checkpoint=()) for _ in range(marker_count)]
    with pytest.raises(ValueError, match='exactly one'):
        cache_lora_targets_for_checkpoint([FakeChunk(*markers)], SimpleNamespace(lora_ffn_layers=()))


def test_nonvirtual_checkpoint_cache_rejects_multiple_chunks():
    with pytest.raises(ValueError, match='one model chunk'):
        cache_lora_targets_for_checkpoint([FakeChunk(), FakeChunk()], SimpleNamespace(lora_ffn_layers=()))


@pytest.mark.parametrize('key, expected', [
    ('decoder.layers.0.mlp.linear_fc1.weight', True),
    ('decoder.layers.0.mlp.linear_fc1.bias', True),
    ('base_model.model.decoder.layers.0.mlp.linear_fc1.weight', True),
    ('decoder.layers.1.mlp.linear_fc1.weight', False),
    ('decoder.layers.0.mlp.linear_fc1_extra.weight', False),
    ('decoder.layers.0.mlp.linear_fc1.weight_extra', False),
    ('decoder.layers.0.mlp.linear_fc1.base_layer.weight', False),
    ('decoder.layers.0.mlp.linear_fc1.lora_A.default.weight', False),
    ('decoder.layers.0.mlp.mylinear_fc1.weight', False),
    ('notdecoder.layers.0.mlp.linear_fc1.weight', False),
    ('decoder.layers.0.mlp.linear_fc1', False),
    (None, False),
])
def test_checkpoint_target_matching_obeys_module_boundaries(key, expected):
    targets = ('decoder.layers.0.mlp.linear_fc1',)
    assert checkpoint_key_is_lora_target(key, targets) == expected


def checkpoint_args(selection=None, **overrides):
    values = {
        'lora_target_modules': ['linear_fc1'], 'lora_ffn_layers': selection, 'num_layers': 44, 'noop_layers': {43}
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('saved, current', [
    (None, 'all'), ('all', None), ('all', tuple(range(43))),
    ('0-42:3', tuple(range(0, 43, 3))), ('none', ()), ('0,2,0', (2, 0)),
])
def test_checkpoint_selection_accepts_equivalent_forms(saved, current):
    validate_checkpoint_ffn_layers(checkpoint_args(saved), checkpoint_args(current))


def test_legacy_lora_checkpoint_without_new_argument_means_all():
    legacy = checkpoint_args()
    del legacy.lora_ffn_layers
    validate_checkpoint_ffn_layers(legacy, checkpoint_args('all'))
    with pytest.raises(ValueError, match='does not match'):
        validate_checkpoint_ffn_layers(legacy, checkpoint_args((30, 31)))


@pytest.mark.parametrize('saved, current', [('none', None), ('0-42:3', '30-42'), ('0', '1')])
def test_checkpoint_selection_rejects_mismatched_adapters(saved, current):
    with pytest.raises(ValueError, match='does not match'):
        validate_checkpoint_ffn_layers(checkpoint_args(saved), checkpoint_args(current))


@pytest.mark.parametrize('saved', [None, SimpleNamespace(), checkpoint_args(lora_target_modules=[])])
def test_base_checkpoint_bypasses_lora_selection_check(saved):
    validate_checkpoint_ffn_layers(saved, checkpoint_args('30-42'))


def test_default_checkpoint_validation_needs_no_new_metadata():
    saved = SimpleNamespace(lora_target_modules=['linear_fc1'])
    validate_checkpoint_ffn_layers(saved, SimpleNamespace())


def test_cpu_test_mirror_matches_source():
    source = _ROOT / 'mindspeed_llm/tasks/posttrain/lora/layer_selection.py'
    mirror = _ROOT / 'tests/mindspeed_llm/tasks/posttrain/lora/layer_selection.py'
    assert source.read_bytes() == mirror.read_bytes()
