# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Global FFN LoRA layer selection, independent of accelerator dependencies."""

import re


def _layer_indices(value, num_layers, option):
    if isinstance(value, str):
        value = value.split(',')
        if any(not item.strip().isdigit() for item in value):
            raise ValueError(f'{option} must contain comma-separated nonnegative integers')
        value = [int(item.strip()) for item in value]
    try:
        indices = tuple(value)
    except TypeError as exc:
        raise ValueError(f'{option} must be an iterable of layer indices') from exc
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < num_layers:
            raise ValueError(f'{option} index {index!r} must be an integer in [0, {num_layers - 1}]')
    return frozenset(indices)


def parse_ffn_lora_layers(value, num_layers, noop_layers=None):
    """Parse zero-based global indices, inclusive ranges, or ``all``/``none``.

    A stepped range such as ``0-42:3`` includes 0, 3, ..., 42. ``None``
    preserves the existing unrestricted behavior. Explicit no-op indices fail;
    ``all`` automatically omits them.
    """
    if value is None:
        return None
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError('num_layers must be a positive integer for FFN LoRA layer selection')
    noops = _layer_indices(noop_layers, num_layers, 'noop_layers') if noop_layers else frozenset()
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == 'all':
            return tuple(index for index in range(num_layers) if index not in noops)
        if value.lower() == 'none':
            return ()
        indices = set()
        for item in value.split(','):
            match = re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+)\s*(?::\s*(\d+)\s*)?)?', item)
            if match is None:
                raise ValueError(
                    f'Invalid FFN LoRA layer selection {item!r}; use all, none, '
                    'indices, inclusive ranges (30-42), or stepped ranges (0-42:3)'
                )
            start = int(match[1])
            end = int(match[2]) if match[2] is not None else start
            step = int(match[3]) if match[3] is not None else 1
            if start > end or step <= 0:
                raise ValueError(f'Invalid FFN LoRA range {item!r}: require start <= end and step > 0')
            # Validate endpoints before expansion, including a non-reachable end.
            _layer_indices((start, end), num_layers, 'FFN LoRA layer')
            indices.update(range(start, end + 1, step))
    else:
        indices = _layer_indices(value, num_layers, 'FFN LoRA layer')
    selected_noops = set(indices).intersection(noops)
    if selected_noops:
        raise ValueError(f'FFN LoRA selection includes no-op layers: {sorted(selected_noops)}')
    return tuple(sorted(indices))


def select_lora_targets(model, target_modules, selected_layers):
    """Resolve PEFT suffix targets, filtering FFNs by their global block tag.

    Transformer blocks must expose ``_lora_ffn_layer_index``. Returning full
    module paths keeps attention targets unchanged while filtering every target
    below a block's ``mlp``, including shared and sequential experts.
    """
    if selected_layers is None:
        return target_modules
    if isinstance(target_modules, str):
        raise TypeError('FFN layer selection requires a list of LoRA target module names')
    selected_layers = frozenset(selected_layers)
    modules = dict(model.named_modules())
    targets = []
    for name in modules:
        if not name or not any(name == target or name.endswith('.' + target) for target in target_modules):
            continue
        parts = name.split('.')
        if 'mlp' in parts:
            block_name = '.'.join(parts[:parts.index('mlp')])
            block = modules.get(block_name)
            index = getattr(block, '_lora_ffn_layer_index', None)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError(
                    f'Cannot select FFN LoRA target {name!r}: parent block {block_name!r} '
                    'has no valid global _lora_ffn_layer_index'
                )
            if index not in selected_layers:
                continue
        targets.append(name)
    return targets


def cache_lora_targets_for_checkpoint(models, args):
    """Cache actual PEFT targets per model chunk through transparent wrappers."""
    if getattr(args, 'lora_ffn_layers', None) is None:
        return
    targets_by_model = {}
    virtual_pipeline = getattr(args, 'virtual_pipeline_model_parallel_size', None) is not None
    if not virtual_pipeline and len(models) != 1:
        raise ValueError('Expected one model chunk without virtual pipeline parallelism')
    for chunk_index, model in enumerate(models):
        marked_modules = {}
        for module in model.modules():
            metadata = vars(module)
            if '_lora_target_modules_for_checkpoint' in metadata:
                marked_modules[id(module)] = metadata['_lora_target_modules_for_checkpoint']
        if len(marked_modules) != 1:
            raise ValueError(
                f'Model chunk {chunk_index} must contain exactly one LoRA checkpoint target marker; '
                f'found {len(marked_modules)}'
            )
        key = f'model{chunk_index}' if virtual_pipeline else 'model'
        targets_by_model[key] = tuple(next(iter(marked_modules.values())))
    args._lora_target_modules_by_model = targets_by_model


def checkpoint_key_is_lora_target(key, targets):
    """Match a base weight/bias to an actual PEFT target by module boundary."""
    if not isinstance(key, str) or not key.endswith(('.weight', '.bias')):
        return False
    module_name = key.rsplit('.', 1)[0]
    return any(module_name == target or module_name.endswith('.' + target) for target in targets)


def validate_checkpoint_ffn_layers(checkpoint_args, args):
    """Reject loading LoRA weights trained with a different FFN selection."""
    if checkpoint_args is None or not getattr(checkpoint_args, 'lora_target_modules', None):
        return
    if getattr(checkpoint_args, 'lora_ffn_layers', None) is None and getattr(args, 'lora_ffn_layers', None) is None:
        return

    def normalized_layers(saved_args):
        layers = getattr(saved_args, 'lora_ffn_layers', None)
        return parse_ffn_lora_layers(
            'all' if layers is None else layers,
            getattr(saved_args, 'num_layers', args.num_layers),
            getattr(saved_args, 'noop_layers', None),
        )

    saved_layers = normalized_layers(checkpoint_args)
    current_layers = normalized_layers(args)
    if saved_layers != current_layers:
        raise ValueError(
            f'LoRA checkpoint FFN layer selection {saved_layers} does not match current selection '
            f'{current_layers}; use the same --lora-ffn-layers when loading LoRA weights'
        )
