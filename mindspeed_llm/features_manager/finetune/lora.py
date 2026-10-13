# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class LoraFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__(feature_name="lora", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--lora-target-modules', nargs='+', type=str, default=[], help='Lora target modules.')
        group.add_argument(
            '--lora-ffn-layers', type=str, default=None,
            help='DeepSeek-V4 FFN LoRA global zero-based layers: all, none, comma-separated indices, '
                 'inclusive ranges (30-42), or stepped ranges (0-42:3). Omit to preserve all-layer LoRA.',
        )
        group.add_argument('--lora-load', type=str, default=None, help='Directory containing a lora model checkpoint.')
        group.add_argument('--lora-r', type=int, default=16, help='Lora r.')
        group.add_argument('--lora-alpha', type=int, default=32, help='Lora alpha.')
        group.add_argument('--lora-modules-to-save', nargs='+', type=str, default=None, help='Lora modules to save.')
        group.add_argument(
            '--lora-register-forward-hook',
            nargs='+',
            type=str,
            default=['word_embeddings', 'input_layernorm'],
            help='Lora register forward hook.',
        )
        group.add_argument('--lora-fusion', action='store_true', help='use fusion to accelerate lora.')
        group.add_argument(
            '--lora-ckpt-filter', action='store_true', default=False, help='Enable only saving lora checkpoint.'
        )

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.training.training import get_model_wrapper

        patch_manager.register_patch('megatron.training.training.get_model', get_model_wrapper)

        from mindspeed_llm.training.utils import unwrap_model_wrapper
        from mindspeed_llm.training.checkpointing import _load_base_checkpoint_wrapper, save_checkpoint_wrapper
        from mindspeed_llm.core.transformer.moe.moe_layer import lora_moe_layer_init
        from mindspeed_llm.core.distributed.finalize_model_grads import _allreduce_word_embedding_grads
        from mindspeed_llm.tasks.posttrain.lora.utils import dispatch_megatron_wrapper

        patch_manager.register_patch(
            'megatron.core.distributed.finalize_model_grads._allreduce_word_embedding_grads',
            _allreduce_word_embedding_grads,
        )

        # fix unwrap PerfModel
        patch_manager.register_patch('megatron.training.checkpointing.unwrap_model', unwrap_model_wrapper)
        patch_manager.register_patch('megatron.training.training.unwrap_model', unwrap_model_wrapper)
        patch_manager.register_patch(
            'megatron.training.checkpointing._load_base_checkpoint', _load_base_checkpoint_wrapper
        )
        patch_manager.register_patch('megatron.training.checkpointing.save_checkpoint', save_checkpoint_wrapper)
        patch_manager.register_patch('peft.tuners.lora.tp_layer.dispatch_megatron', dispatch_megatron_wrapper)
        if hasattr(args, 'lora_target_modules') and args.lora_target_modules:
            patch_manager.register_patch(
                'megatron.core.transformer.moe.moe_layer.MoELayer.__init__', lora_moe_layer_init
            )

    def validate_args(self, args):
        has_valid_lora_target = hasattr(args, 'lora_target_modules') and args.lora_target_modules

        if args.num_experts and (has_valid_lora_target and args.moe_token_dispatcher_type != "alltoall_seq"):  # nosec
            raise AssertionError('Lora in the moe only enable the alltoall_seq.')

        if has_valid_lora_target and args.moe_tp_extend_ep:
            raise AssertionError('Lora is not supported with moe-tp-extend-ep.')

        if getattr(args, 'lora_ffn_layers', None) is None:
            return

        if not has_valid_lora_target:
            raise ValueError('--lora-ffn-layers requires --lora-target-modules.')
        if getattr(args, 'position_embedding_type', None) != 'deepseek4':
            raise ValueError('--lora-ffn-layers currently supports only DeepSeek-V4 (position-embedding-type deepseek4).')
        if not args.num_experts or args.num_experts <= 0 or not getattr(args, 'moe_grouped_gemm', False):
            raise ValueError('--lora-ffn-layers requires MoE experts with --moe-grouped-gemm.')
        if getattr(args, 'first_k_dense_replace', None) != -1 or getattr(args, 'moe_layer_freq', None) != 1:
            raise ValueError('--lora-ffn-layers requires all-MoE layers: --first-k-dense-replace -1 --moe-layer-freq 1.')

        attention_targets = {'linear_q', 'linear_kv', 'linear_q_up_proj', 'linear_o_down_proj', 'linear_o_up_proj'}
        if not attention_targets.intersection(args.lora_target_modules):
            raise ValueError(
                '--lora-ffn-layers requires at least one global DeepSeek-V4 attention target '
                '(linear_q, linear_kv, linear_q_up_proj, linear_o_down_proj, or linear_o_up_proj) '
                'so pipeline stages without selected FFNs still train attention LoRA.'
            )
        if getattr(args, 'mtp_num_layers', None):
            raise ValueError('--lora-ffn-layers does not currently support MTP.')
        if getattr(args, 'lu_lora_final_layer_index', None) is not None:
            raise ValueError('--lora-ffn-layers does not currently support LU-LoRA.')
        unsupported_flags = (
            'use_fused_mlp', 'moe_alltoall_overlap_comm', 'moe_allgather_overlap_comm', 'moe_fb_overlap',
        )
        for flag in unsupported_flags:
            if getattr(args, flag, False):
                raise ValueError(f'--lora-ffn-layers does not currently support --{flag.replace("_", "-")}.')

        from mindspeed_llm.tasks.posttrain.lora.layer_selection import parse_ffn_lora_layers

        args.lora_ffn_layers = parse_ffn_lora_layers(
            args.lora_ffn_layers, args.num_layers, getattr(args, 'noop_layers', None),
        )
