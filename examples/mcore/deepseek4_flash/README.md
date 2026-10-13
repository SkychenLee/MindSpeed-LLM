# DeepSeek-V4模型训练

2026年4月24日，DeepSeekV4系列模型DeepSeekV4-Flash和DeepSeekV4-Pro正式发布并开源，其以先进的 MoE 架构在参数效率与推理性能上实现了重要突破。MindSpeed LLM 目前已实现了 **DeepSeekV4-Flash模型** 的定长数据场景下的预训练支持，并同步开放源代码。目前提供预训练的实践参考，帮助用户快速上手。

**注：当前实现为preview版本，部分场景存在限制，后续我们将持续完善并同步跟进 DeepSeek-V4 技术报告中的演进方向：**

<table style="text-align: center">
  <thead>
    <tr>
      <th style="text-align: center;">场景</th>
      <th style="text-align: center;">特性</th>
      <th style="text-align: center;">支持情况</th>
      <th style="text-align: center;">备注</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">训练场景</td>
      <td>预训练/续训</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>全参微调</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td>Lora微调</td>
      <td>DOING</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3">数据场景</td>
      <td>定长数据</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>pack数据</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td>变长数据</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td rowspan="4">切分策略</td>
      <td>TP</td>
      <td>OK</td>
      <td>支持TP=2</td>
    </tr>
    <tr>
      <td>PP</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>EP</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>CP</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td rowspan="1">功能特性</td>
      <td>Muon优化器</td>
      <td>DOING</td>
    </tr>
  </tbody>
</table>

## 安装指导

请参考仓库 [MindSpeed LLM软件安装](../../../docs/zh/pytorch/training/install_guide.md)文档配置环境和拉取仓库代码

## 权重转换

1. 权重下载

    从 [huggingface](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) 下载权重和配置文件

2. 权重转换

    开源DeepSeekV4-Flash权重为FP8 mixed数据格式，训练前需要对原始权重做反量化后获得bf16格式的权重，反量化方法请参考下述脚本

    ```sh
    cd MindSpeed-LLM
    bash examples/mcore/deepseek4_flash/ckpt_dequant_deepseek4_fp8_to_bf16.sh

    ```

    MindSpeed LLM提供[脚本](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash)将已经huggingface开源权重转换为mcore权重，用于训练、推理、评估等任务。
    使用方法如下，请根据实际需要的TP/PP等切分策略和权重路径修改权重转换脚本

    ```sh
    cd MindSpeed-LLM
    bash examples/mcore/deepseek4_flash/ckpt_convert_deepseek4_flash_hf2mcore.sh # 转换时，请指定路径到反量化后的权重
    ```

## 数据预处理

MindSpeed LLM提供[脚本](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) 进行数据集处理

使用方法如下，请根据实际需要修改以下参数

```sh
cd MindSpeed-LLM
bash examples/mcore/deepseek4_flash/data_convert_deepseek4_pretrain.sh
```

| 参数名  | 含义                |
|--------|-----------------|
| --input | 数据集路径  |
| --tokenizer-name-or-path | 模型tokenizer目录    |
| --output-prefix | 数据集处理完的输出路径及前缀名  |

## 预训练

MindSpeed LLM提供[脚本](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) 进行模型训练

```sh
cd MindSpeed-LLM
bash examples/mcore/deepseek4_flash/pretrain_deepseek4_flash_4k_A3_ptd.sh
```

用户需要根据实际情况修改脚本中以下变量

  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | CKPT_SAVE_DIR | 训练中权重保存路径  |
  | DATA_PATH | 数据预处理后的数据路径  |
  | TOKENIZER_PATH | tokenizer目录  |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，为初始加载的权重，如无初始权重则随机初始化  |

## 全参微调

MindSpeed LLM提供[脚本](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) 进行模型训练

```sh
cd MindSpeed-LLM
bash examples/mcore/deepseek4_flash/tune_deepseek4_flash_4k_A3_ptd.sh
```

用户需要根据实际情况修改脚本中以下变量

  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | CKPT_SAVE_DIR | 训练中权重保存路径  |
  | DATA_PATH | 数据预处理后的数据路径  |
  | TOKENIZER_PATH | tokenizer目录  |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，为初始加载的权重 |

## LoRA微调

MindSpeed LLM提供[脚本](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) 进行模型训练

```sh
cd MindSpeed-LLM
bash examples/mcore/deepseek4_flash/lora_finetune_deepseek4_flash_4k_A3_ptd.sh
```

用户需要根据实际情况修改脚本中以下变量

  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | CKPT_SAVE_DIR | 训练中权重保存路径  |
  | DATA_PATH | 数据预处理后的数据路径  |
  | TOKENIZER_PATH | tokenizer目录  |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，为初始加载的权重 |

注：LoRA微调使用指令数据集，需先执行 `examples/mcore/deepseek4_flash/data_convert_deepseek4_instruction.sh` 完成数据预处理，再将产物前缀填入 `DATA_PATH`。

### 选择参与 LoRA 的 FFN 层

可用 `--lora-ffn-layers` 指定参与 LoRA 的 FFN 层，示例脚本通过可选环境变量 `LORA_FFN_LAYERS` 传入。未设置或环境变量为空时，保持原来的所有 FFN 层 LoRA 行为。

层号为**整个模型的全局零基编号**，不随 PP/VPP 切分重新编号。当前脚本设置 `--num-layers 44 --noop-layers 43`，因此有效 FFN 层为 `0-42`；显式选择空层 `43` 会报错。区间包含两端，步长从区间起点开始计算。

| 选择值 | FFN LoRA 范围 |
|--------|---------------|
| 不设置 | 保持默认，所有有效 FFN 层 |
| `30-42` | 最后 13 个有效 FFN 层 |
| `0-42:3` | `0,3,6,...,42`，共 15 层 |
| `0,7,14,21,28,35,42` | 显式列出的 7 层 |
| `all` | 所有有效 FFN 层，自动排除空层 |
| `none` | 所有 FFN 层均不加 LoRA，仍训练原有 Attention LoRA |

例如，只训练最后 13 个 FFN 层的 LoRA：

```sh
LORA_FFN_LAYERS=30-42 bash examples/mcore/deepseek4_flash/lora_finetune_deepseek4_flash_4k_A3_ptd.sh
```

选择同时作用于每层的路由专家和共享专家。选中层为路由专家添加 LoRA，并保留 `--lora-target-modules` 中匹配的共享专家 FC1/FC2 LoRA；未选层不创建 FFN LoRA，基座参数仍冻结。Attention 的 LoRA 目标保持不变；因此 `none` 也不会关闭 Attention LoRA。启动日志会输出所选全局 FFN 层号和可训练参数统计，便于核对。

当前选层功能限定为 DeepSeek-V4 的全 MoE、grouped GEMM 路径（`--first-k-dense-replace -1 --moe-layer-freq 1 --moe-grouped-gemm`）。启用选层时须保留至少一个全局 Attention 目标，例如 `linear_q`，以保证没有选中 FFN 的 PP 阶段仍有可训练参数；暂不支持与 MTP、LU-LoRA、`--use-fused-mlp`、MoE alltoall/allgather 通信重叠或前后向重叠组合使用。这些组合会明确报错。

加载 LoRA checkpoint 继续训练时，应保持相同的 FFN 选择、LoRA 目标模块和并行配置。选层控制流已通过 CPU 测试，尚未完成 NPU 训练验证；仅减少 FFN LoRA 层数不保证提速或同比例减少激活显存，因为各层 Attention LoRA 仍需要反向传播。

本地无需 NPU、PyTorch 或 Megatron，在仓库根目录安装 `pytest` 后即可运行选层测试：

```sh
python -m pytest -q tests/unit_tests/test_lora_layer_selection.py tests/unit_tests/test_lora_layer_integration.py
```

测试覆盖参数解析、全局层号传递、专家构造、PEFT 目标筛选和 checkpoint 加载逻辑；硬件依赖使用测试桩替代，因此不验证实际算子、数值梯度或多卡通信。
