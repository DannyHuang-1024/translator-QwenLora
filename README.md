# Qwen3 LoRA 翻译微调工程

这个工程把 Qwen3 小模型微调成英译中的翻译模型，默认使用
`Qwen/Qwen3-0.6B`（Qwen3 官方小模型的 Hugging Face 标识是 0.6B；如果你
已有其他兼容 checkpoint，只需修改配置中的 `model_name_or_path`）。训练使用
4-bit 量化 + LoRA，适合 Colab L4，A100 也可以直接运行。

## 目录

```text
configs/train.yaml                 训练和 Drive 配置
src/qwen_translation/data.py       数据集加载、聊天模板和 label mask
src/qwen_translation/train.py      LoRA 训练与断点自动恢复
src/qwen_translation/generate.py   加载 adapter 做翻译
scripts/mount_drive.py             Colab Drive 挂载
scripts/check_env.py               训练前环境检查
run.sh                              Colab 一键启动脚本
```

## Colab 运行

1. 在 Colab 中选择 GPU（优先 L4，其次 A100）。
2. 在 Notebook 的 Python 单元格中先挂载 Google Drive：

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. 将本项目上传或 clone 到 `/content/translator-QwenLora`。例如在 Notebook 单元格中执行：

```bash
!git clone <你的GitHub仓库地址> /content/translator-QwenLora
```

4. 在 Colab Notebook 单元格或 Colab 终端执行：

```bash
cd /content/translator-QwenLora
bash run.sh
```

Drive 必须先在 Notebook 单元格中挂载；`run.sh` 本身不再尝试从 shell 子进程调用
`drive.mount`，因为这种调用没有 Colab Notebook kernel 上下文。首次执行会安装依赖。
模型缓存、checkpoint、优化器状态、训练指标和 TensorBoard 日志
和最终 adapter 都写到 `MyDrive/qwen3-translation/`，训练意外中断后再次执行
`bash run.sh` 会自动找到最新 checkpoint 并续训。默认只取 20,000 条训练样本和
1,000 条验证样本，先用于确认流程无报错；确认成功后再在 `configs/train.yaml`
中增加样本量。

训练日志位于同一个训练目录下：

```text
metrics.jsonl   每次 logging/evaluation 的 step、epoch、loss、learning rate
logs/           TensorBoard event 文件
checkpoint-*/   模型、优化器、scheduler、随机状态和 trainer_state.json
```

在 Colab Notebook 中查看 TensorBoard：

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/qwen3-translation/checkpoints/qwen3-0.6b-en-zh-lora/logs
```

`metrics.jsonl` 是追加写入的，断点续训后仍会保留之前的记录。

终端只显示训练开始、评估、checkpoint 保存和训练结束等关键节点；逐步 loss、学习率
等详细指标不会逐条刷屏，而是完整写入 `metrics.jsonl` 和 TensorBoard。

当前工作区没有可调用的 `google-colab` 远程实例创建接口，因此 `run.sh` 是
Colab 运行时入口，而不是本地远程调用器。Drive 挂载需要由 Colab Notebook 单元格
完成，之后 `run.sh` 负责依赖安装和训练。

## 生成测试

训练结束后，在同一个 Colab 终端执行：

```bash
PYTHONPATH=src python -m qwen_translation.generate \
  --adapter /content/drive/MyDrive/qwen3-translation/checkpoints/qwen3-0.6b-en-zh-lora/final \
  --text "Machine learning helps people solve difficult problems."
```

## 本地检查

本地不需要下载模型也可以执行静态检查：

```bash
python3 -m compileall -q src scripts
```

真正的训练需要 Colab GPU、Hugging Face 模型/数据集下载权限和 Google Drive；
`run.sh` 不会在不具备这些条件的本地机器上误启动长任务。
