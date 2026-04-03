# 基于Transformer的文本摘要生成系统

这是一个中文文本摘要生成系统原型项目，目标是完成一个“可运行、可训练、可演示”的摘要系统。

## 1. 项目做什么

本项目实现了一个文本摘要生成系统，输入一段较长中文文本，输出一段更短、更聚焦核心信息的摘要。

系统包含三层能力：

- Transformer 生成式摘要主流程
- 长度可控摘要
- 异常恢复与基础摘要能力

其中，Transformer 是论文的核心算法；基础摘要能力只用于系统兜底和调试，不是论文重点。

## 2. 使用语言和技术栈

- 编程语言：Python
- 深度学习框架：PyTorch
- 预训练模型框架：HuggingFace Transformers
- Web 接口：FastAPI + Uvicorn
- 数据格式：JSONL

> 如果设备没有安装 `torch` 或 `transformers`，系统会退回到一个基础的抽取式摘要流程，保证项目至少能演示和排错，即`fallback`。

## 3. 项目目录

- `src/summarizer_app/engine.py`：摘要主引擎，优先调用 Transformer
- `src/summarizer_app/fallback.py`：基础抽取式摘要兜底
- `src/summarizer_app/train.py`：训练入口
- `src/summarizer_app/evaluate.py`：评价指标
- `src/summarizer_app/api.py`：FastAPI 接口
- `src/summarizer_app/cli.py`：命令行入口
- `data/demo.jsonl`：样例数据
- `run_cli.py`：命令行启动文件
- `run_api.py`：API 启动文件

## 4. 底层逻辑

### 4.1 总体流程

系统的处理链路是：

1. 接收用户输入的长文本
2. 对文本做清洗和归一化
3. 优先调用 Transformer 进行摘要生成
4. 根据目标长度调整生成参数
5. 如果主模型不可用，则自动切换到基础抽取式摘要
6. 返回摘要结果和当前使用的后端类型

### 4.2 Transformer 部分的逻辑

Transformer 通过自注意力机制建模句子内部和句子之间的关系，适合处理文本摘要这种“输入长、输出短”的任务。

在本项目中，Transformer 主流程做了这几件事：

- 将原文交给 tokenizer 转成模型可处理的 token
- 使用预训练 seq2seq 模型生成摘要
- 通过 `max_new_tokens`、`min_new_tokens`、`num_beams`、`no_repeat_ngram_size` 等参数控制生成结果
- 根据目标长度动态调整输出长度


### 4.3 基础抽取式兜底逻辑

兜底模块不使用深度学习，而是用 Python 完成：

- 句子切分
- 词频统计
- 句子打分
- 选取最重要的若干句拼接成摘要

它的作用是：

- 让系统在依赖缺失时仍能运行
- 方便本地调试
- 便于演示接口和系统结构

只是工程稳定性设计。

## 5. 数据格式

训练和测试数据统一使用 JSONL，每行一个样本：

```json
{"article":"原文","summary":"摘要"}
```

字段说明：

- `article`：输入原文
- `summary`：参考摘要

## 6. 如何安装和运行

### 6.1 建议先创建虚拟环境

建议使用 Python 3.12，并在 `code` 目录下创建独立虚拟环境：

```powershell
cd E:\sdafdad\Desktop\毕业论文\code
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 6.2 命令行摘要

```powershell
python run_cli.py summarize --text "这里是一段较长文本" --target-length 120
```

输出会显示摘要正文和当前后端，例如：

```text
摘要内容
[backend=transformer]
```

如果显示 `fallback`，说明 Transformer 没有成功加载，系统自动退回到基础摘要。

### 6.3 启动 API

```powershell
python run_api.py
```

如果 8000 端口被占用，可以先把 `run_api.py` 里的端口改成 8888，再启动。

启动后访问：

- `http://127.0.0.1:8888/docs`
- `http://127.0.0.1:8888/health`

### 6.4 调用 API

请求 `POST /summarize`，请求体示例：

```json
{
  "text": "这里是一段较长的中文文本",
  "target_length": 120
}
```

如果文本里有换行，要写成合法 JSON，不能直接把原始换行塞进字符串。正确做法是：

- 用 Swagger 界面直接提交
- 或用 `Invoke-RestMethod`
- 或在 curl 中把换行写成 `\n`
  
> 测试：DINER是一个大规模的真实中文数据集，由北京大学王选计算机技术研究所创建，旨在通过识别菜名中的食物、动作和口味组合来评估组合泛化能力。数据集包含3,803种菜名和223,581条对应的食谱，涉及丰富的语言现象如指代、省略和歧义。创建过程中，数据集通过最大复合分布差异(TMCD)方法进行分割，以确保训练和测试集的分布差异最大化。DINER数据集的应用领域主要集中在自然语言处理和机器学习，特别是在菜名识别和组合泛化能力的评估上，为模型提供了挑战性的任务和丰富的语言现象分析。
## 7. 如何训练 Transformer

训练脚本在 `src/summarizer_app/train.py`。

### 7.1 准备数据

准备 `train.jsonl` 和 `valid.jsonl`，格式如下：

```json
{"article":"原文","summary":"摘要"}
```

### 7.2 启动训练

```powershell
python run_cli.py train --train-path data\train.jsonl --valid-path data\valid.jsonl --output-dir outputs\mt5_finetuned
```

### 7.3 训练后推理

训练完成后，直接把模型路径传给 `--model-name`：

```powershell
python run_cli.py summarize --model-name outputs\mt5_finetuned --text "这里是一段较长文本" --target-length 120
```

## 8. 评价方式

本项目提供的评价逻辑包括：

- ROUGE-1
- ROUGE-2
- ROUGE-L
- 长度达标率

这些指标主要用于验证：

- 摘要和参考答案是否接近
- 摘要长度是否符合目标

## 9. 运行时可能遇到的问题

### 9.1 `uvicorn` 找不到

原因通常是：

- 安装依赖时与运行脚本时用的是 Python 版本不一致

解决办法是：安装和运行都统一使用同一个解释器。

### 9.2 服务 端口被占用

原因是已有服务占用了该端口。

解决办法：

- 改成 8888 等其他端口
- 或关闭占用端口的进程

### 9.3 API 返回 `422 JSON decode error`

原因是请求体里的文本包含未转义换行，导致 JSON 非法。

解决办法：

- 使用合法 JSON
- 或让换行写成 `\n`
- 或直接在 Swagger 页面里提交

### 9.4摘要内容[backend=`fallback`]而不是transformer

检查是否报错`SentencePieceExtractor requires the protobuf library`/`SentencePieceExtractor requires the protobuf library but it was not found`

- 使用的 **mt5-small 模型** 需要 `protobuf` 这个库来加载分词模型。

- `pip install protobuf`安装即可

