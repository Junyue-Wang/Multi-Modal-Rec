# MBD Lite：多模态银行事件序列教程

这是一套面向个人电脑的教学版项目，参考官方 [Multimodal Banking Dataset (MBD)](https://arxiv.org/abs/2409.17587) 与 [官方代码仓库](https://github.com/Dzhambo/MBD)，演示如何用交易（`trx`）、地理活动（`geo`）和对话嵌入（`dialog`）预测下一报告月的四个产品标签。

本项目不是官方结果的轻量复现。目标是把官方的实验逻辑拆成可读、可修改、可在普通 CPU 上运行的 notebooks，同时严格保留几个关键原则：

- 按客户的官方 `fold` 划分，避免同一客户跨训练/测试集；
- 每个报告月只使用该月开始前的事件，避免时间穿越；
- 任务是四标签分类，主指标是四个标签的平均 ROC-AUC；
- 分别比较单模态、Early Fusion、Late Fusion 和 Blending；
- 明确记录缺失模态，不把“没有事件”误当成普通数值零。

## 先选运行模式

| 模式 | 数据 | 典型硬件 | 预计用途 |
|---|---|---|---|
| `synthetic`（默认） | 自动生成的小型同构数据 | 8 GB RAM、CPU | 先跑通全部 notebook，约 10–30 分钟 |
| `local` | 已下载并解压的 MBD-mini | 16–32 GB RAM、CPU；建议 SSD | 在真实数据抽样上训练聚合特征模型 |
| `hf` | 从 Hugging Face 下载 MBD-mini | 至少约 10 GB 可用磁盘，建议 32 GB RAM | 首次获取真实数据；下载后转为 `local` |
| Colab 可选版 | 处理后的抽样数据 | T4/L4 GPU | PyTorch Transformer / cross-attention；LLM 路线 |

真实数据在 notebook 00 中默认随机抽取 **3,000 个 client**，保留这些客户在交易、地理、对话和标签表中的全部记录，不再按事件数二次抽样。原始子集保存到 `data/raw/mbd-mini-3000-clients/`，之后会自动复用；notebook 01 只读取该子集并处理特征。HF 首次下载仍需获取完整 MBD-mini 压缩档（约 3.38 GB，总中文件 `detail.tar.gz` 约 1.97 GB）；全量 MBD 约 69 GB。

### MBD-mini、MBD-small 与本项目的 MBD Lite

- **MBD-mini** 是 Hugging Face 上单独发布的约 10% 客户样本（约 10 万客户），结构与完整 MBD 一致；本教程的真实数据入口使用它。
- **MBD-small** 是论文为了资源受限实验构造的 69,474 客户平衡子集：保留最后一个月有任一正标签的客户，再按约 1:10 加入全负客户。它不是 “mini” 的同义词。
- **MBD Lite** 是本教程名称，不是官方数据版本。默认 synthetic 数据只用于跑通流程；真实结论请使用 MBD-mini 并报告抽样规则。

## 安装

本机需要 Python 3.10–3.12。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

项目没有 `mbd_lite` 等私有 Python 包。数据处理、建模和评估示例直接写在 notebook 中，只依赖常见开源库；复制 notebook 时改数据路径和列名即可。

如果只想使用真实数据但不下载 `detail.tar.gz`，可以手动把已解压的数据放到 `data/raw/mbd-mini/`。00 notebook 负责识别原始 Parquet/CSV、抽取客户并保存标准化的五张 Parquet；01 只读取这些准备好的表。

## Notebook 顺序

1. `00_setup_and_data.ipynb`：环境检查、合成数据、可选的 Hugging Face 下载。
2. `01_processing.ipynb`：读取 00 里准备好的数据，逐步展示时间筛选、特征聚合与保存，输出 `data/processed/features.parquet`。
3. `02_eda.ipynb`：标签不平衡、事件长度、模态缺失和时间覆盖。
4. `03_unimodal_baselines.ipynb`：逐模态训练、预测概率和计算指标。
5. `04_early_fusion.ipynb`：监督式 CrossTransformer；各模态 token 在分类前通过 cross-attention 交换信息。
6. `05_late_fusion.ipynb`：分模态得到 PCA embedding，拼接表示后训练 XGBoost 分类器。
7. `06_blending.ipynb`：三个单模态 XGBoost 的后验概率加权；验证集选权重，测试折只用于最终报告。
8. `07_evaluation.ipynb`：逐步比较 ROC-AUC、PR-AUC、缺失模态切片，并导出结果。
9. `08_transformer_baseline.ipynb`：标准 TransformerEncoder；先理解模态 token 与 self-attention。
10. `09_cross_transformer.ipynb`：显式 Query 与 Key/Value 分离的 cross-attention 教学版。
11. `10_optional_llm_colab.ipynb`：可选的事件文本序列化与 LLM embedding 路线。

基础流程请按 00–07 顺序运行。08–09 会把 Transformer 预测写入相同的 `artifacts/predictions/`；完成后重新运行 07，即可把它们加入统一模型比较。10 完全可选。

## 与官方实现的差异

| 环节 | 官方论文/仓库 | MBD Lite |
|---|---|---|
| 数据规模 | MBD/MBD-small，分布式处理 | MBD-mini 或本地抽样；默认合成 smoke data |
| Processing | PySpark + PyTorch LifeStream | pandas + pyarrow，读取时按客户过滤，再按月聚合 |
| 单模态 | Aggregation、CoLES、TabBERT、TabGPT、Supervised GRU | 聚合统计 + Logistic Regression；可切换 XGBoost |
| Early Fusion | 监督式 CrossTransformer 跨模态注意力 | 04 用标准 PyTorch 对三个聚合模态 token 做 cross-attention；09 进一步拆解 attention 过程 |
| Late Fusion | 独立序列编码器的 embedding 拼接后训练分类器；官方仓库提供 LightGBM 代码 | 每个模态独立 scaler/PCA 后拼接，再用 XGBoost 分类；融合位置与官方一致 |
| Blending | 独立单模态分类器的后验概率加权 | 三个单模态 XGBoost 独立训练，只在验证集搜索非负且和为 1 的权重 |
| 验证 | 官方五折、较大训练资源 | 教学 notebook 默认固定测试折；统一评估可切换五折 |
| 深度学习 | Lightning/Hydra、V100、服务器 | 标准 PyTorch Transformer 与显式 cross-attention 示例 |
| LLM | 事件文本序列化、较大模型 embedding + PCA | 给出序列化接口与资源说明，不把 LLM 结果伪装成可在普通 CPU 复现 |

因此，本项目的绝对分数不能与论文表格直接比较；它适合学习流程、验证融合逻辑和快速做消融实验。

## 资源建议

- 普通本地 CPU：运行 00–03、05–07；04 在 3,000-client 数据上也可运行，但会比树模型慢。
- 轻量 GPU（8 GB）：可运行 04、08–09 的 Transformer/CrossTransformer；合成/抽样数据无需 Colab。
- Google Colab T4/L4：真实 MBD-mini 的 Transformer、cross-attention、GRU 或事件级模型。
- 32 GB 以上 GPU + 大内存服务器：接近官方 CoLES/TabGPT/监督 GRU 设置。
- LLM 路线：官方将事件序列化为文本，单模态最多 256 笔交易，多模态使用 128 笔交易 + 64 个 geo 事件，并对 embedding 做 PCA。该路线涉及模型下载、显存和较长推理时间，建议 Colab Pro 或专用 GPU。

## 结果文件

```text
data/processed/features.parquet        # 一行 = 客户 × 报告月
artifacts/predictions/*.parquet        # client_id/report_month/fold/y_*/p_*
artifacts/metrics/model_comparison.csv # 最终模型比较
artifacts/figures/*.png                # EDA 与评估图
```

## 常见问题

- **AUC 报错只有一个类别**：小样本某个标签在测试折中可能全为 0；评估器会把该标签记为 `NaN`，均值只汇总有效标签。
- **真实数据内存不足**：在 01 中降低 `MAX_CLIENTS`、缩短 `LOOKBACK_DAYS`，并保持 `ENGINE="pyarrow"`；不要一次把三个大事件表全部读入内存。
- **找不到真实数据列**：先在 01 中打印 `RawTables` 各表的列名，再给 `load_mbd_tables(..., column_map=...)` 传映射。
- **结果看起来过高**：确认所有事件都满足 `event_time < report_month`，且没有把 `target_*`、未来月份统计或测试折用于权重搜索。

## 数据与许可

MBD/MBD-mini 数据集由原作者发布为 CC BY 4.0。使用真实数据或发布衍生结果时，请引用原论文，并遵守 [MBD-mini 数据卡](https://huggingface.co/datasets/ai-lab/MBD-mini) 的许可说明。本教程代码不重新分发数据。


