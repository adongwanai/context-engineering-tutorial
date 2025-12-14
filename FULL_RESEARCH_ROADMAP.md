# MetaCogRAG 完整研究路线图 - 顶会投稿版

**目标会议**: **ACL/EMNLP 2025** (4-6周快速通道) | NeurIPS 2025 / ICLR 2026 (6-12月完整版)
**当前日期**: 2025-11-27
**项目整体进度**: 20% (初步验证阶段)

---

## 🎯 发表策略更新 (2025-12-02) ⚠️ 基于最新竞品分析修订

### 🚨 关键警告：创新性声明需大幅调整
**最新发现** (2025-12-02联网检索):
- **TARG** (Nov 2025): 已实现训练免调自适应RAG，减少检索70-90%
- **SeaKR** (June 2024): 已实现多维不确定性融合
- **UncertaintyRAG** (Oct 2024): 已实现token/span级不确定性
- **CONFLARE** (Apr 2024): 已实现校准的保形预测框架

**结论**: 你的"首个XXX"声明**全部存在风险**，必须重新定位为"统一校准的多维不确定性控制框架"

### 推荐路径：EMNLP Findings / Industry Track (现实选择)
- **目标会议**: EMNLP 2025 Findings (6月截稿) / ACL 2025 Industry Track (2月截稿)
- **时间窗口**: 6-8周完成核心实验
- **预期接受率**: 30-40% (如果性能达标 + 定位准确)
- **优势**: 接受工程贡献 + 成本-质量权衡分析
- **关键要求**:
  - ✅ 性能至少达到baseline水平 (当前EM=11% **远未达标**)
  - ✅ 展示等质量下的成本降低曲线 (缺失)
  - ✅ 4+数据集一致性验证 (仅有NQ)
  - ✅ 与TARG/SeaKR等2024-2025新基线对比 (缺失)

### 备选路径：KDD/SIGIR/CIKM (工业应用导向)
- **目标会议**: KDD Applied DS Track / SIGIR Short / CIKM 2025
- **时间窗口**: 8-10周
- **预期接受率**: 40-50% (强调系统工程 + 成本优化)
- **优势**: 重视生产部署、效率收益、实用价值
- **降级原因**: 顶会(NeurIPS/ICLR/ACL主会)要求SOTA性能+强理论，当前不满足

### ❌ 不推荐：NeurIPS/ICLR/ACL主会
- **原因1**: 性能严重不达标 (EM=11% vs baseline 25-30%)
- **原因2**: 创新性声明与TARG/SeaKR直接冲突
- **原因3**: 缺少理论证明 (ECE/MCE校准、收敛性)
- **预期接受率**: < 5%

---

## 📊 真实进度评估

### 当前完成度: 25% (Updated 2025-12-08) ⬆️ +5%

```
顶会论文完整进度:
├─ 理论与方法设计     [████████████░░░░░░░░]  60% 🚧
├─ 核心代码实现       [███████████░░░░░░░░░]  55% ✅ (+5% 修复完成)
├─ 基线复现与对比     [████░░░░░░░░░░░░░░░░]  20% 🚧
├─ 完整实验验证       [███░░░░░░░░░░░░░░░░░]  15% ✅ (+5% 脚本优化)
├─ 消融实验(Ablation) [░░░░░░░░░░░░░░░░░░░░]   0% ⏸️
├─ 深度分析           [█░░░░░░░░░░░░░░░░░░░]   5% ✅ (Recall Audit完成)
├─ 论文撰写           [░░░░░░░░░░░░░░░░░░░░]   0% ⏸️
├─ 实验补充与revision [░░░░░░░░░░░░░░░░░░░░]   0% ⏸️
└─ 最终投稿准备       [░░░░░░░░░░░░░░░░░░░░]   0% ⏸️

总进度: █████░░░░░░░░░░░░░░░ 25%
```

**最新成果** (2025-12-08):
- ✅ **代码修复完成**: 所有TypeError和AttributeError已解决
  - FlashRAGDataset适配：创建TempItem类支持qa_answer_post_process_v3
  - L0-L3字典安全访问：`res.get('l1_retrieval') or {}`模式
  - 检索统计优化：基于docs非空判断而非mode字段
- ✅ **配置优化**: L0阈值0.6→0.3，L1阈值0.3→0.25，双卡tensor_parallel_size=2
- ✅ **统计增强**:
  - L0决策分布（need_retrieval vs skip_retrieval）
  - result_item增加retrieved、num_docs字段
  - metacog_stats.json增加l0_decision_distribution、baseline_comparison
- ✅ **对比输出**: 详细的vs Naive RAG对比分析（EM/F1/检索率/成本节省）
- ⏳ **实验进行中**: NQ 100样本验证运行中，等待结果

**之前成果** (2025-11-27):
- ✅ NQ 200样本实验完成: EM=11%, F1=24.7% (性能不达标)
- ✅ 检索召回审计完成: BGE Recall@5=61%, Recall@20=78%
- ✅ L0 Gating效率验证: 56%查询无需检索，成本降低显著
- ✅ Prompt优化完成: 增强文档读取指令

---

## 🔍 发表可行性评估 (Publication Readiness Assessment)

### 当前状态诊断

**✅ 潜在优势** (需实验验证):
1. **系统性统一**: 整合token级、自洽性、语义熵三种不确定性信号 (vs 单一指标)
2. **训练免调**: 无需微调，即插即用部署 (与TARG类似，但需证明差异)
3. **在线自适应**: L3反馈循环动态调整阈值 (vs 固定阈值，需证明域偏移鲁棒性)
4. **四层架构**: L0-L3协同决策框架已实现
5. **检索效率**: L0 Gating在NQ上减少56%检索 (需等质量验证)

**❌ 致命缺陷** (阻碍发表):
1. **性能严重不达标**: EM=11% vs Naive RAG 25-30%, Self-RAG 36%, **低于所有baseline**
2. **缺少关键证据**: 无校准分析(ECE/MCE)、无等质量成本曲线、无帕累托前沿
3. **实验覆盖不足**: 仅1个数据集(NQ)、1个基线(Naive RAG)、0个消融
4. **创新性声明过强**: "首个XXX"与TARG/SeaKR/UncertaintyRAG直接冲突
5. **缺少2024-2025新基线**: 未对比TARG、SeaKR、CONFLARE、UncertaintyRAG
6. **无跨数据集一致性**: 无法证明方法泛化能力

### 创新点差异化分析 (2025-12-02更新)

**⚠️ 重要**: 添加2024-2025最新竞品对比

| 维度         | Self-RAG | Adaptive-RAG | **TARG (Nov 2025)** | **SeaKR (Jun 2024)** | **MetaCogRAG (Ours)** |
| ---------- | -------- | ------------ | ------------------- | -------------------- | --------------------- |
| **检索决策**   | 生成时反思    | 固定分类器        | **Token熵门控**        | **不确定性驱动**           | **多维融合门控**            |
| **不确定性度量** | 单一(生成概率) | 无            | **Margin信号+熵**      | **内部状态提取**           | **Token+SC+SE融合**     |
| **训练需求**   | ✅ 需要     | ✅ 需要         | **❌ 训练免调** ⚠️       | ❌ 不需要                | **❌ 训练免调**            |
| **自适应能力**  | ❌ 固定阈值   | ❌ 固定分类器      | ✅ 阈值优化              | ✅ 动态调整               | **✅ L3在线学习**          |
| **效率优化**   | ❌ 无门控    | ✅ 有门控        | **✅ 减少70-90%** ⚠️   | ✅ 减少检索               | **✅ 减少56%** (需验证)     |
| **校准保证**   | ❌ 无      | ❌ 无          | ❌ 无                 | ❌ 无                  | **✅ ECE/MCE** (待实现)   |
| **反思机制**   | ✅ 生成时    | ❌ 无          | ❌ 无                 | ❌ 无                  | **✅ L2 NLI评估**        |
| **闭环学习**   | ❌ 无      | ❌ 无          | ❌ 无                 | ❌ 无                  | **✅ L3轨迹学习**          |

**关键差异化定位** (修订后，移除"首个"声明):

1. **统一校准的多维不确定性框架** ⭐ 核心创新
   - 整合Token-level + Self-Consistency + Semantic Entropy
   - **差异点**: TARG仅用token熵，SeaKR用内部状态，我们**系统性融合并校准**三种信号
   - **需证明**: ECE/MCE指标显示融合后决策质量优于单一方法
   - **风险**: 必须提供校准证据，否则与TARG/SeaKR无实质区别

2. **成本约束下的质量优化** ⭐ 实用价值
   - 展示等质量下的检索率/成本降低曲线（帕累托前沿）
   - **差异点**: TARG声称减少70-90%，我们需证明在**相同EM/F1下**成本更低或质量更高
   - **需证明**: 至少4个数据集的iso-quality成本曲线
   - **目标**: 在EM=25-30%时，检索率比Always-RAG低40-60%

3. **四层元认知协同 + 在线自适应** 📌 系统架构
   - L0(门控) + L1(主动) + L2(反思) + L3(自适应阈值学习)
   - **差异点**: TARG是单层门控，SeaKR是双阶段，我们是**四层闭环**
   - **需证明**: 消融实验显示L0/L1/L2/L3各层的边际贡献
   - **风险**: 可能被质疑"过度工程化"，需展示架构必要性

4. **训练免调 + 跨域泛化** ✅ 工程优势
   - 无需微调，零样本迁移到新数据集
   - **差异点**: 与TARG类似，但需证明L3自适应带来的**域偏移鲁棒性**
   - **需证明**: 在NQ训练阈值 → 在TriviaQA/HotpotQA零样本测试，校准仍保持
   - **目标**: 跨数据集ECE变化 < 5%

**⚠️ 必须避免的声明** (已被竞品占据):
- ❌ "首个训练免调自适应RAG" → TARG (Nov 2025) 已占据
- ❌ "首个token级动态门控" → TARG、DRAGIN、FLARE已实现
- ❌ "首个多维不确定性驱动" → SeaKR (Jun 2024) 已占据
- ❌ "首个校准RAG" → CONFLARE (Apr 2024) 已实现保形预测

**✅ 安全的定位**:
- ✅ "统一校准的多维不确定性控制框架"
- ✅ "系统性整合token级、自洽性、语义熵并提供校准保证"
- ✅ "四层元认知闭环架构实现在线自适应"
- ✅ "训练免调的跨域泛化能力"

---

## 📋 最小可接受包 (Minimum Acceptable Package) - 2025-12-02修订 🆕 基于GPT-4o建议强化

**⚠️ 基于竞品分析和GPT-4o深度审视，大幅提高实验要求**

**核心战略转变** 🔥:
- **从**: "提升EM性能"（与TARG/SeaKR硬碰）
- **到**: "等质量下的成本最优 + 校准可靠性"（差异化定位）

### 必需实验 (Must-Have) - P0优先级

#### 1. 数据集覆盖 (至少3个，含1个多跳)
- [x] **NQ** (200 samples, EM=11%) ✅ **性能不达标，需修复**
- [ ] **TriviaQA** (1000 samples) - Week 2-3 P0（单跳泛化）
- [ ] **HotpotQA** (500 samples) - Week 2-3 P0 🔥（多跳推理，必须）
- [ ] **PopQA** (500 samples) - Week 3-4 P1 (TARG用此数据集，便于对比)

**新增Nice-to-Have**:
- [ ] **MuSiQue-Ans** (300 samples) - P2 (第二个多跳数据集，增强泛化证明)
- [ ] **2WikiMultiHopQA** (300 samples) - P2 (备选多跳)

#### 2. 多模型验证 (至少2个) 🆕 GPT建议
**目的**: 证明方法的模型无关性，回应审稿人"是否只对特定模型有效"

- [ ] **Qwen2.5-7B-Instruct** - Week 2 P0 (主力模型)
- [ ] **Qwen2.5-14B-Instruct** - Week 3 P0 (更大规模，证明规模泛化)
- [ ] **Qwen3-8B-Instruct** - Week 4 P1 (最新架构，思考模式对比)

**新增Nice-to-Have**:
- [ ] **Llama-3.1-8B-Instruct** - P2 (跨家族验证，证明非Qwen依赖)

**关键指标**: 跨模型决策F1标准差 < 5%

#### 3. 基线对比 (至少4个) ⚠️ 增加2024-2025新基线
- [ ] **Naive RAG** (必须) - Week 1 P0
- [ ] **Self-RAG** (核心竞品) - Week 2 P0
- [ ] **Adaptive-RAG** (核心竞品) - Week 2 P0
- [ ] **TARG** (Nov 2025, **直接竞争对手**) - Week 3 P0 🔥
  - 如代码不可用，至少复现其核心思路（token熵 + margin信号门控）
- [ ] **SeaKR** (Jun 2024, 多维不确定性) - Week 3 P1
  - 如代码不可用，实现简化版（内部状态不确定性）
- [ ] **FLARE** (迭代检索基线) - Week 4 P1

#### 4. 核心消融实验 (至少6个配置)
- [ ] MetaCogRAG (Full) - 已有
- [ ] w/o L0 Gating (Always Retrieve) - Week 2
- [ ] w/o Unified Uncertainty (仅Token-level) - Week 2
- [ ] w/o L2 Reflection - Week 3
- [ ] w/o L3 Feedback (固定阈值) - Week 3
- [ ] **仅Token熵 (复现TARG核心)** - Week 3 🆕
- [ ] **仅内部状态 (复现SeaKR核心)** - Week 3 🆕

#### 5. 关键分析实验 (至少7个) ⚠️ 新增校准、成本、决策质量分析 🆕

##### 5.1 检索质量分析
- [x] **检索召回分析** (BGE Recall Audit) ✅

##### 5.2 校准分析 🔥 核心创新证明
- [ ] **不确定性校准分析** (ECE/MCE curves) - Week 3 P0 🆕
  - **🆕 使用概率margin (p1-p2) 而非原始logit差**
  - **🆕 实现温标/Platt校准**，在dev集学习温度参数τ
  - 对比配置:
    1. Token-only (单一指标，对应TARG)
    2. SC-only (单一指标)
    3. SE-only (单一指标)
    4. Fusion未校准 (我们的方法，未校准)
    5. **Fusion + 温标校准** (核心创新)
  - **目标**: Fusion+校准的ECE < 单一方法 ≥ 5%
  - **跨模型验证**: 在2-3个模型上验证校准改进的一致性

##### 5.3 成本分析 🔥 主打卖点
- [ ] **等质量成本曲线** (Iso-quality cost curves) - Week 3 P0 🆕
  - 固定EM = [25%, 30%, 35%]
  - 对比各方法达到该质量所需的检索率
  - 对比方法: Always-RAG, TARG-core, Self-RAG, MetaCogRAG
  - **目标**: 在EM=30%时，检索率比Always-RAG低 50-65%

- [ ] **帕累托前沿分析** (Pareto frontier) - Week 4 P0
  - X轴: 检索率, Y轴: EM
  - 绘制所有方法的质量-成本权衡曲线
  - **目标**: MetaCogRAG位于帕累托前沿

##### 5.4 决策质量分析 🆕 GPT强调
- [ ] **决策准确率分析** (决策混淆矩阵 + 决策F1) - Week 3 P0 🆕
  - **定义ground truth**: 事后分析，在有/无检索下的EM差异
    - 该检索 = 检索后EM提升 ≥ 5%
    - 不该检索 = 检索后EM提升 < 5%
  - **计算混淆矩阵**:
    - TP: 该检索且检索了（正确触发）
    - TN: 不该检索且没检索（正确跳过）
    - FP: 不该检索但检索了（浪费成本）
    - FN: 该检索但没检索（性能损失）
  - **计算决策指标**:
    - Decision Precision = TP / (TP + FP)
    - Decision Recall = TP / (TP + FN)
    - Decision F1 = 2 * P * R / (P + R)
    - Decision Accuracy = (TP + TN) / Total
  - **目标**: MetaCogRAG的决策F1 > TARG-core **至少2-3%**

##### 5.5 跨域泛化分析
- [ ] **跨数据集校准一致性** - Week 4 P1
  - NQ训练阈值 → TriviaQA/HotpotQA/PopQA测试
  - 计算每个数据集的ECE和决策F1
  - 对比: 固定阈值 vs L3自适应阈值
  - **目标**: L3自适应下的ECE变化 < 固定阈值，ECE跨数据集变化 < 10%（理想 < 5%）

- [ ] **跨模型一致性** - Week 4 P1 🆕
  - 在2-3个模型上验证: Qwen2.5-7B, 14B, (Qwen3-8B)
  - **目标**: 决策F1跨模型标准差 < 5%

#### 6. 论文撰写 (8页ACL格式)
- [ ] Introduction + Related Work - Week 7
- [ ] Method (2-2.5页) - Week 7
- [ ] Experiments (2.5-3页) - Week 7
- [ ] Discussion + Conclusion - Week 7

### 预期性能目标 (Realistic Targets) - 2025-12-02修订 🆕 战略转向"等质量成本最优"

**⚠️ 核心战略调整**: 不再以"超越TARG性能"为主要卖点，而是证明**"等质量下的成本更优 + 校准可靠"**

假设完成所有优化后的**现实目标**（多模型平均）:

#### 表1: 主结果表（EM性能 + 检索率）

| 数据集 | Naive RAG | Self-RAG | Adaptive-RAG | **TARG (2025)** | **MetaCogRAG** | 检索率对比 |
|--------|-----------|----------|--------------|-----------------|----------------|------------|
| NQ | 35% | 36% | 35% | **~36-38%** ⚠️ | **30-35%** ⭐ | 35-45% vs 100% ✅ |
| TriviaQA | 45% | 47% | 46% | **~48-50%** | **42-48%** | 35-45% vs 100% |
| HotpotQA | 30% | 32% | 31% | N/A | **28-33%** | 40-50% vs 100% |
| PopQA | 40% | 42% | 41% | **~43-45%** | **38-43%** | 35-45% vs 100% |
| **Average** | **37.5%** | **39.3%** | **38.3%** | **~42-44%** | **34.5-39.8%** ⭐ | **~38-46%** ✅ |

**注**: ⚠️ TARG性能可能更高，我们不期望在EM上全面超越

#### 表2: 等质量下的检索率对比（核心卖点）🔥

| 固定EM | Always-RAG | TARG-core | Self-RAG | **MetaCogRAG** | 我们的优势 |
|--------|------------|-----------|----------|----------------|------------|
| **25%** | 100% | ~30-40% ⚠️ | ~85% | **20-30%** ⭐ | 比Always低70-80% |
| **30%** | 100% | ~40-50% | ~85% | **35-45%** ⭐ | 比Always低55-65% |
| **35%** | 100% | ~50-60% | ~85% | **45-55%** | 比Always低45-55% |

**关键叙述**:
- 在EM=30%时，MetaCogRAG的检索率为40%（vs Always-RAG 100%），**节省60%成本**
- 可能与TARG检索率相当或略高，但通过**校准和决策质量**证明差异价值

#### 表3: 校准与决策质量（核心创新）⭐

| 方法 | ECE ↓ | MCE ↓ | 决策F1 ↑ | 决策Accuracy |
|------|-------|-------|----------|--------------|
| Token-only (TARG-like) | 0.18 | 0.35 | 0.72 | 0.75 |
| SC-only | 0.16 | 0.32 | 0.74 | 0.77 |
| SE-only | 0.17 | 0.33 | 0.73 | 0.76 |
| **Fusion未校准** | 0.14 | 0.28 | 0.78 | 0.80 |
| **Fusion + 温标校准** ⭐ | **0.10** | **0.22** | **0.82** | **0.84** |

**目标**:
- 融合+校准的ECE < Token-only **至少8个百分点** (0.10 vs 0.18)
- 决策F1 > Token-only **至少10个百分点** (0.82 vs 0.72)

#### 表4: 跨数据集+跨模型一致性（泛化能力）✅

| 模型 | NQ ECE | TriviaQA ECE | HotpotQA ECE | 标准差 | 决策F1标准差 |
|------|--------|--------------|--------------|--------|--------------|
| Qwen2.5-7B | 0.10 | 0.11 | 0.12 | 0.008 | 0.03 |
| Qwen2.5-14B | 0.09 | 0.10 | 0.11 | 0.008 | 0.02 |
| Qwen3-8B (可选) | 0.11 | 0.12 | 0.13 | 0.008 | 0.04 |

**目标**:
- ECE跨数据集变化 < 10% (实际 ~8%)
- 决策F1跨模型标准差 < 5% (实际 ~3%)

---

**关键卖点** (修订后，3个必须全部证明):

### 1. **等质量下的成本优势** 🔥 主打卖点
- 在EM=30%时，检索率比Always-RAG(100%)低**55-65%**
- 提供**等质量成本曲线** (Iso-Quality Curves)，固定EM=[25%, 30%, 35%]
- 提供**帕累托前沿**，证明在质量-成本权衡空间中的最优位置
- **与TARG对比**: 可能检索率相当，但通过校准和决策质量证明优势

### 2. **校准的决策质量** ⭐ 核心创新
- 融合不确定性的ECE < 单一方法(Token-only) **8-10个百分点**
- **实现温标/Platt校准**，统一不同模型的logit尺度
- **使用概率margin (p1-p2)**，而非原始logit差
- **决策F1** (正确判断何时检索) > TARG-core **10个百分点**
- 提供**决策混淆矩阵** (TP/TN/FP/FN)，可解释决策错误类型

### 3. **训练免调 + 跨域跨模型泛化** ✅ 工程优势
- 与TARG类似的训练免调优势
- **L3自适应**提供更强的域偏移鲁棒性（vs TARG固定阈值）
- ECE跨数据集变化 < 10% (vs TARG未提供此证据)
- **多模型验证**: 在2-3个模型(Qwen2.5-7B/14B, Qwen3-8B)上一致
- 决策F1跨模型标准差 < 5%

### 4. **四层架构的系统性** 📌 差异点
- 消融实验证明L0/L1/L2/L3的边际贡献
- **L2反思机制**捕获TARG忽略的后验错误（事后纠正）
- **L3在线学习**适应域偏移（vs 固定阈值）

---

**⚠️ 现实认知** (必须接受):
- **性能可能不如TARG** (TARG声称在NQ/PopQA上减少70-90%检索且保持性能)
- **不能仅凭"性能稍高"取胜**: 需多维度证明(校准+成本+泛化+可解释性)
- **必须证明差异价值**: 校准改进、四层架构必要性、反思机制价值

**最低接受标准** (发表门槛):
- ✅ EM ≥ Naive RAG (30-35%) - **当前11%，严重不达标**
- ✅ 在EM=30%时，检索率 < 50% (vs Always-RAG 100%)
- ✅ ECE < 0.15（良好校准），融合+校准 < 单一方法 ≥ 5%
- ✅ 决策F1 > TARG-core ≥ 2-3%
- ✅ 至少3个数据集一致性 (NQ/TriviaQA/HotpotQA)，含1个多跳
- ✅ 至少2个模型验证 (Qwen2.5-7B + 14B)

---

## 🎯 顶会论文完整需求

### 核心贡献点 (必须全部完成)

#### 1. 理论创新 (Theory & Method)
- [x] **MetaCogRAG 核心定位**: 
  > 现有的 RAG 是“开环”的（检索了就用），Self-RAG 是“半闭环”的（生成时反思），而 MetaCogRAG 是**首个基于多维不确定性感知的全闭环、自进化 RAG 系统**。

- [x] **统一不确定性感知 ("The Eye")** - 已完成
  - *核心创新*: 并非简单的指标堆砌，而是构建了一个多维感知的"眼睛"。
  - Token-level (直觉) + Self-Consistency (推理) + Semantic Entropy (语义) 的深度融合。
  - 关键在于不仅知道"不知道"，还能区分"为什么不知道"（是缺少知识还是推理错误）。

- [ ] **GRPO 训练 (L3)** - Optional/Plan B ⚠️
  - *策略调整*: 优先完成 Training-free (Prompt-based) 版本。如果效果足够好 (超越 Baseline 3-5%)，则作为主要贡献。
  - *降级理由*: RL 训练周期长、风险高。若 Prompt 版本已足够强，GRPO 可作为 Future Work 或加分项，而非必须项。
  - *触发条件*: 仅当 Prompt 版本性能遭遇瓶颈 (< Baseline + 2%) 时启动。

- [x] **基础感知与行动层 (Foundational Layers)** - L0 & L1
  - L0 Gating & L1 Proactive Retrieval: 作为系统的基础执行单元。
  - 重点不在于它们的存在，而在于它们完全受控于"The Brain"的动态调度。

- [ ] **理论分析** - 未开始 (Critical!) ⭐ P0
  - [ ] **定理1 (信息增益分解)**: 证明 $I(Q;D|A) = I_{meta} + I_{active} + I_{reflect}$。
  - [ ] **定理2 (单层次优性)**: 证明单层驱动只能优化局部目标，四层架构达到 Pareto 最优。
  - [ ] **定理3 (GRPO收敛性)**: 证明联合训练在特定条件下收敛到 Nash 均衡。
  - [ ] **认知心理学映射**: 建立 Flavell 元认知理论 (Knowledge/Experience/Regulation) 与 L0-L3 的对应关系。

- [ ] **与已有方法的深度对比** - 待完善
  - vs **ComoRAG (2025)**: 
    - *定位差异*: "Loop" (解决困难问题) vs "Gating" (避免无效计算)。
    - *效率优势*: ComoRAG 即使面对简单问题也需 Tri-Retrieve，我们 L0 可直接跳过 (-40% 检索量)。
    - *成本优势*: ComoRAG 依赖 GPT-4o Prompting; 我们通过 GRPO 使 7B 模型具备元认知。
    - *场景差异*: ComoRAG 专攻 200K+ 长文档; 我们专注通用 RAG 的高并发/低延迟场景。
  - vs Self-RAG: Training-free 的优势。
  - vs Adaptive-RAG: 动态阈值 vs 固定分类器。

#### 2. 实验验证 (Experiments) - **严重不足**
当前状态: 只有失败的10样本测试

**顶会要求的完整实验**:

##### A. 数据集覆盖 (至少4-6个)
- [ ] **问答数据集** (必须)
  - [ ] Natural Questions (NQ) - 3000+ samples
  - [ ] TriviaQA - 2000+ samples
  - [ ] HotpotQA - 1000+ samples (multi-hop)
  - [ ] WebQuestions - 1000+ samples
- [ ] **长文档理解** (推荐)
  - [ ] MS MARCO - document ranking
  - [ ] QASPER - 科学论文QA
- [ ] **开放域生成** (加分项)
  - [ ] ELI5 - 解释性问答
  - [ ] AmbigQA - 歧义问答

##### B. 基线对比 (至少5-8个)
- [x] Naive RAG - 部分完成 (仅HotpotQA)
- [ ] **关键基线** (必须全部复现)
  - [ ] **ComoRAG (2025)** - ⭐ 最强竞品 (循环迭代机制)
  - [ ] Self-RAG (ICLR 2024) - (Reflection机制)
  - [ ] Adaptive-RAG (2023) - (路由机制)
  - [ ] FLARE (2023) - (主动检索)
  - [ ] SmartRAG (2024) - (反馈机制)
- [ ] **强基线** (推荐)
  - [ ] GPT-4o / Claude-3.5-Sonnet with retrieval
  - [ ] DeepSeek-R1 (推理能力对比)

##### C. 消融实验 (Ablation Study)
需要证明每个组件的贡献:

```
配置矩阵 (核心配置):
├─ MetaCogRAG (Inference-only) ⭐ 主推配置
├─ w/o L2 Reflection (即时纠错)
├─ w/o L0 Gating (Always Retrieve) -> 证明效率优势
├─ w/o Unified Uncertainty (单一度量)
└─ Naive RAG baseline

可选配置 (如果时间允许/需要救场):
├─ MetaCogRAG + GRPO (RL Tuned)
└─ w/o L3 Feedback (仅在RL版本中有意义)
```

##### D. 深度分析实验 (Analysis) - **完全缺失**

**必须的分析**:
- [ ] **不确定性校准分析**
  - 不确定性分数 vs 实际准确率的相关性
  - Calibration curves (ECE, MCE指标)
  - 不同不确定性度量的对比
- [ ] **检索决策分析**
  - 检索率 vs 性能的trade-off curve
  - False positive/negative检索分析
  - 最优阈值敏感性分析
- [ ] **效率分析**
  - 推理时间 vs 性能
  - 检索次数 vs 成本
  - 与baseline的效率对比
- [ ] **Case Study**
  - 成功案例分析 (至少10个)
  - 失败案例分析 (至少10个)
  - 不同query类型的表现差异
- [ ] **泛化能力分析**
  - Domain transfer (不同数据集)
  - Model transfer (不同LLM)
  - 数据规模影响 (learning curve)

##### E. 人工评估 (Human Evaluation) - **完全缺失**
对于顶会,自动指标不够:
- [ ] 随机采样200-300个cases
- [ ] 多个标注员评估 (3-5人)
- [ ] 评估维度:
  - Correctness (准确性)
  - Helpfulness (有用性)
  - Fluency (流畅性)
  - Groundedness (基于事实)
- [ ] Inter-annotator agreement计算
- [ ] 与自动指标的相关性分析

#### 3. 论文撰写 (Paper Writing) - **策略调整**

**ACL 8页论文结构建议**:

```
Abstract
├─ 聚焦: Efficiency & Effectiveness (Training-free)
└─ 亮点: "Plug-and-Play", "40% 检索减少"

1. Introduction (1-1.5页)
...

3. Method
├─ ...
└─ 3.3 Dynamic Adaptation (L3) -> 此时作为 Optional 模块介绍，强调其作为 Training-free 框架的可扩展性。

4. Experiments
├─ 主实验: Training-free 版本的 SOTA 表现
└─ 效率分析: 重点展示 L0 Gating 带来的巨大收益 (Time/Cost reduction)
```
├─ 1.1 Motivation
├─ 1.2 Limitations of Existing Work
├─ 1.3 Our Approach
└─ 1.4 Contributions (3-4个bullet points)

2. Related Work (0.75-1页)
├─ 2.1 Retrieval-Augmented Generation
├─ 2.2 Uncertainty Estimation in LLMs
├─ 2.3 Adaptive Retrieval Strategies
└─ 2.4 Meta-Cognitive Approaches

3. Method (2-2.5页) ⭐ 核心
├─ 3.1 Problem Formulation: The "Open-Loop" Problem in RAG
├─ 3.2 "The Eye": Unified Uncertainty Perception
│   ├─ 3.2.1 Multi-dimensional Uncertainty Metrics
│   └─ 3.2.2 Fusion Mechanism (Why it works better)
├─ 3.3 "The Brain": Dynamic Adaptation & Feedback Loop (核心)
│   ├─ 3.3.1 L2 Reflection: Short-term Error Correction
│   └─ 3.3.2 L3 Feedback: Long-term Threshold Evolution
└─ 3.4 Foundational Execution (L0 & L1 Brief)
    └─ How Perception guides Action

4. Experiments (2.5-3页) ⭐ 核心
├─ 4.1 Experimental Setup
│   ├─ Datasets
│   ├─ Baselines
│   ├─ Metrics
│   └─ Implementation Details
├─ 4.2 Main Results
│   ├─ Overall Performance (大表格)
│   ├─ Statistical Significance Tests
│   └─ Efficiency Analysis
├─ 4.3 Ablation Study
└─ 4.4 Analysis
    ├─ Uncertainty Calibration
    ├─ Retrieval Decision Quality
    └─ Case Studies

5. Discussion (0.5页)
├─ Key Findings
├─ Limitations
└─ Broader Impacts

6. Conclusion (0.25页)

References (1-1.5页)

Appendix (补充材料,不计入8页)
├─ A. Additional Results
├─ B. Hyperparameters
├─ C. More Case Studies
├─ D. Failure Analysis
└─ E. Human Evaluation Details
```

#### 4. 代码与可复现性 (Reproducibility) - **部分完成**
- [x] 核心代码实现 - 40%完成
- [ ] **代码整理** - 未开始
  - 清理debug代码
  - 统一代码风格
  - 添加完整注释
  - Type hints和文档
- [ ] **开源准备** - 未开始
  - README编写
  - 安装文档
  - 运行示例
  - 预训练模型/索引发布
  - License选择
- [ ] **可复现性验证** - 未开始
  - 在新环境重新运行
  - 随机种子固定
  - 结果方差报告
  - Docker镜像准备

---

## 📅 完整时间线 (顶会投稿版)

### 阶段1: 紧急修复与验证 (Week 1-2, 当前)

#### Week 1 (11.11-11.17)
**目标**: 让系统跑通,达到baseline水平

- [ ] Day 1-2: 问题诊断与修复
  - 对比Naive RAG找出问题
  - 修复核心bug
  - 10样本达到EM>20%

- [ ] Day 3-4: 初步验证
  - NQ 200样本: EM>25%
  - HotpotQA 100样本: EM>30%
  - 确认方向可行

- [ ] Day 5-7: 代码优化
  - 提升性能到接近baseline
  - 目标: NQ EM>30%, HotpotQA EM>35%
  - 代码重构,准备大规模实验

#### Week 2 (11.18-11.24)
**目标**: 超越baseline,初步证明方法有效

- [ ] Day 1-3: 参数调优
  - Grid search关键超参
  - L0阈值优化
  - 检索top-k优化
  - 目标: 稳定超越baseline 3-5个点

- [ ] Day 4-5: 初步消融实验
  - 核心配置对比 (Full vs w/o L0/L1/L2/L3)
  - 证明架构有效性

- [ ] Day 6-7: 初步论文框架
  - 撰写Method section草稿
  - 准备初步实验表格
  - **里程碑检查点**: 如果结果不好,考虑调整方向

---

### 阶段2: 大规模实验 (Week 3-6)

#### Week 3-4 (11.25-12.08)
**目标**: 完整数据集覆盖

- [ ] 复现所有baseline (Self-RAG, Adaptive-RAG等)
  - 每个baseline 2-3天
  - 确保公平对比

- [ ] 在4-6个数据集上运行完整实验
  - NQ (3000 samples)
  - TriviaQA (2000 samples)
  - HotpotQA (1000 samples)
  - WebQuestions (1000 samples)
  - MS MARCO (optional)
  - QASPER (optional)

#### Week 5-6 (12.09-12.22)
**目标**: 完整消融实验

- [ ] 10+ ablation configurations
- [ ] 每个配置在所有数据集运行
- [ ] 统计显著性检验
- [ ] 结果可视化

**预期成果**: 完整实验表格,证明方法优越性

---

### 阶段3: 深度分析 (Week 7-9)

#### Week 7-8 (12.23-01.05)
**目标**: 深入理解方法

- [ ] 不确定性校准分析
  - Calibration curves
  - 不同度量对比
  - 理论验证

- [ ] 检索决策分析
  - Trade-off curves
  - 最优阈值分析
  - 错误案例分类

- [ ] 效率分析
  - 时间成本测量
  - 与baseline对比
  - 优化建议

#### Week 9 (01.06-01.12)
**目标**: Case study与人工评估

- [ ] 准备200-300个样本
- [ ] 设计评估指南
- [ ] 招募标注员 (3-5人)
- [ ] 完成人工评估
- [ ] 分析结果

---

### 阶段4: 论文撰写 (Week 10-13)

#### Week 10-11 (01.13-01.26)
**目标**: 初稿完成

- [ ] Week 10:
  - Introduction + Related Work
  - Method section完善
  - Experiment section框架

- [ ] Week 11:
  - 实验结果整理
  - 图表制作 (至少10个高质量图)
  - Discussion + Conclusion
  - Abstract撰写

**交付物**: 完整初稿

#### Week 12 (01.27-02.02)
**目标**: 内部审阅与迭代

- [ ] 导师/合作者反馈
- [ ] 重大修改
- [ ] 补充实验 (如果需要)
- [ ] 论文润色

#### Week 13 (02.03-02.09)
**目标**: 最终投稿准备

- [ ] 格式检查 (符合会议模板)
- [ ] 引用完整性检查
- [ ] 补充材料准备
- [ ] 代码开源准备
- [ ] Checklist验证

**截止日期**: 假设会议deadline在2月15日

---

### 阶段5: Rebuttal准备 (如果进入review)

#### Week 14-17 (02.10-03.09)
**目标**: 应对审稿意见

- [ ] 审稿意见分析
- [ ] 补充实验 (reviewer要求)
- [ ] Rebuttal撰写
- [ ] 修改论文 (如果需要)

**审稿周期**: 通常2-3个月

---

## 🎯 关键里程碑与检查点

### Milestone 1: 系统可用 (Week 2结束)
**标准**:
- ✅ EM稳定>25% (NQ)
- ✅ 超越Naive RAG baseline
- ✅ 无系统性bug

**决策**: Go / No-go继续大规模实验

### Milestone 2: 初步证明 (Week 6结束)
**标准**:
- ✅ 4+数据集上稳定表现
- ✅ 超越主要baseline (Self-RAG等) 2-3个点
- ✅ 消融实验证明有效性

**决策**: Go / No-go撰写论文

### Milestone 3: 论文初稿 (Week 11结束)
**标准**:
- ✅ 完整8页论文
- ✅ 所有实验完成
- ✅ 图表齐全

**决策**: Go投稿 / Major revision需要 / 延期投下一个会议

### Milestone 4: 投稿 (Week 13结束)
**标准**:
- ✅ 论文提交
- ✅ 代码开源
- ✅ 补充材料上传

---

## 📊 预期实验结果 (目标)

### 主实验表格 (Table 1: Main Results)

| Method | NQ EM/F1 | TriviaQA EM/F1 | HotpotQA EM/F1 | Avg EM | Retrieval Rate |
|--------|----------|----------------|----------------|--------|----------------|
| No Retrieval | 20.5/28.3 | 25.1/32.4 | 18.2/26.1 | 21.3 | 0% |
| Naive RAG | 29.1/44.8 | 35.2/48.6 | 32.5/46.2 | 32.3 | 100% |
| Self-RAG | 31.5/46.2 | 37.8/50.1 | 34.6/48.5 | 34.6 | 85% |
| Adaptive-RAG | 32.8/47.5 | 39.1/51.3 | 35.9/49.8 | 35.9 | 75% |
| FLARE | 30.9/45.8 | 36.5/49.2 | 33.8/47.6 | 33.7 | 80% |
| **MetaCogRAG** | **35.2/49.8** | **41.5/53.7** | **38.3/52.1** | **38.3** | **68%** |

**要求**:
- 在所有数据集上显著优于baseline (p<0.05)
- 检索率更低 (更高效)
- 平均提升3-5个EM点

### 消融实验表格 (Table 2: Ablation Study)

| Configuration | NQ EM | HotpotQA EM | Avg EM | Δ |
|--------------|-------|-------------|--------|---|
| MetaCogRAG (Full) | 35.2 | 38.3 | 36.8 | - |
| w/o L3 Feedback | 34.5 | 37.6 | 36.1 | -0.7 |
| w/o L2 Reflection | 33.8 | 36.9 | 35.4 | -1.4 |
| w/o L0 Gating (always retrieve) | 32.1 | 35.2 | 33.7 | -3.1 |
| w/ Token-level only | 33.2 | 36.5 | 34.9 | -1.9 |
| w/ Self-Consistency only | 33.6 | 36.8 | 35.2 | -1.6 |
| Fixed threshold | 32.8 | 35.9 | 34.4 | -2.4 |
| Naive RAG | 29.1 | 32.5 | 30.8 | -6.0 |

**证明**: 每个组件都有贡献,L0最关键

### 分析图表 (必须的)

1. **Uncertainty Calibration Curve** (图1)
   - X轴: 预测不确定性分数
   - Y轴: 实际准确率
   - 完美校准线对比

2. **Retrieval Rate vs Performance** (图2)
   - Trade-off curve
   - 展示最优工作点

3. **Performance by Query Type** (图3)
   - 不同类型query的表现
   - 展示自适应能力

4. **Efficiency Analysis** (图4)
   - 时间成本 vs EM
   - 与baseline对比

5. **Ablation Results Visualization** (图5)
   - 各组件贡献可视化
   - Stacked bar chart

6. **Case Study Examples** (图6-7)
   - 成功案例
   - 失败案例

---

## 🚨 主要风险与应对

### Risk 1: 性能不如预期 (High)
**现状**: EM=0%, 远低于baseline

**应对**:
- Plan A (2周内): 快速修复,达到baseline水平
- Plan B (4周内): 如果无法超越baseline 3+个点,考虑:
  - 调整方法 (简化架构)
  - 换研究方向 (efficiency为主而非accuracy)
  - 投稿workshop而非主会

### Risk 2: 竞品压力 (ComoRAG) (High)
**挑战**: ComoRAG (2025) 同样主打元认知 (Metacognitive Regulation)，概念重合度高。

**应对 (差异化生存)**:
1. **攻击其"贵"**: ComoRAG 需要 Tri-Retrieve (3种索引) x 5轮循环，推理成本极高，无法用于实时搜索。
2. **攻击其"慢"**: 强调 MetaCogRAG 的 L0 Gating 是 O(1) 复杂度，能拦截 50% 流量，是工业级方案。
3. **强调"可学习"**: ComoRAG 只是 Prompt Engineering，我们是 Model Alignment (GRPO)。

### Risk 3: 基线复现困难 (Medium)
**挑战**: Self-RAG等方法可能难以复现

**应对**:
- 使用官方代码
- 联系作者获取帮助
- 如果实在无法复现,用论文报告的数字+我们的实现对比

### Risk 3: 计算资源不足 (Medium)
**需求**: 4-6个数据集 × 8-10个方法 × 多次运行 = 大量GPU时间

**应对**:
- 优先级排序,先做核心实验
- 使用更小的验证集快速迭代
- 考虑申请云GPU资源

### Risk 4: 时间不足 (High)
**现实**: 13周完成所有工作非常紧张

**应对**:
- 严格执行时间表
- 每周milestone检查
- 必要时舍弃非核心实验 (如某些数据集/baseline)
- 考虑投稿下一个deadline (晚3-6个月)

---

## 💰 资源需求估算

### 计算资源
- **GPU时间**: ~500-1000 GPU小时
  - 主实验: 200 GPU小时
  - Baseline复现: 150 GPU小时
  - 消融实验: 100 GPU小时
  - 参数调优: 50 GPU小时
  - 补充实验: 100-300 GPU小时

- **存储**: ~500GB
  - 数据集: 50GB
  - 模型checkpoints: 100GB
  - 实验结果: 50GB
  - 中间文件: 300GB

### 人力
- **主要研究者**: 1人 (全职)
- **合作者/导师**: 定期讨论
- **标注员**: 3-5人 (人工评估)

### 其他
- **API费用**: 如果使用GPT-4/Claude作为baseline (~$500-1000)
- **会议注册费**: ~$800-1200
- **差旅费**: ~$2000-3000 (如果中稿)

---

## 📝 当前最紧急的TODO

### 本周必须完成 (Week 1)

#### 优先级P0 (Critical)
- [ ] **今天**: 运行Naive RAG 10样本对照
- [ ] **今天**: 对比分析,找出MetaCogRAG问题根源
- [ ] **明天**: 修复核心bug,10样本EM>20%
- [ ] **后天**: 验证修复,确保可复现

#### 优先级P1 (High)
- [ ] 周三-周四: NQ 200样本实验
- [ ] 周五: HotpotQA 100样本实验
- [ ] 周末: 结果分析,准备下周调优

### 下周计划 (Week 2)
- [ ] 参数调优 (Grid search)
- [ ] 性能提升到超越baseline
- [ ] 初步消融实验
- [ ] **Milestone 1检查**: Go/No-go决策

---

## ✅ 成功标准 (顶会接收)

### 必要条件 (Must Have)
1. **显著创新**: 三层元认知架构有明确理论支撑
2. **性能提升**: 所有数据集上稳定超越SOTA 2-5个点
3. **完整实验**: 4+数据集,5+baseline,完整消融
4. **深入分析**: 不少于4个分析实验,证明方法insights
5. **可复现性**: 开源代码,详细实验设置

### 加分项 (Nice to Have)
1. 理论分析 (收敛性/复杂度证明)
2. 人工评估
3. 多语言实验
4. 长文档/多跳推理表现突出
5. Efficiency优势明显

### 投稿策略
- **首选**: NeurIPS 2025 (ML顶会,6月截稿)
- **次选**: ICLR 2026 (10月截稿)
- **备选**: ACL 2025 (NLP顶会,2月截稿) 或 EMNLP 2025 (6月截稿)
- **保底**: Workshop或次级会议

---

## 🎓 论文标题候选 (Tentative)

1. **MetaCogRAG: Meta-Cognitive Retrieval-Augmented Generation with Adaptive Uncertainty-Guided Control**

2. **Adaptive Retrieval-Augmented Generation via Multi-Level Meta-Cognitive Reasoning**

3. **Uncertainty-Guided Meta-Cognitive Control for Efficient Retrieval-Augmented Generation**

4. **MetaCog: Self-Aware Retrieval-Augmented Generation with Hierarchical Meta-Cognitive Architecture**

选择标准:
- 清晰传达核心创新
- 包含关键词 (RAG, Meta-Cognitive, Uncertainty, Adaptive)
- 简洁有力 (<15 words)

---

## 📚 必读参考文献 (当前领域SOTA)

### 核心相关工作
1. **Self-RAG** (ICLR 2024) - 必须超越
2. **Adaptive-RAG** (2023) - 主要对比对象
3. **FLARE** (2023) - Active retrieval
4. **IRCoT** (2023) - Interleaved retrieval
5. **REPLUG** (2023) - Retrieval as plugin

### 理论基础
6. **Semantic Uncertainty** (Nature 2024)
7. **Self-Consistency** (ICLR 2023)
8. **Conformal Prediction for LLMs**

### 应用场景
9. **Atlas** (2023) - Large-scale RAG
10. **RAG Survey** (2024) - 综述

---

**最后更新**: 2025-11-11
**下次Review**: Week 1结束 (11.17)
**状态**: 🚨 紧急修复阶段

**关键消息**: 真实进度只有15%,距离顶会投稿还有大量工作。当前最紧急的是让系统跑通并超越baseline,否则后续一切无从谈起。
