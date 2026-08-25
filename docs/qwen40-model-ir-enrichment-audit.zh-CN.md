# Qwen 4.0 Model IR 精细化与缺失原因审计

## 结论

原 Model IR 的主干 topology 基本正确，但只完成了“模块/边界可映射”，没有完成“模型语义闭包”。这不是 eager/CUDA Graph trace 缺失造成的，也不应该靠 trace 自动补齐。根因是旧 Stage 1 只要求 module、edge、symbolic shape 和 stable ID，没有要求 parameter、state/cache、公式与 architecture-bearing config field 的闭包验收；schema 和 test 也允许一张正确但信息很薄的图通过。

本次保留默认图的简洁粒度，把精度放入 `semantic_details`、ledger 和 drill view：

- 增加参数总账、state/cache 总账、QSA/GDN/PLE/MoE/MTP 的精确 head、shape、生命周期和参数口径；
- 新增 QSA compressed index 下钻图，区分 raw index K、4-token compression、compressed K、Top-512 block 和 `≤2051` full-token positions；
- 明确 HC mix 与 combine 的公式和输入边界；
- 明确 GDN fixed-size conv/recurrent state、QSA token-growing cache、PLE fixed-size side state；
- 记录语义 reference 和 snapshot caveat；
- 保留 `ir_version=2` 的 execution-topology generation；完整 source-obligation closure 在 revision 5 完成，revision 6 增加统一的 shape/operator notation、视觉语法，并关闭过去缺失的 tensor-edge contract。纯 `operator_signature` 展示不进入 Execution fingerprint；本轮补齐的 edge shape/dtype/role 属于结构 contract，因此建立新的 fingerprint baseline。Compiler 从 revision 4 开始拒绝没有 drill view 的 multi-operator leaf；从 revision 6 开始拒绝 label 中的裸 dimension transform、未声明的 signature symbol，以及缺少 shape/dtype 的 tensor edge。新增的 fused primitive显式共享已有 timing owner，不复制时间。

## Revision 6 的统一 notation

图上的三类信息现在严格分离：edge 只表达 tensor layout 与 dtype；节点标题只表达算子语义；线性/投影等 operator transform 由结构化 `operator_signature` 统一渲染为 symbolic-first、concrete-second，例如 `H → E  (2560 → 512)`。`R=4`、`L=320`、`I=640`、`E=512` 等复用维度统一在顶层 `dimensions` 声明，主图与 drill view 不再混用 `[B,T,4,H]`、`[B,T,10240]` 和裸 `2560 → 640`。

视觉通道也彼此独立：node fill/glyph 表示 operator family 与结构角色，edge line style 表示 data/residual/cache/control，蓝色外框只表示 selected，profile heat 使用独立细条而不覆盖 operator color。共享 viewer 中的 legend 是该约定的可见说明；模型 catalog 不允许添加专用颜色或专用 viewer 分支。

## Revision 5 的完整闭包

Model IR 现在把稳定的数学路径继续展开到 primitive data-flow，而不把当前 kernel fusion 当作叶子边界：

- Hyper-Connection read gate：`RMSNorm → down projection → /4 + SiLU → up projection → sigmoid/view → branch weighting → branch mean`；
- Hyper-Connection write gate：`inject projection → /4 + 2×sigmoid → broadcast multiply → residual add`；
- MoE routed/shared expert：gate/up 两路 projection、SiLU、elementwise product、down projection，以及 shared scalar gate；
- MoE route selection/combine、PLE grouped norm/gate、GDN conv/SiLU、QSA norm/MRoPE 和 index compression/scoring/expansion，也都有独立 drill view。

这些节点是跨 framework 稳定的模型数学语义。SGLang、vLLM 或 TRT-LLM 可以把若干节点融合到同一个 kernel；Profile 通过唯一 timing owner 和 `shared_interval` fusion group 表达这种实现差异，不能把同一个 interval 重复累计到每个 primitive。

本轮不再按截图逐点补图，而是依次关闭 PLE、HC、GDN、QSA、MoE、MTP/EAGLE 六个 source scope，并执行双向审计。最终结果为：15 个 pinned source file、26 个 verified entrypoint、155 条 source obligation、213 个 audited Model IR leaf；0 pending entrypoint、0 pending obligation、0 unclassified source member、0 uncovered leaf、0 compound primitive target。10 个 reverse exclusion 只保留 execution-support 或旧 profile timing-owner compatibility 节点，不冒充模型数学。

此外，MTP 与 target model 仍共享同一套稳定 QSA/MoE drill view，但 measurement 采用 caller-isolated 规则：没有 MTP 子 leaf evidence 时只显示其父区间或未单独归因，禁止回退借用 target-model timing。

## 从同事 HTML 借用了什么

参考文件：`/Users/yangminl/Downloads/qwen4-exp-architecture.html`。

纳入 Model IR 的内容必须同时满足：由 config/构造语义决定、跨 framework 稳定、不会因 kernel/fusion 改写而变化。本次纳入：

- `[GDN,GDN,GDN,QSA] × 12`、PLE 位于 1-based L2；
- HC `4 × 2560`、rank 320、mix/combine 数学边界；
- GDN 的 16 Q/K、48 V/Z、128 head dim、conv/state contract；
- QSA 的 24Q/2KV、index 4Q/1K、128 index dim、c4、Top-512 blocks、2048 + causal tail `≤2051`；
- MoE 的 512 routed / Top-10 / 1 shared、intermediate 640 与参数闭包；
- PLE 的 16 × 160、2/3-gram head split、projection、dilation-3 short-conv 与 state；
- 单层 Full+QSA MTP、PLE off、target embedding/head 共享及额外参数口径。

以下内容没有进入 Model IR：

- `torch.cat ×4` 的具体 materialization；
- TileLang、Triton、PAI-FA3、FlashInfer、DeepGEMM、AITer 等 backend；
- A2A、TP all-reduce、DP/EP placement；
- “几个 logical kernel”、具体 fusion/launch/stream 建议。

它们分别属于 Execution IR、Binding、Timeline/Profile 或 optimization analysis。特别是参考 HTML 的 MoE `TOPK · A2A · GEMM` 不是 pure-TP 下稳定的模型语义，不能原样复制。

## 为什么原来会漏

### 1. Pipeline 的 Stage 1 是 topology-first，不是 completeness-first

旧规则只要求读 config/source、画 semantic modules、标 tensor/state boundary、stable ID 和 symbolic shape。它能保证“图能用来挂 profile”，却不能保证：

- 所有 architecture-bearing config field 都被消费；
- 参数分项能和模型总量闭合；
- 所有 persistent state/cache 都有 shape、dtype、增长规律和 update owner；
- 公式和 weight sharing 能从 IR 直接读取；
- optional path 是被表达，还是被静默遗漏。

### 2. Schema 允许节点只写四个字段

`id + label + shape class + semantic_op` 就能通过。没有结构化的 `semantic_details`，因此精确信息只能塞进 label、README 或人的脑子里，也没有自动化验收点。

### 3. Test 重点在 Execution/Profile 一致性

过去的 test 主要检查 drill reference、collective 边界、binding/profile target 和 timing attribution。它们能抓“AR 被画在 MoE 内部”或 mapping 错误，却不会抓“QSA raw/compressed cache 没画”“GDN state shape 未声明”。

### 4. 初始交付 scope 是 text-serving profile

为了优先完成 pure-TP profile 和跨图跳转，Model IR 被当作 timing attribution skeleton。Vision 路径没有进入默认 text graph，是明确 scope 选择；但过去没有把这种排除写入 machine-readable coverage，因此看起来像普通遗漏。本次先把 Vision 记入 ledger 并标为 `ledger_only`；在真正捕获 multimodal workload 前，不把未测路径硬塞进所有 text profile 的默认图。

### 5. Trace 不能修复这个问题

Eager stack 可以验证 Execution IR/Binding 的调用顺序、shape、multiplicity 和 state transition，也可以证明某个 semantic node 在 framework 中如何实现；但未执行的 optional path、参数共享、完整参数总账和长期 cache 语义不一定会出现在一次 trace 中。Model IR 必须由 config + construction source 建立，trace 只做 validation，不能反向成为模型架构的唯一来源。

## Pipeline 修复

Stage 1 现在增加四个 closure gate：

1. data-flow closure；
2. layer/optional variant closure；
3. parameter closure；
4. persistent state/cache closure。

并要求每个 architecture-bearing config field 有明确 disposition。默认图仍然只展示一个代表 layer；heads、experts 和相同 layer 不展开复制，精确信息进入节点详情或必要的 drill view。这样既保持当前 viewer 的可读性，也能复用于 vLLM、TensorRT-LLM 等不同实现。

### 补充审计：新增 drill 后的 mapping reconciliation

第一次精细化提交新增了 `qsa_indexer` drill，但沿用了父级 `qsa_attention.indexer` 的既有 Binding/Profile。父级聚合时间仍然正确，新增 leaf 却没有自动获得 source binding、独立 timing cell 和 timeline target。这说明 semantic completeness 与 runtime mapping completeness 是两道不同的 gate。

本次已基于原有、100% attributed 的 timeline evidence 回填全部 33 个 profile：

- projection、Q-prep、compression、compressed MQA score、Top-k 和 block expansion 分别映射到可测 leaf；
- raw index-K cache store 归属 Q-prep fused kernel，compressed-K cache store 归属 compression fused kernel，均不重复时间；
- input/output boundary 保持 structural；
- 每个 event 同时保留父级 rollup target 和最细 verified target；
- 每个 profile 的 drill residency mapping coverage 均为 100%，timeline content hash 已重新生成。

Pipeline 现在明确规定：Model IR semantic revision 即使不改变 Execution fingerprint，只要增加 runtime-bearing leaf，也必须重新完成 Binding/Profile mapping reconciliation。

## 当前边界

- 参数数字来自参考 artifact 标注的 config/source snapshot 和闭合推导，不能冒充 checkpoint-index 官方统计；snapshot 改变时必须重新跑 semantic audit。
- 当前默认 graph/profile scope 仍是 text-only serving；Vision 需要单独的 stable frontend view 和对应 workload/profile 后才能提升为完整已验证路径。
- 新增语义详情不代表当前 SGLang kernel 一定按相同粒度执行；fusion 和 kernel decomposition 由 Binding/Timeline 解释。
