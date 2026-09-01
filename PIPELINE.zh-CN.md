# 唯一标准的 IR-first Pipeline

[English version](PIPELINE.md)

状态：**这是所有新模型和新 profile 唯一支持的 pipeline。**

本文档是从模型定义和运行时 trace 生成 Architecture 与 Timeline 视图的唯一规范。Trace 可以为某个 implementation 提供实现归属和时间证据，但绝不能创建或改写模型架构。

## 1. 设计目标

Pipeline 必须同时满足以下要求：

- **语义稳定：** 即使源文件、helper function、fusion 边界、kernel 或 serving framework 发生变化，架构仍然能够被一致地理解。
- **跨 framework 比较：** SGLang、vLLM、TensorRT-LLM 以及未来的实现可以绑定到同一个 Model IR；如果它们的分布式执行 contract 相同，还可以共享同一个 Execution IR。
- **感知执行拓扑：** 只要 TP、Attention DP、CP、MoE EP、DeepEP 等路径改变了 placement、tensor layout、collective、state ownership 或数据流，就必须使用不同的 Execution Plan。
- **Execution 与时间均经过 trace 验证：** eager Python stack、shape、scope 和 collective order 用于验证 proposed Execution IR 并建立语义归属；production trace 用于提供 CUDA Graph 时间、stream、overlap 和 idle。
- **失败时保守处理：** 无法确定归属的语义 event 必须让验收失败，不能为了填补空白而分配给附近节点。明确位于 Model/Execution IR 之外的工作必须带有类型化 runtime/support 分类和具体原因；泛化的 `unmapped` 不能交付。
- **可复现：** 每个输出都必须记录 source commit、config、execution fingerprint、workload、trace hash、attribution 方法和 producer version。
- **展示层数据驱动：** Viewer 只渲染编译后的 metadata，不能包含针对某个模型、framework 或 generation mode 的 routing 规则。

## 2. 五类持久化 Contract

### 2.1 Model IR

Model IR 负责表达与代码实现无关的稳定语义：

- 逻辑 operator 和 module boundary；
- tensor/state data flow、symbolic shape、layout、dtype 和 state lifetime；
- 每个语义节点显式的数学 transition、boundary 声明、state update 或 control equation；
- repeat/layer 结构以及稳定的多级展开层次；
- MTP auxiliary head 等可选语义路径。

Model IR 应根据 model config、模型规范或论文以及 source review 起草，然后经过人工 review。它**不能由某一次 trace 生成**。

#### 语义 Transition Contract

每个 Model IR 节点都由两个不重复的真源编译得到：

- edge 唯一负责 tensor/state identity、shape、layout、dtype 和 lifetime；
- semantic operation table 唯一负责与 framework 无关的 equation 和 invariant。

Compiler 将两者合成为节点本地的 **Inputs → Transition / Equation → Outputs** contract，并在右侧详情栏固定展示。Label 只用于显示，不能作为语义证据。如果 operation 不改变 contract，输入输出仍需显式保持相同 shape/layout；如果 shape、layout、dtype、identity 或 state lifetime 发生变化，outgoing edge 必须写出新值。

Symbolic tensor dimension 同样属于一等 Model IR contract。每个 symbol 都必须
声明 meaning、value class（`model_constant`、`profile_runtime` 或
`stage_dependent`）和 provenance。Model constant 只能来自精确 checkpoint/config；
profile-runtime value 只能读取所选 profile 中显式 authored 的 metadata path；
stage-dependent value 必须带明确的 phase、generation mode 和 IR scope 规则。
Catalog 一旦声明该 dictionary，Symbol 缺失、stage target 不存在或动态值缺少验证
provenance 时，Compiler 必须 fail closed。特别是 speculative draft width 绝不能被当作全局 `D`；它只能在显式
定义的 MTP/draft scope 内解析。Symbolic shape 始终是真源；只有当所选 profile 和
当前 drill path 下引用的每个动态 symbol 都有效时，Viewer 才能附加 resolved shape，
否则必须显示 unresolved symbol 和原因。

已经命名的 axis 必须在 architecture edge 上保留其名字。即使 checkpoint constant
已知，Catalog 也不能把 `[N,GDN_value_heads,GDN_value_head_dim]` 退化成没有语义的
`[N,64,128]`。主图先显示 symbolic shape；只有 profile/stage 中所有 symbol 都完成
验证后，才允许另起一行显示 `resolved ...`。Semantic closure test 必须拒绝把
Catalog 已声明的 model-specific symbol 重新写成旧数字常量。

每个 drill-down 都必须声明 boundary direction 和可验证的 boundary contract。子 view 的输入输出必须与 parent edge contract 一致。对于 mHC pre-collapse 与 post-sublayer recombination 这种跨多次调用的完整语义生命周期，必须显式声明 scoped parent nodes 和中间 handoff，不能伪装成单个节点等价。由 runtime 提供输入的 optional entry 也必须明确标记，不能绕过校验。

#### Repeated layer 与展开规则

- 重复 layer stack 默认折叠，只显示 layer 数量和稳定排列规律，不为每个 layer instance 复制一张图。
- 每种语义不同的 layer type 使用一张代表 view；结构相同的 layer 共享该 view。例如 linear-attention layer 与 full-attention layer 分开，而 36 个相同的 linear-attention layer 不逐个展开。
- 代表 layer view 按稳定数据流展开 Attention、MoE/MLP、residual/HyperConnection 等主要语义模块；模块内部确实存在有架构意义的数据流时，可以继续 drill down。
- 只在特定 layer 出现的一次性或旁路模块（例如 PLE injection）在 stack view 中单独表示，不因此复制一张完整的特殊 layer 图。

Timeline 和 Profile evidence 可以保留实际 `layer_id`、expert/rank 等 invocation context；从具体 event 跳回 Architecture 时，打开对应的代表 layer/module leaf，并在详情中显示实际 instance。Runtime instance context 不会产生重复的 Model IR node。

Implementation helper、CUDA stream、kernel、fusion 和 collective 不属于 Model IR；只有当 collective 本身属于模型数学语义时才是例外。

### 2.2 Execution Plan 与编译后的 Execution IR

Execution Plan 描述与 framework 无关的分布式执行 contract：

- parallelism dimension 和 rank group；
- 每个 module boundary 上的 tensor placement 和 layout；
- communication operation、payload 和 result；
- state ownership 和 layout transition；
- 相对于 Model IR 的 topology-specific 插入或替换。

Compiler 首先将 Execution Plan 应用到 Model IR，生成一个 **candidate Execution IR** 和确定性的 structural fingerprint。Fingerprint 只根据规范化后的 execution contract 计算；source symbol、Python function name、kernel name 和 trace timestamp 都不会进入 fingerprint。只有在 Stage 4 使用关闭 CUDA Graph 的 eager run 完成 reconciliation 后，candidate 才能成为 validated Execution IR。

只要两个 framework 在 canonical IR boundary 上具有相同的可观察 contract，它们就可以共享同一个 Execution IR。这里比较的是 placement、layout、state ownership、data dependency 和 logical communication result。Physical collective algorithm、fusion、kernel sequence 和 stream scheduling 都属于 Binding/Timeline 细节，不改变 fingerprint。例如 NCCL all-reduce、custom two-shot all-reduce，以及内部使用 reduce-scatter + all-gather 的 lowering，只要都是把每个 rank 上的 partial hidden 转换成相同的 replicated hidden，就可以实现同一个 `TP output collective` contract。

只有当 intermediate layout 或 state 跨越 canonical boundary、被其他 module 消费，或者改变了可观察 data flow 时，才需要新的 Execution Plan。例如 reduce-scatter 的结果保持 sharded，并在之后 all-gather 之前被下一个 module 消费，就属于不同 plan；如果 reduce-scatter + all-gather 只是 all-reduce 的内部算法，则仍属于同一个 plan。

Execution IR 保持在 **contract 粒度**。Local argmax、global top-k helper、allocator call、framework scheduling helper 等操作，除非引入了具有架构意义的 layout、communication 或 state boundary，否则应放在 Binding 或 Timeline evidence 中，而不单独成为 Execution IR 节点。

### 2.3 Implementation Binding

Binding 分两步生成。Draft Binding 先将 source/config evidence 映射到 candidate Execution IR 节点，并提供捕获和解释 eager run 所需的 anchor。只有 eager reconciliation 验证 candidate graph 通过后，Binding 才会 finalized。它记录：

- framework 及其版本或 source commit；
- model/backend configuration；
- canonical Python/C++ symbol 和 source permalink；
- eager stack match rule 和稳定 runtime anchor；
- eager trace hash、validated scope、observed shape、collective order 和 execution-validation result；
- kernel signature、fusion group 和已知 framework helper scope；
- 证明该 binding 实现了所选择的 execution fingerprint。

一个 Binding 可以把多个 source scope 或 fused kernel 映射到一个 IR 节点；也可以通过显式 `fusion_group`，让同一个 kernel 作为多个 IR 节点的共享证据。Binding 不能创建 semantic node 或 execution node。

### 2.4 Profile

Profile 是以下组合的一份不可变 measurement overlay：

`model + execution fingerprint + implementation + generation mode + phase + hardware + workload + rank policy`

Profile 保存 per-node timing、provenance、coverage、mapping state、workload 和 raw artifact hash。Prefill 与 decode 必须是不同 profile；不同 batch size 也必须是独立 measurement，不能平均成一个数字。

### 2.5 Timeline Artifact

Timeline artifact 保存精确的运行时证据：

- kernel start、duration、device、rank、stream 和 correlation identifier；
- 如果存在，则保存 eager stack/source evidence；
- 映射到的 Model/Execution IR target 及 mapping confidence；
- layer/module display lane 和更底层的 kernel lane；
- idle interval、overlap lane、synchronization 和 collective event；
- 通过 content hash 关联 raw trace。

Timeline 数据不能重新定义任何一层 IR。

Generation mode 是 Profile/Binding 的一个维度，**不是第六层 IR**。例如 EAGLE MTP 使用稳定的 MTP Model IR view，在 target verification 时复用 target-model graph，并选择 MTP 专用 entry view；不存在单独的“Generation IR”。

### 2.6 可选的 SoL 与 Gap Analysis 派生物

SoL 不增加新的 canonical IR 层，而是使用既有 Model/Execution IR node ID 的理论 overlay：

- `workload-ir.v1`：冻结 measured Profile 的真实 phase、CUDA Graph mode、batch/sequence shape、cached token、active expert、MTP 与 scheduler 状态，禁止 simulator 偷换成 generic shape；
- `cost-ir.v1`：从 Model/Execution IR 派生的跨 framework operator contract，保存 resolved problem shape、useful work、compulsory traffic、repetition 和 operator family，不包含 framework symbol 或 kernel name；
- `transition-plan.v1`：将一次执行拆成显式 transition DAG。同一 transition 内的 Tensor Core、HBM、L2、SFU 等资源下界并发取 `max`；只有通过 `depends_on` 连接的 transition 才串行相加。Collective startup 与 wire transfer 必须分开；
- `kernel-plan.v1`：描述 algorithm、tile、warps/stages、persistent scheduling、fusion、cache reuse、launch 和 sync；它是 implementation evidence，不进入 Model IR；
- `hardware-spec.v1`：跨模型共享、per-GPU、带来源与版本的理论 ceiling，以及绑定 kernel-plan fingerprint 的 microbenchmark/correlation surface；
- `sol-profile.v1`：绑定 model、execution fingerprint、workload fingerprint、hardware、phase 和 assumption set，保存 transition-derived `ideal_ms`、可选 plan-exact `attainable P10/P50/P90`、top-N limiter、coverage 和 dependency critical path；
- `gap-report.v1`：将一个 immutable measured Profile 与对应 SoL Profile 比较，区分硬件/shape gap、implementation gap 和尚未分配原因的 gap。

证据强度依次是：transition-derived physical ideal、plan-exact calibration/projection、measured silicon。物理时间必须在容差内满足 `ideal <= observed`，否则 physical model invalid。Attainable projection 必须匹配 exact workload shape 与 kernel-plan fingerprint，并报告 P10/P50/P90；`observed < P10` 时 projection invalid，必须重新 correlation。缺少匹配 projection 时不能把 `observed - ideal` 称为 framework implementation gap。旧 operator-family efficiency/fixed-overhead envelope 只能作为显式 opt-in 的 `legacy sensitivity`，默认关闭且永远不能写入 `attainable_ms`。Unsupported operator 必须显式展示，且不能借用其他 operator family 的效率。

## 3. 端到端流程

```text
model config + specification/paper + source review
                         |
                         v
              起草并 review Model IR
                         |
 Model IR + Execution Plan(s) + source/config
                         |
                         v
            candidate Execution IR(s)
                         |
                         +<----------------------+
                         |                       |
                         |             eager semantic run
                         |             关闭 CUDA Graph
                         |             stack + shape + order
                         +------ reconciliation
                         |
                         v
 validated Execution IR fingerprint + finalized Binding
                         |
                         v
              production timing run
              真实 serving mode / CUDA Graph
                         |
                         v
                   Profile + Timeline
                         |
                         v
       validate -> compile -> static viewer bundle
```

这个 pipeline 有意让不同 evidence 承担不同职责：source/config 与 Execution Plan 先提出 contract；eager evidence 验证 **“实际执行了什么，以及每个 event 在语义上是什么”**；production trace 回答 **“它何时、在哪里执行”**。Python stack 用来验证 structural fingerprint，但不进入 fingerprint 本身，因此不同 framework 仍然可以证明它们实现的是同一个 contract。

## 4. Pipeline 阶段

### Stage 0 — 冻结 Run Manifest

开始 profile 之前必须记录：

- model config 和 weights identifier；
- source repository、精确 commit，以及存在本地修改时的 dirty-patch hash；
- framework/backend version 和 launch command；
- hardware、rank topology、parallelism、dtype、quantization 和 generation mode；
- phase、requested/realized ISL/OSL、global/local batch size、request rate、request ordering、warmup/formal request multiplier 和 seed；
- scheduler policy、chunked/mixed-prefill setting、token budget、preemption/retraction policy、prefix-cache state 和 framework-native step counter；
- 请求生成的 artifact 和 acceptance level。

Manifest 是唯一的 orchestration input。任何 builder 都不能静默替换 batch size、backend、CUDA Graph mode 或 topology。

### Stage 1 — 编写并 Review Model IR

1. 阅读 model config 和稳定架构定义。
2. 确定 semantic module、tensor/state boundary、重复 layer schedule 和 optional path。
3. Source 只用于消除歧义，不能把当前 call graph 直接复制为架构。
4. 分配稳定 ID，并完整定义 edge identity、symbolic shape、layout、dtype 和 state lifetime。
5. 为每个语义节点编写与 framework 无关的 equation 及必要 invariant。Boundary、state、control 和可展开的 module 节点也不能例外：必须明确写出 pass-through、更新、选择或 composite transformation，不能依赖 compiler/viewer 猜测的 fallback。
6. 将每个 drill boundary 声明为 exact node、exact multi-node lifecycle 或 explicit external entry，并定义 input、handoff 和 output shape。
7. 在附加任何 runtime data 之前运行 semantic closure tests 并 review 整张 graph。

只有模型语义架构确实改变或原有语义表达有误时，才能修改 Model IR。Framework refactor 或 kernel fusion 不足以成为修改理由。

### Stage 2 — 编写 Execution Plan

首先创建默认 pure-TP plan，然后只为有意义并且确实能够工作的 code path 添加新 plan，例如 Attention DP、CP、MoE EP，以及使用特定 communication backend 的 Attention-DP + MoE-EP。

每个插入的 communication node 都必须说明：

- collective 和 rank group；
- input payload、shape/layout 和 dtype；
- output/result layout；
- 它属于 module boundary 还是 module internal。

Compiler 校验引用并生成 candidate structural fingerprint。例如 pure TP MoE output reduction 是 module-boundary TP output operation；即使某个 framework 把它实现在 MoE 类内部，也不能因此隐藏在 MoE semantic module 里面。在 eager reconciliation 通过之前，这个 fingerprint 不能进入 deliverable 状态。

### Stage 3 — 创建 Framework Binding

Binding adapter 读取精确的 source revision，首先生成 Draft Binding，包括 canonical symbol identity、source link、stack rule 和 runtime anchor。通过 source AST 或 callsite validation，防止仅用于展示的 alias 被误当成 canonical identity。Stage 4 再附加实际 eager evidence 和 validation result，形成 finalized Binding。

Framework-specific helper 留在 Binding 中；稳定执行 contract 留在 Execution IR 中。正是这一边界，使 SGLang 和 vLLM trace 可以在同一张 graph 上比较。

### Stage 4 — 捕获 Eager Evidence 并验证 Execution IR

针对每个不同 code path 和 phase，在关闭 CUDA Graph 的情况下捕获 trace，从而获得 Python stack 和 operator scope。至少需要满足：

- prefill 和 decode 分开；
- 每个 execution fingerprint 分开捕获；
- 当 MTP 或其他 auxiliary path 改变实际调用模块时，generation-off 与 generation-on 分开；
- target-model verification 和 auxiliary/draft scope 分别确定边界。

Semantic trace 记录 stack、可获得的 tensor shape、operator order、collective order 与 payload、rank、stream 和精确 invocation boundary。然后使用它对完整的 candidate Execution IR 进行 reconciliation：

1. 每个 runtime scope 必须对应已有 contract node、允许的 framework-helper category，或者明确记录为无法解释的 discrepancy。
2. 实际观察到的 placement/layout transition 和 collective result 必须与 plan 一致。
3. Layer multiplicity、optional path、state update 以及 phase/generation scope 必须与 candidate graph 一致。
4. 按计划应该执行、却没有任何 eager evidence 的节点属于失败。`structural` 只表示作者明确声明的语义／控制／状态边界，绝不能作为计算映射缺失时的 fallback；只有确实未激活的节点才能显式标为 `not_selected`、`disabled` 或 `out_of_scope`。
5. 未预期的 eager scope 可以提示 Execution IR contract 缺失或位置错误，但不能自动修改 graph。

Eager artifact 必须保留 event-level evidence graph，而不能只保存聚合后的 node label：

```text
eager event ID -> Python/operator stack -> Binding rule
               -> Execution IR contract -> Model IR semantic leaf/leaves
               -> invocation scope（phase、layer、sublayer、occurrence）
```

这个关系天然允许多对多：implementation fusion 会让一个 eager event 覆盖多个 semantic leaf；一个 semantic leaf 也可能 lower 成多个 eager event。每一条边都必须记录 mapping rule、confidence 和作用域。只有 node-level aggregate、没有 event edge 的结果，不能作为 production transfer 或 fusion 展示的充分证据。

如果不一致，Execution Plan 必须退回 review。Reconciliation 成功后，以 implementation-specific validation attestation 对 structural fingerprint 完成验证，并把该 attestation 存入 finalized Binding。Trace 不能生成 Model IR，其 framework-specific stack name 也不会成为 shared fingerprint 的组成部分。

Stage 4 会生成三个可以直接 review 的 artifact：

- `observed_execution`：根据 eager stack、invocation order、shape、collective 和 state update 生成的 framework-specific graph；
- `execution_reconciliation`：Observed graph 与 candidate Execution IR 之间显式的 matched/missing/unexpected diff；
- Finalized Binding，以及带 content hash 的 validation attestation。

`observed_execution` 就是传统 profiler 语境下很容易被称作 Execution IR 的那张具体、trace-derived execution graph。它是非常重要的证据，但不是另一层 canonical IR：helper scope 和 framework call structure 仍然是 implementation-specific 的；Binding 才是从这张 observed graph 到共享 contract-level Execution IR 的、经过 review 的映射。

### Stage 5 — 捕获 Production Timing Evidence

必须使用真正准备交付的 serving mode：

- 对于 target concurrency `C`，warmup round 提交 `3 × C` 个 request，formal round 提交 `1 × C` 个 request。这里的 3 倍和 1 倍是 request-count multiplier，不是 model forward iteration 数量；
- 在标准 srt-slurm/sa-bench workload 中设置 `random_range_ratio: 1.0`，使用 fixed OSL 并 ignore/disable EOS，同时记录每个 request 实际产生的长度。这能让 request completion 尽量对齐，为稳定 decode batch 创造条件；
- prefill：当 stack collection 会影响时间时，使用关闭 stack 的 eager timing trace；
- decode：启用 CUDA Graph，并且只从经过验证的 formal-round steady-state window 捕获；
- 默认 decode sweep：global BS 1、16、64、256；
- 显式记录 ISL/OSL，例如 8K/1K；
- MTP profile 必须使用真实 CUDA Graph path，同时捕获 target verification 和 auxiliary/draft work。

`random_range_ratio` 不是可以跨 framework native CLI 直接照搬的 contract。上面的 canonical manifest value 对我们共同使用的 sa-bench workflow 是确定的；adapter 不能把它不加验证地转发给 framework-native benchmark client。例如当前 vLLM native benchmark code 使用 `0.0` 表示精确 target length。应优先使用共同 workload generator；如果必须使用 native client，adapter 需要翻译规范化后的 `fixed_lengths: true` 意图，保留 manifest 中用户指定的值，并通过每个 request 实际生成的 ISL/OSL 证明二者相等。

#### Stage 5A — Baseline Run 与 Step 选取

Window selection 必须采用两次运行：

1. 第一次使用完全相同的 workload **正常运行，不开启 profiling，也不直接 capture**。保持真实 serving config、seed、request order、`3 × C` warmup round 和 `1 × C` formal round。以足够细的粒度记录 framework-native scheduler step，以及每个 step 的 forward mode、running request、scheduled token、realized shape 和 batch composition。
2. 从 baseline 中分别寻找 formal prefill 和 formal decode 的连续稳定区间。Step number 只对这一组精确的 model、framework commit、topology、scheduler config、generation mode 和 workload 有效；MTP 与 non-MTP 绝不能复用 window。
3. 在每个稳定区间的中部选择 step，并与 request admission、phase transition、batch drain 和 request completion 保留足够距离。Raw profiler window 可以覆盖多个 step，以便验证稳定性；最终 canonical profile sample 可以是其中一个经过验证的 representative step，或者一个明确声明语义的 window rollup。
4. 第二次使用相同的 trajectory-affecting field 重新运行，并开启 profiling。只有 profiler control 和选定的 start/stop trigger 可以变化。
5. Capture 完成后再次验证真正捕获到的 step。Baseline step number 只是 window-selection 依据，不能证明第二次运行一定落在正确位置。

Framework adapter 将自己的 native counter——例如 `forward_ct`、scheduler iteration、executor iteration 或其他等价 monotonic coordinate——转换为 `window_selection` artifact。该 artifact 保存 baseline log hash、candidate stable range、选定 start/stop step、选择原因和 post-capture evidence。

#### Stage 5B — Stable-Step Acceptance

Pure prefill sample 必须具有稳定 forward mode 和稳定的 scheduled request/token shape，并且所选 invocation 内不能混入 decode token。Pure decode sample 的所有 selected step 必须满足：

- 实际 global/local decode batch size 等于 target concurrency；
- batch 中没有混入 prefill/extend work；
- selected interval 内没有 request admission、completion、preemption、retraction 或 KV-cache recomputation；
- sequence-length/shape bucket 和所要求的 CUDA Graph state 保持稳定；
- 对于 MTP，verification/draft configuration 和 scheduler iteration scope 与 manifest 一致。

不能根据 framework 名称推断上述条件。Scheduler behavior 是一个 versioned configuration dimension：

- SGLang 存在 prefill admission、chunked-prefill 和 mixed-batch path；真正顺序取决于精确 source revision 和 flags。
- vLLM V1 在条件允许时默认启用 chunked prefill；在该模式下会先调度 decode，再用剩余 token budget 调度 prefill。
- TensorRT-LLM 的 in-flight batching 可以在同一个 iteration 中放入 context-phase 和 generation-phase sequence，capacity scheduler policy 还会影响 request admission 和 pause。

Primary reference：[SGLang scheduler source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)、[SGLang scheduler arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)、[vLLM chunked-prefill policy](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)、[TensorRT-LLM in-flight batching](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html#in-flight-batching)，以及 [TensorRT-LLM scheduler policies](https://nvidia.github.io/TensorRT-LLM/latest/legacy/performance/performance-tuning-guide/useful-runtime-flags.html#capacity-scheduler-policy)。

如果准备交付的 native serving mode 必然混合两个 phase，就把它保存为单独的 `mixed_serving` profile。它可以用于 end-to-end scheduler analysis，但不能标成 pure prefill/decode kernel profile，也不能与 pure phase profile 直接比较。跨 framework 做 architecture/kernel 比较时，必须使用 phase-isolated window，或者采用具有相同 realized batch composition 的共同 phase-specific harness。

Apple-to-apple acceptance 比较的是实际 observation，而不只是 CLI 名称：realized ISL/OSL、target concurrency、scheduled sequence/token count、phase composition、generation mode、execution fingerprint、graph mode 和 cache state 必须相同。Scheduler policy 保留为可见 profile dimension，不能被静默归一化掉。

如果 trace 的实际 mode、phase、realized shape/batch composition、graph state、selected step 或 formal window 与 manifest 和 `window_selection` evidence 不一致，pipeline 必须拒绝该 trace。

### Stage 6 — 将 Eager Attribution 转移到 Production Trace

CUDA Graph trace 通常没有 Python stack，因此只能在经过校验的 invocation segment 内转移映射。

必须满足以下保护条件：

1. 匹配稳定 segment anchor 和 execution fingerprint。
2. 匹配 phase、generation scope、layer/module scope、rank、shape bucket 和 formal-step cardinality。
3. 对齐精确的 kernel/event subsequence；只有精确结构对齐成功后，才允许使用 kernel family。
4. Collective、graph boundary、synchronization 和 state commit 都是硬边界，attribution 不能跨越这些边界。
5. 保持顺序和出现次数，不能使用 greedy nearest-neighbor matching。
6. 每个映射后的 production event 都记录 eager evidence ID 和 transfer rule。
7. 不允许发布未匹配的语义 event。每个 production event 必须绑定到
   IR/fusion evidence，或者以明确原因分类为类型化 runtime/support 工作；
   泛化的 `unmapped` bucket 直接失败。
8. 每个必需的 Model/Execution IR 节点最终都必须归入 `measured`、带共享
   interval owner 的 `fused/shared`、`state`、`structural` 或
   `not_selected`。交付 profile 必须同时满足：IR 中必需节点的
   `mapping_incomplete` 为零，raw timeline 中疑似 GEMM、attention、MoE、
   normalization、convolution 或 collective 的语义 kernel 在 IR 外为零。
   Runtime、scheduler、allocator/cache、state bookkeeping、attention-plan
   metadata、sampling/output 只有在带有类型和具体原因时才能留在 IR 外。

这些规则可以防止 `lm_head` 等节点跨过 TP vocabulary collective，错误吸收后续的 post-logit helper kernel。

Stage 6 的输出是显式的 production-to-eager evidence graph：

```text
production event ID <-> eager event ID(s) <-> Execution IR node(s)
                                      <-> Model IR leaf/leaves
```

每条边同时携带 `phase`、`layer_id`、`substage` 和稳定的 `occurrence_id`。Mapper 不能在使用 segment/layer identity 完成对齐后又把它丢弃；正是这个作用域，区分了“第 12 层 attention 的 mHC boundary”和仅仅具有同名 kernel signature 的全模型聚合。

只有当每个 covered IR leaf 在已经验证的 occurrence scope 中都拥有明确的 production-event evidence 时，fusion relation 才成立：

- `shared_interval`：一个精确 production interval，必须具有 exact-occurrence scope；
- `shared_event_set`：多个已验证 production interval 的聚合，必须标记为 `profile_aggregate`，不能在界面上伪装成一个大 kernel。
- `shared_event_coverage`：某个 semantic member 由唯一 owner 的 production-event set 中一个非空、显式的子集实现。必须同时记录完整 owner event set 和每个 member-to-event 映射。Timeline 跳转和 roll-up 只能把该 member 附着到列表中明确记录的 events 上，不能再把这个子集扩大为 owner 的全部 events。
- `fused_by_occurrence`：同一个 semantic node 在不同 layer／substage occurrence 中分别 fused 到不同 physical owner。每个不相交 partition 都必须记录 owner 和 production event IDs。

每个 fusion group 只有一个 timing owner。Covered leaf 继续作为显式 Model IR contract 存在，但卡片必须指向这个 owner，并且不能复制 owner 时间参与求和。Composite parent 使用 child production event 的并集；不能因为部分 descendant 被 fusion，就把整个 parent 标记成 fused。

### Stage 7 — 计算 Timing Metric

对于某个 rank 上每一次经过校验的 node invocation，计算：

- `elapsed`：invocation envelope 的 wall time；
- `active_gpu`：归属于该节点的 GPU interval 的并集，重叠只计算一次；
- `residency`：归属于该节点的 kernel duration 总和；
- `overlap_repeated = residency - active_gpu`；
- `other_gpu_only`：envelope 内不属于该节点、且不与其 active union 重叠的其他 GPU-active time；
- `device_idle`：envelope 内所有 stream 都没有 GPU work 的时间。

不重叠的 envelope 恒等式为：

```text
elapsed = active_gpu + other_gpu_only + device_idle
residency >= active_gpu
```

Multi-rank profile 保留每个 rank 的独立 measurement，绝不能把不同 rank 的 wall time 相加。对 request wall time 的报告必须声明 aggregation policy，通常使用 critical/tail rank，并保存被选择的 rank 和跨 rank 分布。

Parent/child rollup 必须显式携带 `exclusive` 或 `inclusive` semantics；当 interval 存在重叠时，绝不能机械相加。

Compiler 必须从 Model IR 的 `drill` 关系推导 rollup ancestry，而不能依赖 framework 名字。任何拥有 measured descendants 的可执行 drill node，都必须物化一个 `inclusive_rollup`：`active_gpu` 对底层 production event 做区间并集，residency 才做 duration 求和。每个已选中的 runtime-bearing primitive leaf 同样必须拥有正数 direct timing 或 typed fusion ownership；绝不能把映射缺失的计算 leaf 降级成 `structural`。repeat、conditional selection 等纯控制节点可以继续保持 `structural`。标记为 `module_boundary` 的 Execution IR 节点（例如 TP output collective）不得计入紧邻的 Model IR 模块 roll-up，但必须计入外层 decoder／scheduler roll-up。如果同一个 detail view 被多个 parent 复用，Compiler 不得猜测归属；必须由 occurrence scope 或显式 fusion/event-set binding 消除歧义。Fused semantic node 只显示 `fused into <timing owner>` 及其 fusion/evidence 链接，不能复制 owner 的标量时间；只有 timing owner 显示 measured value。Composite parent 可以单独显示明确标记的 `inclusive_rollup`，其含义是 descendant production event 的并集，不能与 descendant 相加。

凡是被 repeat 或多个上下文复用的模块边界，都必须声明
`timing_scope_contract`。Contract 必须写明 composite target、真实 production
owner、精确上下文过滤条件（例如 `substage=attention`）、期望 occurrence
数量以及 drill view。Mapper 必须把
这些坐标保留在每个 event 上；materializer 只能对满足过滤条件的真实物理
区间做并集来生成 parent 时间，绝不能把一个 profile-wide owner 标量复制给
多个 parent。Occurrence 缺失、重复或作用域不一致时，profile 必须 fail
closed，不能静默生成数字。

重新 materialize 时，必须先从 timeline event 中移除所有旧的 drill/scoped
派生父目标，再依据当前 Model IR 重新建立 ancestry；IR boundary 已变更时，旧
parent tag 绝不能残留。

Catalog compiler 默认强制校验可执行 drill 父节点的 rollup 闭包。未来模型若
在 measured descendants 上仍把父节点发布为 structural，将直接拒绝构建；
只有显式声明为 control/state boundary 的语义操作，或经过书面 pipeline
exception，才能无 timing。

这套 contract 与测试必须是模型无关的。测试至少证明：每个已接受 parent
具有准确的 occurrence 集合；改变任一上下文坐标会改变归属；active 等于匹配
区间并集，residency 等于匹配 duration 总和；parent 不属于 leaf fusion
group；drill view 展示的是同一份 scoped owner evidence。

### Stage 8 — 构建 Timeline Hierarchy

Timeline 明确分成两个层次：

- **证据层**保留每个真实 CUDA stream，以及每个 event 的精确
  `start_us`、`duration_us`、`stream_id`、IR attribution 和 formal-step
  timing fields；
- **呈现层**默认使用 concurrency-aware compact activity lanes。只有 stream
  role 兼容且 activity segment 在时间上不重叠时，才允许复用同一行；该投影
  绝不能修改原始证据或参与 timing 计算。

Compact 投影遵循以下与模型无关的规则：

- main compute 固定在第一条 compute lane；
- communication、copy/transfer，以及 catalog 显式声明的稳定 role family
  与 generic compute 分开；
- 时间上有重叠的 physical-stream segment 绝不能放到同一 compact lane；
- 同一 physical stream 内的重叠（包括 PDL）使用 kernel sublane 展示，不能
  被错误串行化，也不能伪造新的 physical stream；
- 点击 compact lane 会展开所有且仅有为该 lane 提供 activity 的真实 stream；
- **physical streams** 模式始终保留为无损 debug view。

在每条呈现后的 stream/activity lane 内：

- 上层 IR lane 显示稳定的 layer/module-level ownership；
- 下层 kernel lane 显示单个 kernel，并使用 kernel-family color；
- 重叠 interval 放入额外 lane，而不是覆盖绘制；
- PDL 或其他有意 overlap 保持为 overlap，不能被错误串行化；
- 选择 IR interval 或 kernel 时，都能跳转到 canonical architecture drill path。

Compact lane 数量只属于 presentation metadata，绝不能参与 module rollup、
active/idle/residency 计算、stream identity、Execution fingerprint 或
eager-to-production attribution transfer。

Layer/module label、timeline tier、color 和 drill path 都由 catalog metadata 编译生成。Viewer 不能从 `qsa`、`eagle_mtp` 或某个 framework class name 推断这些信息。

### Stage 9 — 编译 Static Bundle

Compiler 只能组合彼此兼容的 document：

```text
Model IR
  + matching Execution Plan / fingerprint
  + matching Implementation Binding
  + matching Profile and Timeline
  -> deterministic static bundle
```

Bundle 包含 canonical navigation index，因此 Architecture、Timeline、Split、source link、profile selector、generation entry view 和 deep link 都使用同一组 ID。

### Stage 10 — Acceptance

只有通过所有适用 gate 后，一个 profile 才能以 deliverable 状态默认展示。

### Stage 11 — 可选 SoL 与 Gap Analysis

1. 选择与 measured Profile 完全一致的 execution fingerprint、phase、graph mode、realized workload 和 hardware spec，编译不可变 Workload IR 与 fingerprint。
2. 为每个稳定 node 编译跨 framework Cost IR：resolved problem shape、useful work、compulsory bytes、communication payload/state traffic、repetition 和 operator family；出现 framework symbol 或 kernel name 时拒绝。
3. 将 Cost IR 编译为 resource-transition DAG。Transition 内资源并发取下界，显式依赖串行；输出每个 transition 的 top-N limiter 与 local critical path。
4. Hardware adapter 提供 dtype-specific Tensor Core、memory hierarchy、SFU 和 interconnect ceiling，先生成不使用经验效率的 physical `ideal_ms`。
5. 从 code review、eager trace、production trace 或 microbenchmark 建立版本化 Kernel Plan；fusion、persistent/cache reuse、tile、launch 和 sync 必须成为 plan identity 的一部分。
6. 只有 exact workload shape 且 kernel-plan fingerprint 完全匹配的 calibration surface 才能生成 attainable P10/P50/P90；shape-only 系数和固定 operator-family efficiency 均 fail closed。
7. Launch/sync overhead 必须由 plan 中的实际事件数和带证据的 hardware structural model 推导，不能使用无来源的每节点固定微秒数。CPU/scheduler idle 只属于 runtime gap，不写进 kernel projection。
8. 在 Execution IR dependency DAG 上计算 critical path；coverage 不完整时只能称为 modeled-subgraph critical path，不能称为 full-step SoL。
9. 生成 Gap Report，并执行 work/byte conservation、plan identity、hardware provenance、exact-shape microbenchmark、holdout、`ideal <= silicon`、`projection P10 <= silicon` 和 cross-framework contract gate。

## 5. Adapter 边界

Reusable engine 负责：

- schema 和 cross-document validation；
- 确定性的 Execution IR compilation 和 fingerprinting；
- trace parsing 和 interval arithmetic；
- exact-sequence attribution transfer；
- profile/timeline generation 和 bundle compilation；
- acceptance report。

**Model adapter** 只负责 model-specific semantic alias、重复 layer schedule、state scope 和真正独有的 signature。

**Framework adapter** 负责某个 framework 的 stack format、annotation convention、graph capture convention、runtime helper scope 和 source-link extraction。

**Execution/backend adapter** 负责识别 trace 中的 collective 和 backend-specific kernel，但必须把它们映射到已经声明的 execution contract。

Adapter 可以增加 evidence 和 alias，但不能修改 canonical IR，也不能向 Viewer 添加 special case。

## 6. Catalog 与 Artifact 目录结构

```text
catalog/<model>/
  model_ir.yaml
  execution_paths/<plan>.yaml
  bindings/<framework>-<commit>-<backend>.yaml
  profiles/<execution>/<binding>/<profile>.yaml
  sol_manifests/<workload>.yaml          # SoL adapter inputs for measured profiles
  pipeline.yaml                         # run manifest(s) and requested targets

catalog/hardware/<gpu>.yaml             # shared theoretical ceilings + calibration

schema/v2/                              # executable persisted contracts
src/llm_arch_v2/
  compiler.py
  attribution.py
  metrics.py
  timeline.py
  adapters/
    models/<model>.py
    frameworks/<framework>.py
    backends/<backend>.py

current/<profile-task>/                 # raw trace、log 和 intermediate evidence
docs/<model>_v2/                        # 只存放生成后的 static bundle
```

Raw trace 和 intermediate task material 不放进 repository。Catalog document 记录它们的 content hash 和可解析的 local/artifact reference。

最终计划只保留一个入口命令：

```bash
python3 scripts/run_pipeline_v2.py \
  --manifest catalog/<model>/pipeline.yaml \
  --target deliverable
```

`scripts/build_v2.py` 继续作为 prepared catalog data 的 compiler。上面的 orchestrator 是本规范 review 通过后的下一项实现工作；它会调用 capture、attribution、validation 和 compilation stage，但不会形成第二套 pipeline。

## 7. Profile Matrix 与扩展策略

首先以 pure TP 作为默认 reference。当仅 hardware、workload、backend implementation 或 measurement 变化时，在已有 execution fingerprint 下增加 Profile；当 framework/source/backend code 变化时增加 Binding；当分布式执行 contract 变化时增加 Execution Plan。

推荐扩展顺序：

1. pure TP：prefill，以及 CUDA Graph decode BS 1/16/64/256；
2. 使用最广或性能最好的 Attention-DP/CP 路径；
3. MoE EP 以及选定的 communication/GEMM backend；
4. Attention DP + MoE EP 等有实际价值的组合；
5. 绑定到相同 Model/Execution IR 的其他 framework；
6. MTP 等 optional generation mode，继续使用相同 contract，只增加独立 profile dimension。

每条不同 code path 都必须独立 profile。即使不同路径中的 Model IR node 名称相同，也不能复制测量结果。

### 7.1 跨 Framework 的精确 Comparison Contract

哪些 profile 可以比较，必须由 Compiler 决定，不能由 Viewer 猜测。Compiler
对规范化、framework-neutral 的 `comparison-contract.v1` 做 hash，内容包括：

- Model IR identity、generation mode、phase、formal-step semantics，以及显式
  authored 的 `comparison_variant`；
- dtype、quantization，以及显式 authored 的 backend-significant model config；
- TP/DP/CP/EP 参数、规范化 workload/realized shape、scheduler contract、CUDA
  Graph 状态，以及相关 hardware/interconnect identity。

Warmup/formal request 数、native step counter、文件名、selected rank、source
commit 和 timing value 属于 profiling procedure/provenance，不进入 workload
identity。凡是会改变实际执行问题的字段都必须先规范化进入 contract；绝不能从
profile id、label 或 framework name 推断。一个 comparison contract 下，每个
implementation 最多只能有一个 profile；出现歧义时 Compiler 必须 fail closed。
没有 exact profile 的 implementation 仍可见，但必须禁用并显示缺失维度/原因。

Execution fingerprint 存放在 comparison contract 的**旁边**，而不是 hash
进 workload identity。若选中的 framework 共享同一个 validated fingerprint，
Viewer 可以同时在 Model IR 和 Execution IR 上比较；若 fingerprint 不同，Viewer
只能共享 Model IR，并为每个 framework 保留独立 Execution IR overlay，必须禁用
任何会把这些 plan 合并成虚假“共享 Execution IR”的呈现。

每个 Binding 都必须编译出 canonical `framework_id`（`sglang`、`vllm` 或
`tensorrt_llm`）。Viewer 只能读取 Compiler 生成的 contract/profile index 和该
ID，禁止在浏览器里按 label/name 匹配。Architecture comparison 只画一次 Model
IR geometry，并为每个 framework 显示一行 compact evidence；Timeline comparison
按固定 SGLang/vLLM/TensorRT-LLM 顺序堆叠每个 framework 自己的 production
artifact。可同步 normalized visible range 和 IR selection，但每个 profile 内的
physical stream ID、event、fusion ownership 和 timing 必须原样保留。Fused member
仍然只能链接到该 framework 唯一的 timing owner，不能复制 owner 的标量时间。
当共享 Architecture node 要求居中时，每张 comparison timeline 必须分别将该
framework 自己最近的 matching event 居中；后续 normalized-range 同步消息不能覆盖
这些 framework-local center。右侧 detail 同样必须隔离：按相同固定顺序为每个
implementation 渲染一个 pane。Architecture selection 会用各 framework 自己的
node evidence 更新所有 pane；点击某个精确 kernel 时，来源 pane 保留该 kernel 的
精确 evidence，其他 pane 显示共享 IR target 的对应 module evidence。
每个 framework evidence row 必须读取该 profile 的 enriched cell，而不是 roll-up
之前的 raw state：只要已选 descendant 存在 timing，可执行 drill parent 就必须
显示该 framework 的数值化 `inclusive_rollup`，不得显示 `structural`。Graph 和
detail pane 中每一条 `fused into` 都必须是真正可导航的超链接，跳转到该
framework 的 architecture owner。

## 8. Acceptance Gate

### Structural

- 所有 document 通过 JSON Schema 和 cross-document reference check。
- Model IR ID 稳定，所有 drill target 都能解析。
- 已选中的可展开 compute/module 节点以及 runtime-bearing primitive leaf 绝不能显示为 `structural` boundary；它必须拥有正数的 measured／inclusive-union timing、是明确指向唯一 timing owner 的 fused member，或者具有不相交的 `fused_by_occurrence` partitions。只有显式定义的 boundary/control/state 节点，以及 `not_selected`、`disabled`、`out_of_scope` 等明确未激活分支可以没有 timing。
- 每条 edge 都有 identity、shape、layout、dtype 和 state lifetime；每个语义节点都有 authored equation。缺少 operation、equation 为空，或出现 `None = None(None)` 一类 fallback artifact 时，编译必须直接失败。
- 每个新建或发生修改的 catalog，都必须为每个声明的 dimension 提供唯一 authored
  symbol contract。Constant、profile-runtime value 和 stage-dependent resolution
  都必须通过 schema check；未知 stage 和无 scope 的动态推断必须编译失败。尚未迁移
  symbol dictionary 的 legacy catalog 仍可兼容加载，但 Viewer 只能解析字面数字常量，
  所有动态 symbol 必须明确显示为 unresolved，不能猜测。Viewer test 必须覆盖
  constant、所选 profile batch 及其切换刷新、stage-scoped draft、unresolved fallback
  和 standalone parity。
- 编译后节点的 Inputs/Outputs 与 incident edge contract 完全一致；每个 drill boundary 都通过 parent/child 或 scoped-lifecycle closure。
- Viewer 从编译后的 contract 展示 Inputs、Transition / Equation 和 Outputs，而不是解析 label。
- 编译后的 Execution IR fingerprint 与 Binding、Profile 一致。
- Binding 中包含针对该 structural fingerprint、source revision、phase 和 execution path 的 passing eager-validation attestation。
- 每个 communication node 都声明 collective、group、payload 和 result。
- Viewer code 不依赖任何 framework/model-specific identifier。
- 每个 Compiler 生成的 comparison contract 都必须无歧义，选中的 profile 必须
  精确复现该 contract；不同 Execution IR fingerprint 只能共享 canonical Model
  IR，不能被合并。

### Attribution

- 每个 production code path 和 phase 都有 eager semantic evidence。
- 每个 mapped production event 都保留 eager event ID、transfer rule、confidence 和 occurrence scope，并且 event-to-IR 双向索引完全闭合。
- Raw timeline attribution audit 必须覆盖每个 production event：每项要么有 IR/fusion binding，要么同时具有 `support_class` 和 `support_reason`；疑似 GEMM、attention、MoE、normalization、convolution 或 collective 的语义 kernel 在 IR 外必须为零。
- 每个 single-owner `fused` node 必须且只能属于一个 fusion group，group owner 与 `included_in` 一致；每个 group 都明确 exact interval、equal aggregate event set 或 member-event coverage 语义以及可 review 的 evidence scope。Multi-owner `fused_by_occurrence` node 则必须有不相交的 physical-event partitions，并且每个 partition 只有一个合法 owner。
- `fused` node 不得携带 standalone scalar timing field。每个 group 只有 owner 承载 measured production timing；如果同一节点同时声明 fused state 和独立 `node_metrics`，编译必须失败。
- Viewer 的 node card 与详情必须显示 timing owner、covered semantic contract、mapping proof 和 occurrence/aggregate scope；泛化的 `fused implementation` 文字不能作为可交付结果。
- 每一处 `fused into <timing owner>` 都必须是由编译后 metadata 驱动的 architecture 链接：默认跳到 timing owner 自身；若 timing owner 是不可见的聚合节点，则必须在 Model IR 中显式声明 `architecture_target`，Compiler 将它物化为 `architecture_owner`。点击后必须沿该目标的 canonical drill path 打开、居中并选中。Viewer 不得从显示文字猜测目标，也不得加入 model-specific 跳转逻辑；目标缺失或从 profile `entry_view` 不可达必须在编译／发布阶段失败，不能退化为无法点击的普通文字。
- 发布时的真实浏览器验收必须枚举每一条实际渲染的 fused row，为每个已选 implementation 验证 owner hyperlink；对于只存在于 Execution IR 的 owner 也要实际点击，并证明 Viewer 打开且选中了精确 owner。任意一条退化成纯文本的 `fused into` 都必须让发布失败。
- 完整 candidate Execution IR 已经使用 eager stack、shape、invocation multiplicity、state transition 和 collective order 完成 reconciliation。
- 每个 measured event 都属于以下状态之一：已映射、显式 fused/shared，或带类型和原因的 framework/runtime support。没有完成 production attribution closure 的 Model/Execution IR node 必须标记为 `mapping_incomplete`，不能伪装成 measured zero，也不能把 generic `unmapped` 当作 node 状态。
- Release build 中必需节点的 `mapping_incomplete` 必须为零；每个 fused
  语义叶子都必须声明承载共享执行时间的 measured interval owner。
- Collective ordering 和 scope boundary 与 Execution Plan 一致。
- Exact-sequence transfer 通过；attribution 不跨越 collective、generation scope、layer invocation 或 formal-step boundary。
- 按 duration 和 count 报告 coverage，未知 evidence 不能被隐藏。

### Timing

- Formal iteration boundary 已通过校验。
- Passing `window_selection` artifact 证明先完成了 unprofiled baseline，并且第二次运行真正捕获到了选定的 formal steady-state interval。
- Warmup/formal request count 分别等于 `3 × C` 和 `1 × C`；requested/realized per-request length、scheduled batch composition 和 target concurrency 均已记录并与 profile identity 一致。
- Pure phase profile 的 selected interval 内没有 mixed phase、admission/completion churn、preemption、retraction 或 recomputation。
- `residency >= active_gpu`，且 elapsed-envelope 恒等式在数值误差范围内成立。
- Rank policy 明确，不能把 tail/critical-rank wall time 与 aggregate residency 混淆。
- Profile identity 中明确展示 prefill/decode、eager/CUDA Graph、batch size 和 generation mode。

### Presentation

- 每个 IR node 必须显示以下一种状态：measured、fused/shared、structural、state-only、not selected for this phase，或会阻止发布的 mapping-incomplete。类型化 runtime/support 只显示在 timeline；无法解释的空时间属于失败。
- 任何拥有 measured descendants 的可执行 drill node 都必须有数值化的 `inclusive_rollup`；只有纯控制或 state boundary 可以没有 timing。
- Rollup 测试必须证明重叠 descendant event 使用区间并集而不是求和；复用 detail view 时，在 profile scope 选定唯一 parent 或提供显式 many-to-many event-set binding 之前必须 fail closed。
- 选择 Architecture node 时高亮所有匹配的 Timeline event；选择 Timeline event 时自动展开、居中并选中精确的 Architecture leaf。
- Multi-framework 验收必须覆盖 1/2/3 个选择、固定顺序、缺失 exact match、相同
  与不同 Execution IR、URL reload/back/forward、framework-specific fusion link、
  range 同步、每个 framework 独立居中、一套 detail pane 对应一个 framework、
  精确 kernel 与 peer module detail，并通过真实 SVG/Canvas click 验证双向跳转。
- 发布验收必须真实点击渲染后的 SVG node 和 Canvas kernel/owner lane，分别验证两个方向；只调用内部导航函数不算通过。测试还必须证明被点击的 kernel 是唯一保持实色的 kernel slice，其他无关 slice 均被淡化。
- Stream、overlap、idle、module wall envelope、active GPU 和 residency 可以分别查看。
- 默认显示 compact activity lanes，同时允许切换到精确 physical streams。
  两种模式切换前后，event fingerprint（`start_us`、`duration_us`、physical
  `stream_id`、IR targets）以及 formal-step timing fields 必须逐字节不变。
- 所有包含 production event 的 physical stream ID 都必须被 compact 投影覆盖；
  compact lane 数不能大于 physical stream 数；同时活跃的不同 physical-stream
  segment 不能共享一条 compact lane。同一 stream 的 overlap 必须保留在
  kernel sublane 中。
- 真实浏览器验收必须打开每个 accepted profile、切换两种 stream mode，并
  点击 compact row 验证它能展开精确的 physical-stream contributors。全量
  artifact contract test 必须覆盖每个 formal step，因此新编译的 model/profile
  会自动进入该 gate，无需增加 viewer-specific 分支。
- Raw trace/Perfetto handoff 必须经过 content-hash 校验。

### Reproducibility

- 记录 source/config/run/baseline-log/window-selection/trace hash 和 producer version。
- 对同一个 catalog 的 rebuild 必须是 deterministic 的。
- 在认定 adapter 或 viewer change 具有通用性之前，至少使用第二个 framework 或第二个 model fixture 验证 generic path。
- Release 必须从 canonical catalog 重新构建，不能复制 generated checkpoint 或旧 viewer bundle。CI 必须重新 build，并在 checked-in generated output 与 rebuild 结果不一致时失败。
- Semantic refinement 必须与对应的 schema、source ledger、compiler、binding、mapping、test 和 generated bundle 原子化进入 release。Release gate 必须校验预期的 `semantic_revision`、semantic-ledger audit fingerprint 和必需的 primitive drill/view ID；只存在于未进入 release ancestry 的分支不能被视为已经交付。
- 发布 bundle 中的 semantic revision 和 primitive view inventory 必须与 source catalog 一致。Release validation 必须实际打开发布产物并检查代表性的 primitive path，不能只依赖本地 source test。

## 9. 人工 Review Gate

以下情况必须进行人工 review：

- Model IR semantics 或稳定 data flow 改变；
- 引入新的 Execution Plan 或 boundary contract；
- eager-to-production transfer 存在自动流程无法解决的歧义；
- eager reconciliation 与 candidate Execution Plan 不一致；
- 新 adapter 需要修改 generic contract。

如果新 Profile 使用已有 fingerprint，并且 exact transfer 全部通过，则不需要重新进行 architecture review。

## 10. 当前迁移状态

Repository 目前已经通过同一个 V2 compiler 和同一个 Viewer 发布六个 catalog：Qwen 4.0、Qwen3.5、GLM-5.2、GLM-5.3-Flash、Kimi K3 和 DeepSeek V4 Pro。Catalog 已经包含 Model IR、Execution Plan、Implementation Binding、accepted Profile，以及在对应 framework 支持时来自 SGLang、vLLM 和 TensorRT-LLM 的 immutable Timeline evidence。Shared execution fingerprint、eager-validation attestation、fusion ownership、semantic closure、Timeline attribution、同步 comparison 和 real-browser audit primitives 都已经实现。

M0 新增 `scripts/release_audit.py` 作为 model-neutral 的 release 入口。Static level 会重新编译 catalog、比较 published bundle 的精确内容、验证公开 model inventory 和 content-addressed Timeline artifact，并对无法解释的 production kernel fail closed。Release level 会进一步执行真实 browser audit。仅通过 static gate 不会被标记为 release-ready。

剩余工作统一记录在 [ROADMAP.md](ROADMAP.md)。当前 M0 debt 不会被隐藏：统一 static audit 已经接受 Qwen 4.0、Qwen3.5、Kimi K3 和 DeepSeek V4 Pro；GLM-5.2 与 GLM-5.3-Flash 仍需修复 Timeline attribution，之后该 gate 才能成为 CI mandatory gate。随后还需要完成 full browser gate、published acceptance summary、generic adapter boundary 和 release identity。Manifest-driven `run_pipeline_v2.py` orchestrator 属于 M1，而不是当前已经可用的命令。

已经移除的 Qwen3.5 trace-first/manual pipeline 不再是第二条受支持路径。旧实现中有价值的部分已经保留在本规范中，包括：冻结输入、可复用 trace parsing、source/callsite validation、artifact provenance、config-driven validation 和单一 orchestration command。旧流程中“由 runtime skeleton 定义 architecture”的行为不再保留。
