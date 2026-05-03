# arch-viewer — DeepSeek-V4 architecture × profile 交互式视图

> STATUS: ACTIVE (Phase 0)
> 上层 task: `current/deepseekv4-docs-code-learning/`
> 关联: `../trace-kernel-learning/` (profile 数据来源), `../DeepSeek V4 perf study.pdf`

把 DeepSeek-V4 的架构 (CSA / HCA / NextN-MTP) 画成可缩放、可折叠展开的
**模块化"地图"**，每个模块叠加多份 profile 数据 (sglang prefill / decode
实测、PDF DLSim projection、未来 vllm 实测), hover 显示 ms 数字 + kernel
breakdown + code links + 已知优化项。

## 设计目标

- **模块化** — 节点分 4 级 (model → layer → module → leaf), 任何一级都
  能展开/收起, 像 Google Maps semantic zoom
- **数据驱动** — 架构图 / table / tooltip 全部从 IR (yaml) + profile
  yaml render, 没有 hardcode
- **可拓展** — 换模型只换 IR + config, 渲染器不动. 加新 profile 数据
  源只 append 一个 yaml
- **代码联动** — 节点带 `code_links: [file:line:symbol]`, 旁边能 click
  to open

## 仓库布局

```
arch-viewer/
├── README.md                              # 本文
├── AGENT_LOG.md                           # 工作日志
├── ir/
│   ├── schema.yaml                        # IR 格式定义 (节点 / 边 / profile)
│   ├── stages.yaml                        # PDF 官方 stage 命名 + trace alias
│   ├── config.v4_pro.yaml                 # V4-Pro shape 实例化
│   ├── config.v4_flash.yaml               # V4-flash shape 实例化
│   ├── arch.dsv4.yaml                     # 架构 IR (hierarchy + edges)
│   └── profiles/                          # 各份 profile 数据 (yaml)
│       ├── sglang_v4_pro_prefill.yaml
│       ├── sglang_v4_pro_decode.yaml
│       └── pdf_projection_v4_pro_ctx_GB300.yaml
├── build/
│   ├── build_view.py                      # IR + profiles → enriched JSON
│   └── parse_trace_csv.py                 # trace-kernel-learning CSV → profile yaml
└── out/
    ├── arch_data.json                     # 渲染数据
    └── arch_view.html                     # 单文件 viewer (Cytoscape via CDN)
```

## 复现

```bash
# 一次性: 把 trace-kernel-learning 的 CSV 转成 profile yaml
python3 build/parse_trace_csv.py

# 主流程: 拼装 enriched JSON
python3 build/build_view.py

# 看效果
open out/arch_view.html   # 直接打开就能用 (不需要 server)
```

## 部署到 GitHub Pages

整个 viewer 是**纯静态** (HTML + JSON + CDN ELK.js), 适合 GH Pages.

### 准备 (在你的 fork repo, e.g. `YAMY1234/deepseekv4-arch-viewer`)

```bash
# 1) 在 fork repo 里建一个 docs/ 目录 (gh-pages 默认从 main/docs 或 gh-pages branch 部署)
mkdir -p docs
cp out/arch_view.html docs/index.html
cp out/arch_data.json docs/

# 2) push
git add docs && git commit -m "deploy arch-viewer" && git push

# 3) GitHub repo Settings → Pages
#    Source: "Deploy from a branch"
#    Branch: main, Folder: /docs
#    几秒后 → https://YAMY1234.github.io/<repo>/
```

### code_link 自定义 commit / fork

`ir/source_map.yaml` 里改 `repo` + `commit` 一行即可重新生成所有
GitHub URL (无需改 IR 主体):

```yaml
source_map:
  - prefix: ""
    repo:    "sgl-project/sglang"     # 想换成自己 fork 就改这里
    commit:  "5031406e5c0e985499b4b4ae86e02b859ccf49b8"
    path_prefix: "python/sglang/srt/"
```

改完跑 `python3 build/build_view.py`, link 全部刷新.

### 注意事项

- 所有 fetch 都是相对路径 (`fetch("arch_data.json")`), GH Pages 子目录部署 OK
- ELK.js 通过 `unpkg.com` CDN 加载, GH Pages 上能直接用 (要离线版可以
  把 `elk.bundled.js` 一起 commit 进去, 改 `<script src=...>` 为相对路径)
- 不要把 sglang-dsv4 worktree 一起 commit — 只需要 `out/` 这两个文件 +
  `index.html`

## IR Schema 概要

### 1. 节点 (module)

```yaml
- id: csa_compressor          # 唯一
  label: "Compressor"         # 显示名
  kind: module                # model / layer / module / leaf
  parent: csa_mqa             # 嵌套关系 (compound node)
  variants: [csa]             # 在哪些 layer 变体里有 (csa / hca / nextn)
  code_links:
    - "models/deepseek_v4.py:170:Compressor"
  stage_keys:                 # 关联到 stages.yaml 里的 stage id
    - compressor_kv_score_proj
    - compressor_others
  trace_keys:                 # 关联到 trace 解析出的 stage label (alias)
    - "3a_compressor_outer"
    - "3b_compressor_kernel"
  pdf_slide_ref: 7            # 对应 PDF 哪一页
  optimizations:              # 已知优化项 (人工标)
    - id: opt_fuse_main_indexer_wkv_gate
      title: "fuse main + indexer wkv_gate"
      roi_ms_per_iter: 13
      difficulty: medium
      status: pending
```

### 2. 边 (tensor flow)

```yaml
- from: csa_input_x
  to: csa_compressor
  shape: "[B,S,D]"            # symbolic, 实例化时从 config 注入
  dtype: bf16                 # FP8/FP4/FP32/TF32, 影响着色
```

### 3. Stage (PDF 官方命名)

```yaml
# stages.yaml
- id: compressor_kv_score_proj
  label: "compressor: wkv_gate GEMM"
  pdf_name: compressor_kv_score_proj    # 跟 PDF table 一致
  trace_aliases:                        # 跟 trace-kernel-learning 输出一致
    - "3a_compressor_outer"
  description: "wkv_gate.weight @ x, 出 kv_score (B,S, 2*coff*dH)"
```

### 4. Profile (一份测量数据)

```yaml
# profiles/sglang_v4_pro_prefill.yaml
meta:
  source: trace
  variant: sglang
  config: v4_pro
  phase: prefill
  setup: "sglang dev branch, 855 recipe (4 iter trace), token-normalized to per iter"
  date: 2026-04-30
data:
  compressor_kv_score_proj:
    HCA:
      ms_per_iter: 4.1
      kernels:
        - {name: "linear_bf16_fp32 / nvjet_*", count: 30, avg_us: 137, total_us: 4100}
    CSA:
      ms_per_iter: 1.8
  compressor_others:
    HCA:
      ms_per_iter: 22.0
    CSA:
      ms_per_iter: 15.8
  ...
```

## Phase Roadmap

- [x] **Phase 0 (this)**: 静态单文件 HTML, 手填 IR, 三份 profile (sglang
      prefill / decode + PDF V4-Pro Ctx projection 一份)
- [ ] **Phase 1**: 加 vllm trace, 加 sglang `deepseek_v4_compressor`
      branch 实测; profile diff 视图
- [ ] **Phase 2**: pipeline 化 (trace → profile yaml → JSON → HTML 一键)
- [ ] **Phase 3**: 自动 IR (从 sglang `models/deepseek_v4.py` 的 nn.Module
      AST 抽出 hierarchy 草稿)
- [ ] **Phase 4**: 多模型 (V4-flash, R1, V3.2 复用同一套 viewer)

## 选型记录

- 渲染: **Cytoscape.js + cytoscape-expand-collapse** (compound nodes 原生
  支持嵌套, 插件支持 fold/expand). 备选: G6 (AntV)
- 布局: 顶层手动布局 (CSA/HCA/NextN 三栏并列), 内部 dagre 自动布局
- profile 数据格式: yaml (人工写舒服 + IDE 支持)
- shape symbolic: 用 `{B,S,D}` 等占位符, render 时按 config 替换具体值
- 单文件 HTML: 所有依赖走 CDN, 浏览器双击即用, 可以 git 进 repo 直接分享

## 当前限制

- IR 是手写的, 任何模型结构变化都得手动同步
- Phase 0 不支持 PDF table 的"配置矩阵"视图 (只展示 1-2 份 profile)
- 不支持 prefill / decode 之外的 phase (e.g. target-verify, draft-extend
  暂不画)
