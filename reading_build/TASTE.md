# 论文阅读页的 Design Read 与固定准则

这份文件是**渲染器的设计契约**。结构 JSON 进来，HTML 出去，中间所有视觉决定都在这里，
不许在渲染代码里临时发明。

它按 taste-skill 的写法组织：先声明 design read 和三个 dial，再锁死 token，
再逐面给规则，最后是机械可查的 pre-flight。凡是从 skill 照搬的，用 skill 的原词；
凡是不适用或被推翻的，写明理由 —— skill 自己要求"dial 值必须说明来源，不能静默用基线"。

---

## 0. Design Read（一句话）

> Reading this as: a **reading edition** of a research paper for a technically literate
> reader on a laptop or phone, with a **quiet, high-contrast dark publication**
> language, leaning toward **native CSS + one deep navy surface + a single cool accent**.

不是 landing page，不是 dashboard，不是产品 UI。它是一个**阅读仪器**：
页面唯一的工作是让一篇十页的论文在屏幕上比 PDF 好读。

## 1. 三个 dial

| dial | 值 | 理由 |
|---|---|---|
| DESIGN_VARIANCE | **3** | skill 给 "editorial / blog" 的默认是 6。这里**故意下调**：论文自带章节结构，版面应该服从它；不对称版式会让人在两种阅读节奏之间来回切换。变体空间只在 masthead 一屏。 |
| MOTION_INTENSITY | **2** | 阅读场景。全套动效只有两处：目录滚动高亮、手机端目录折叠。skill 说 "motion must be motivated"，这里两处都算 state transition，其余一律静止。 |
| VISUAL_DENSITY | **3** | 正文 18px / 1.6 行高 / 45rem 行宽，是"画廊"侧的密度，不是"驾驶舱"。 |

**推翻基线的部分**：skill 的 8/6/4 是给营销页的。这里三个值全部下调，理由如上，
不是忘了改。

## 2. 锁（Locks）

这四条是硬锁。渲染器里任何一处违反都算 bug，不是"设计选择"。

### 2.1 Theme Lock
全页**只有一个主题**：深色。不出现浅色块，不出现"某个 section 反色"。
`color-scheme: dark` 写死在 `:root`。唯一的"反相"是论文插图（见 3.4），
那是把纸面图放进深色页的手段，不是主题切换。

### 2.2 Color Consistency Lock
**一个强调色**，`--accent: #8aa4ff`。它只允许出现在四个位置：

1. 章节编号（`h2.sec .n`）
2. 图表编号（`.capnum`）
3. 目录当前项
4. 正文链接下划线

除此之外不许有第二个彩色。没有绿色 success、没有橙色 warning、没有第二强调色。
强调色的浅版 `--accent-ink` 只用于 hover，属于同一色族。

### 2.3 Shape Consistency Lock
**radius 全局 0**。纸张没有圆角。按钮、图框、表框、代码块一律直角。
唯一的"软"是 hairline（1px `--hairline`），它不是形状。

### 2.4 Typography Lock
三族，各司其职，**不许出现第四族**：

| token | 字体 | 只用在 |
|---|---|---|
| `--serif` | Newsreader, Charter, Georgia | 正文、标题、摘要 |
| `--sans` | Space Grotesk, system-ui | 元信息、图注、目录、页脚 |
| `--mono` | JetBrains Mono | 编号、数字、代码、来源行 |

skill 的 "serif discipline" 在这里**是例外而非默认**：论文正文用衬线是有理由的
（长段落正文、印刷品来源、读者预期），且不是 Fraunces / Instrument Serif 那两个
LLM 默认衬线。这一点写下来，是因为 skill 要求"能说出为什么是这个衬线"。

## 3. 逐面规则

### 3.1 Masthead
- 一屏之内：kicker（可选，全页**最多一个**）、标题、作者行、来源行。
- 标题 `clamp(2rem, 5vw, 3rem)`，**最多两行**。超长标题由渲染器截断加省略号，
  完整标题进 `<title>` 与 masthead 的 `title` 属性，不靠缩字号硬塞。
- kicker 用 mono + uppercase + `letter-spacing: .12em`。**全页只此一处**，
  正文里的章节标题一律不加"眉标"（skill: eyebrow 配额 = ⌈章节数/3⌉，这里直接取 1）。

### 3.2 目录（TOC）
- 只列**真实存在的章节**，不造 "00 / INDEX" 或 "001 · Methods" 这类编号眉。
- 层级最多两级（h2 / h3），深度超过就截断，不做三层缩进。
- 桌面端 sticky，内部滚动；手机端折叠为 "Contents ▾"。
- 不用图标。折叠指示用文字 + `▾`，因为 skill 禁止手绘 SVG 图标，
  而为一个三角引入整套图标库不值。

### 3.3 正文
- measure 45rem（≈65ch），字号 18px，行高 1.6。
- 段落间距 0.9em，**不用 `<br>` 造间距**。
- 首段 drop cap：只作用于摘要后的第一段，CSS `::first-letter`，不用额外 span。
- 公式小串用 `.fnl`（15.5px, ink-soft），不试图重排数学。
- 脚注、参考文献是正文的一部分，不是附录卡片。

### 3.4 图与表
- **只用从 PDF 裁出来的真图**。禁止用 div 画"示意图"，禁止占位色块。
  （skill 的 "div-based fake screenshots are banned" 直接适用。）
- 图版统一 `filter: invert(1) hue-rotate(180deg)`，这是 Theme Lock 的一部分：
  纸上的黑线图在深色页里必须是亮的。表图额外压一点对比度，避免反相后过曝。
- 每张图必须有 `alt`（用 caption 首句），必须有 `width`/`height`，防 CLS。
- 宽表 `overflow-x: auto`，**不缩字号**。缩到读不清的表等于没有。
- 图注用 sans 13px，编号用 accent + mono。图注在 `<figcaption>` 里，不浮在图上的标签。
  （skill 的 "no pills/labels overlaid on images" 适用。）

### 3.5 动效
- 两处，且只有两处：
  1. 目录滚动高亮（IntersectionObserver，不用 scroll 事件）
  2. 手机端目录折叠（一次性展开/收起）
- 全部在 `@media (prefers-reduced-motion: reduce)` 下退化为静止。
- 不许出现 `window.addEventListener("scroll")`（skill 明令禁止）。
- 不许有入场动画、视差、marquee、计数动画。

### 3.6 响应式
- 断点 940px：以下目录折叠、正文满宽、图满宽。
- 320 / 360 / 390 px 一律零横向溢出（已有 `tt_mobile_check.js` 作闸）。
- 不用 `h-screen`，用 `min-h`。
- 每个多列结构在当前组件里显式写出 `< 768px` 的塌缩，不靠"Tailwind 会处理"。

### 3.7 文案
渲染器几乎不写文案，只写固定标签：`Contents`、`Source`、`Figure N`、`Table N`、
印刷年份。因此 skill 的 copy self-audit 大部分自动通过。仍然禁止：

- 装饰性状态点、`v0.6` 之类版本戳、城市/时间/天气条
- "quietly"、"field notes"、"on our desks" 这类表演式小标签
- 任何 em-dash（`—` / `–`）出现在**我们自己写的 chrome 里**

**边界**：论文原文里的破折号是原文，保留。em-dash 禁令只约束渲染器自己产出的
标签、页脚、目录文案、图注前缀。

## 4. 明确不适用的 skill 规则

| 规则 | 为什么不用 |
|---|---|
| hero 视口 / CTA / 表单 / logo 墙 / bento / 定价 / 跑马灯 | 这个页面没有这些面 |
| 官方设计系统包（Fluent / Carbon / Primer…） | 单篇静态文档，原生 CSS 足够；引入框架只会增加产物体积 |
| "必须用真实图片" | **满足**：每张图都是从 PDF 裁的真实图像 |
| 图标库 | **零图标**，比"选一个库"更干净 |

## 5. Pre-Flight（机械可查）

渲染器输出必须全过。括号里是检查方式。

- [ ] chrome 文本里 em-dash 计数 = 0（`grep -c '—\|–'` 只扫模板产出的标签）
- [ ] 彩色值只有 `--accent` 一族（扫 CSS 里的 hex）
- [ ] 所有 `border-radius` = 0
- [ ] 字体族 = 3（扫 `font-family`）
- [ ] 无 `#000000` / `#ffffff` 纯值
- [ ] `--ink-faint` 对 `--bg` 对比度 ≥ 4.5:1（`#7b8294` on `#06070a` = 5.24:1）
- [ ] 正文 measure ≤ 45rem
- [ ] 320/360/390 零横向溢出（`tt_mobile_check.js`）
- [ ] 每张 `<img>` 有 alt + width + height
- [ ] 目录锚点 100% 可解析
- [ ] 每页 uppercase 眉标 ≤ 1
- [ ] 无 `window.addEventListener('scroll')`
- [ ] `prefers-reduced-motion` 覆盖到全部两处动效
- [ ] 移动端塌缩在组件内显式声明

## 6. 落到渲染器

- 模板 CSS 已从 FDP 页里**抽出来冻结**成 `theme.css`；`render.py` 内联它，
  不再从某个手写页里切 `<style>`，改一页不会波及其余。
- 每种 role 对应一种标签（`h2.sec` / `h3.sub` / `p` / `p.rawcap` / `figure.plate` /
  `figure.table` / `p.fnl` / `p.kw`），`render.py` 只做映射，类名就是本文件 §3 的词汇。
- 渲染器**不生成任何论文原文之外的句子**。所有非原文文本来自固定常量表。

## 7. 已修项（留档，防止回退）

1. `--ink-faint` 原为 `#6a7080`（4.07:1，未过 AA），现为 `#7b8294`（5.24:1）。
2. 模板 CSS 已抽离为 `theme.css`（见 §6）。
3. 目录折叠按钮已有 `aria-controls="toc"` + `aria-expanded`。
4. `render.py` 早期用的 `.figbox` / `.tablebox` / `p.cap` / `.lead` 在 `theme.css`
   里**没有对应规则**（图会以原始像素宽顶出视口，手机端横向溢出）。现在统一用
   `figure.plate > .plate__frame`、`figure.table > .tablewrap`、`p.rawcap`、
   `.opening`，与主题词汇一致。

---

**结论**：这张页面的"漂亮"不是靠加东西，是靠**锁**——一个主题、一个强调色、
一个字号体系、两处动效、零图标。渲染器要做的只是把这套锁原样套到任何一篇论文上。
