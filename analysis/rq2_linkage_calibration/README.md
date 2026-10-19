# RQ2 匹配参数补充敏感性分析（20261012_v3_factorial）

在 `20261007_v2` 冻结分量分数的基础上补做三件事，主设定（地址权重0.8、门牌号60分、邮政编码系数0.85、阈值75）保持不变。全程离线运行，分类、RQ1和RQ3结果保持原样，`20261007_v2` 中的文件保持原样。

1. 把四个参数换算成规则：门牌号锁定、名称下限、邮政编码规则（`A.5.6补充分析与修订稿.md` 表C38）。
2. 四个参数全部取值的完整组合：9 × 7 × 7 × 7 = 3,087种设定，含原有75种，并做方差分解。
3. 阈值附近与边界配对的逐对判读（`boundary_pairs_review.csv`，作者需核对 `author_reading` 一列）。

## 核对

`verify.py` 用本文件夹的评分与分配函数重跑原有75种设定，筛选前后的配对与 `20261007_v2/oat_out/assignment_pairs.csv` 逐对一致（0处差异）。主设定重组得分与冻结得分的最大差值为0。读取设定参数时必须用 `float_precision='round_trip'`，否则0.85会读成0.8499999999999999，阈值85处的一个配对（得分恰为85.0）会消失。

## 文件

| 文件 | 内容 |
|---|---|
| `core.py` | 读取冻结分量、评分、带虚拟行列的最大权重分配（逐行复制自冻结 `geo_match.assign_hungarian`） |
| `verify.py` | 复现原有75种设定 |
| `factorial.py` → `factorial_runs.csv` | 3,087种设定的配对数、精确率、召回率、规则状态、名称下限、重复名称差距 |
| `factorial_pair_changes.csv` | 每种设定相对主设定新增或退出的配对 |
| `profile.py` → `profile_tau.csv` | 其余参数为主设定时，阈值60至90、步长0.1的配对数 |
| `profile4.py` → `profiles_fine.csv` | 其余参数为主设定时，四个参数各自的细网格（地址权重步长0.01、门牌号分值步长1、邮政编码系数步长0.01、阈值步长0.1），以及规则成立的区间 |
| `review.py` → `boundary_pairs_review.csv`, `near_threshold_pairs.csv` | 边界配对与阈值附近配对，含初步判读 |
| `figure.py` → `figure_A5_6_namefloor_zh.png`, `_en.png` | 图C1 |
| `selection_rule.md`, `selection_rule_amendment.md` | 校准的挑选规则（判读前写定）与看到结果后的一处补充 |
| `labels.py` → `universe270.csv`, `pair_readings_270.csv` | 270对配对的逐对判读（同一家、不同、不确定） |
| `calibrate.py` → `calibration_runs.csv` | 3,087种组合在判读上的错配、漏配与间隔 |
| `参数校准报告.md` | 校准结果、新参数的解释、RQ2结果变化与需要重跑的内容 |
| `newresults.py` → `new_rq2_numbers.json` | 原设定与校准后设定下，正文引用的RQ2数字（原设定逐项复现正文现有数字） |
| `新设定下正文修改清单.md` | 采用校准后设定时，方法论之外各章节需要修改的句子（中英文） |
| `summary_numbers.json` | 修订稿中各范围数字的来源 |
| `A.5.6补充分析与修订稿.md` | A.5.6中英文修订稿、新增表C38至C40、正文配套句子 |

## 复现

```
cd analysis/rq2_oat/20261012_v3_factorial
python verify.py
python factorial.py
python profile.py
python profile4.py
python labels.py
python calibrate.py
python newresults.py
python review.py
python figure.py
```

依赖 numpy、pandas、scipy、matplotlib。`comp.npz` 是首次运行生成的本地缓存，含逐对分量，与 `pairs.csv` 一样只保存在本地。

## 主设定下的RQ2结果与附录表格（2026-10-13补充）

选定参数（地址权重1.2、门牌号90分、邮政编码系数0.90、阈值60）下的正文与附录数字由以下脚本生成，路径相对仓库根目录解析。

| 文件 | 内容 |
|---|---|
| `recurrence_boot.py` → `name_recurrence_new.csv` | 名称重复性比例与按名称组的bootstrap区间（截点2、3、5；表C17、图4.1） |
| `allactive.py` → `allactive_results.json` | 全部245条Active Board记录作参照的敏感性分析（表C12） |
| `fig41.py` → `figure4_1_new.png`, `figure4_1_new.pdf` | 正文图4.1 |
| `appendix_numbers.py` → `appendix_numbers.json` | 表C11至C16与C18（匹配规格、分层、计数单位、合格配对、同址记录、未匹配Board记录） |
| `appendix_numbers2.py` → `appendix_numbers2.json` | 表C30至C32与A.5.6中的方差分解 |

```
python recurrence_boot.py
python allactive.py
python fig41.py
python appendix_numbers.py
python appendix_numbers2.py
```
