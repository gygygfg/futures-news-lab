# futures-news-lab — 中国商品/金融期货数据与新闻情绪分析

一套适合量化演示/教学的期货数据采集与示例。

## 数据概要(akshare / 新浪源,免费)

| 品种 | 代码 | 交易所 | 交割对象主体 | 日线跨度 | 小时样本 |
|---|---|---|---|---|---|
| 螺纹钢 | RB0 | 上海期货交易所 | 螺纹钢(20mm HRB400) | 2021-01-04~今 | 1023 |
| 沪铜 | CU0 | 上海期货交易所 | 阴极铜 | 2021-01-04~今 | 1023 |
| 沪金 | AU0 | 上海期货交易所 | 黄金(99.99%) | 2021-01-04~今 | 1023 |
| 原油 | SC0 | 上海国际能源交易中心 | 中质含硫原油 | 2021-01-04~今 | 1023 |
| 沪深300股指 | IF0 | 中国金融期货交易所 | 沪深300指数(现金交割) | 2021-01-04~今 | 1023 |
| 10年期国债 | T0 | 中国金融期货交易所 | 10年期国债 | 2021-01-04~今 | 1023 |

- **日线** `data/raw/daily/{symbol}.csv`：`date,open,high,low,close,volume,hold,settle`
- **60分钟** `data/raw/hourly/{symbol}.csv`：`date,open,high,low,close,volume,hold`
- `data/metadata.json`：各品种交易所/交割对象主体/精度/时间跨度/样本量

## 真实性说明(重要)

- 日线使用**主力连续合约**(后缀`0`)，可覆盖多年；交割对象主体=品种本身。
- **免费开源渠道的小时级历史上限约8.5个月**(新浪`futures_zh_minute_sina`固定返回1023根)。因此：
  - 跨期>1年 由**日线**承载；
  - 小时级仅覆盖**近期窗口**，用于细粒度演示/特征工程。
- 若需>1年的连续小时数据，需付费源(如 tushare 积分接口)或交易所付费数据。

## 运行

```bash
uv sync
uv run python data/fetch.py      # 采集并生成 data/raw/** 与 metadata.json
uv run python data/demo.py       # 生成图表 data/demo/*.png + 收益率统计
uv run python data/make_web.py   # 生成交互式网页 web/index.html (+web/data.json)
```

## 交互式网页

生成 `web/index.html`（内嵌 ECharts + JSON，自包含）。打开方式：

```bash
python -m http.server 8000 -d web   # 或直接双击 index.html
# 浏览器访问 http://localhost:8000
```

**公网访问（已部署）**：<https://futures.bxsr.dpdns.org>

页面功能：
- 顶部切换 **品种** 与 **粒度**（日线 / 60分钟）。
- KPI：样本量、年化波动率、最大回撤。
- 图表：K线 + 成交量、单期收益时序。

## 新闻情绪与影响分析

抓取期货相关新闻, 自建词向量模型量化情绪, 分析其对期货价格的影响。

```bash
# 采集
uv run python data/fetch_news.py live                                   # 实时快照
uv run python data/fetch_news.py backfill --start 2024-09-10 --end 2026-09-10 --workers 12
# 自建 embedding 情感(Word2Vec) -> 回写 news.db
uv run python data/news_embed.py            # 训练 + 全量打分
uv run python data/news_embed.py --score-only   # 仅给新条目打分(用已保存模型)
# 影响分析 -> web/news_impact.json
uv run python data/news_impact.py
```

- **新闻源**: 以**可读性高的文章/快讯**为准, 采集时自动过滤视频/直播/回放类(如新闻联播)。历史回填用 `news_economic_baidu`(经济日历)、`macro_info_ws`(华尔街见闻日历) + 文章源(财经早餐/财新网); 实时用财联社/东方财富/同花顺/新浪/上海金属网 + 文章源(东方财富财经早餐/富途牛牛/财新网/华尔街见闻/金十数据)。共 **13 个来源**(10 文章 + 2 结构化 + 1 已移除视频), 实际入库 12 个。
- **可读性过滤** (`news_db.is_readable`): 标题命中"视频/直播/回放/联播/完整版"等视频特征, 或正文过短的条目不入库; 已入库的视频类来源会被 `purge_non_readable` 清理。
- **存储**: `data/news/news.db`(SQLite), 字段含 source/dt/title/content/symbols(关联品种)/sentiment。
- **处理后缓存** (`news_db.sentiment_daily`): 按品种/日期聚合已打分情感; 影响分析与词云只读该缓存做聚合, 避免每次全量扫描。抓取侧对来源/日期做磁盘缓存(`data/news/cache/`), 大范围回填可断点续跑、重复运行秒回。
- **自建情感模型** (`data/news_embed.py`): 全语料训练 Word2Vec 词向量; 用金融多空种子词在向量空间构造"情感轴"; 每条新闻情感分 = cos(文档词向量均值, 情感轴) ∈ [-1,1]。
- **影响分析** (`data/news_impact.py`): **影响时效=2周**。解释变量为"近2周衰减加权情绪", 被解释变量为"未来2周(约10个交易日)累计收益"; 计算同期相关、次日相关、2周相关、宏观情绪相关、OLS(斜率/t/R²)与事件研究。
- **显著相关日**: 定义 `|情绪|` 与 `|未来2周收益|` 同时处于该品种前 20% 的日期为"与收益率显著相关日"。
- **网页交互**: K线与单期收益图上以**橙色**标出显著相关日; 点击任一图的数据点, 会在"点击查看当日新闻"面板列出该日相关新闻(显著日显示全部新闻条目, 含摘要与来源链接)。
- **影响因素词云** (`data/news_wordcloud.py`): 按品种统计相关新闻关键词, 字号=出现频次、颜色=平均情绪(红偏多/蓝偏空)、悬停显示关联平均|未来2周收益|%; **点击某词弹出悬浮窗, 列出该关键词 Top 10 新闻链接**。输出 `web/news_wordcloud.json`。

### 定时任务 (systemd)
- `news-collect.timer` — 每 10 分钟: 抓实时新闻 → 增量打分 → 更新影响分析。
- `news-retrain.timer` — 每日 04:30: 重训 embedding 模型 → 全量打分 → 更新分析。
- 查看: `systemctl list-timers | grep news`; 日志: `journalctl -u news-collect.service -f`

> 说明: 免费实时新闻接口只返回当天快照, 历史由回填(经济日历/文章源)+ 定时累积构成。新闻影响时效设为 **2周**; 实证上新闻情绪对未来2周收益的线性相关仍接近 0(市场已部分定价), 属正常结果。

## 字段说明

- `volume` 成交量(手)；`hold` 持仓量(手)；`settle` 动态结算价(仅日线)。
- 所有价格列已处理为数值；日期升序、无重复、无价格缺失(见 demo 校验输出)。
