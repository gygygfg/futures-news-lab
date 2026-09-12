"""自建词向量(Word2Vec)情感模型。

思路(全部本地自建, 不用预训练模型/外部API):
1. jieba 分词 -> 用全部新闻语料训练 Word2Vec 词向量(自建 embedding)。
2. 用金融多空种子词典在向量空间构造"情感轴": axis = mean(利多词) - mean(利空词)。
3. 每条新闻 = 词向量均值; 情感分 = cos(doc_vec, axis) ∈ [-1,1]。

输出: 回写 news.sentiment, 保存 data/news/word2vec.model 与 data/news/sentiment_axis.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from news_db import NEWS_DIR, connect

MODEL_PATH = NEWS_DIR / "word2vec.model"
AXIS_PATH = NEWS_DIR / "sentiment_axis.json"

# 金融多空种子词(自建词典)
POS = ["上涨", "走高", "走强", "大涨", "涨停", "利好", "利多", "减产", "降息", "降准", "宽松",
       "复苏", "增长", "超预期", "需求旺盛", "库存下降", "供应紧张", "短缺", "提价", "突破",
       "反弹", "强劲", "提振", "回升", "攀升", "创新高", "走俏", "旺盛", "扩张", "改善", "回暖",
       "盈利", "增产不及", "去库存", "旺季"]
NEG = ["下跌", "走低", "走弱", "大跌", "跌停", "利空", "增产", "加息", "收紧", "衰退", "下滑",
       "不及预期", "需求疲软", "库存增加", "供应过剩", "过剩", "降价", "跌破", "违约", "风险",
       "萎缩", "恶化", "低迷", "疲弱", "创新低", "回落", "跳水", "承压", "抛售", "亏损", "通缩",
       "减产不及", "累库", "淡季", "过剩产能"]

STOPWORDS = set(("的 了 和 与 及 或 在 是 为 对 等 将 已 有 也 这 那 一个 我们 你们 他们 以及 表示 称 据 月 日 年 时 分 秒 "
                 "记者 报道 消息 显示 认为 指出 方面 相关 进行 目前 今日 昨日 本周 上周 今年 去年 中国 美国 全球 市场 "
                 "同比 环比 数据 分析 预期 影响 情况 问题 工作 发展 经济 政策 会议 发布 举行 国家 部门 公司 企业 集团").split())


def tokenize(text: str) -> list[str]:
    import jieba
    toks = []
    for t in jieba.lcut(text):
        t = t.strip()
        if not t or t in STOPWORDS:
            continue
        if all(not ch.isalnum() and not ('\u4e00' <= ch <= '\u9fff') for ch in t):
            continue  # 纯标点/符号
        if len(t) == 1 and not ('\u4e00' <= t <= '\u9fff'):
            continue
        toks.append(t)
    return toks


def load_corpus(conn):
    rows = conn.execute("SELECT id, title, content FROM news").fetchall()
    ids = [r[0] for r in rows]
    docs = [tokenize(f"{r[1]} {r[2]}") for r in rows]
    return ids, docs


def train(conn) -> Word2Vec:
    ids, docs = load_corpus(conn)
    print(f"[embed] 语料 {len(docs)} 篇, 总词数 {sum(len(d) for d in docs)}")
    model = Word2Vec(
        sentences=docs, vector_size=100, window=5, min_count=2,
        sg=1, epochs=30, workers=4, seed=42,
    )
    model.save(str(MODEL_PATH))
    print(f"[embed] 词表大小 {len(model.wv)}; 模型 -> {MODEL_PATH}")
    return model


def build_axis(model: Word2Vec):
    wv = model.wv
    pos = [wv[w] for w in POS if w in wv]
    neg = [wv[w] for w in NEG if w in wv]
    print(f"[axis] 命中利多种子 {len(pos)}/{len(POS)}, 利空种子 {len(neg)}/{len(NEG)}")
    if not pos or not neg:
        raise RuntimeError("种子词覆盖不足, 无法构造情感轴")
    axis = np.mean(pos, axis=0) - np.mean(neg, axis=0)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    AXIS_PATH.write_text(json.dumps({
        "pos_hit": [w for w in POS if w in wv],
        "neg_hit": [w for w in NEG if w in wv],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return axis


def score_docs(model: Word2Vec, axis: np.ndarray, conn, only_unscored: bool = False) -> int:
    wv = model.wv
    where = "WHERE sentiment IS NULL" if only_unscored else ""
    rows = conn.execute(f"SELECT id, title, content FROM news {where}").fetchall()
    updates = []
    for nid, title, content in rows:
        toks = [t for t in tokenize(f"{title} {content}") if t in wv]
        if not toks:
            updates.append((None, nid))
            continue
        vec = np.mean([wv[t] for t in toks], axis=0)
        s = float(np.dot(vec, axis) / (np.linalg.norm(vec) + 1e-9))
        updates.append((max(-1.0, min(1.0, s)), nid))
    conn.executemany("UPDATE news SET sentiment=? WHERE id=?", updates)
    conn.commit()
    n_scored = sum(1 for s, _ in updates if s is not None)
    print(f"[score] 已打分 {n_scored}/{len(updates)}")
    return n_scored


def score_only() -> None:
    """加载已保存模型, 仅给未打分的新条目打分(供定时任务)。"""
    import argparse
    from gensim.models import Word2Vec as _W2V
    conn = connect()
    if not MODEL_PATH.exists():
        print("[score-only] 无模型, 转为全量训练")
        main()
        return
    model = _W2V.load(str(MODEL_PATH))
    wv = model.wv
    pos = [wv[w] for w in POS if w in wv]
    neg = [wv[w] for w in NEG if w in wv]
    axis = np.mean(pos, axis=0) - np.mean(neg, axis=0)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    n = score_docs(model, axis, conn, only_unscored=True)
    print(f"[score-only] 新增打分 {n} 条")


def main() -> None:
    conn = connect()
    model = train(conn)
    axis = build_axis(model)
    score_docs(model, axis, conn)

    # 分布与抽样
    vals = [r[0] for r in conn.execute("SELECT sentiment FROM news WHERE sentiment IS NOT NULL")]
    arr = np.array(vals)
    print(f"[dist] n={len(arr)} mean={arr.mean():.3f} std={arr.std():.3f} "
          f"p5={np.percentile(arr,5):.3f} p95={np.percentile(arr,95):.3f}")
    print("--- 最利多 ---")
    for t, s in conn.execute("SELECT substr(title,1,42), sentiment FROM news WHERE sentiment IS NOT NULL ORDER BY sentiment DESC LIMIT 5"):
        print(f"  {s:+.3f}  {t}")
    print("--- 最利空 ---")
    for t, s in conn.execute("SELECT substr(title,1,42), sentiment FROM news WHERE sentiment IS NOT NULL ORDER BY sentiment ASC LIMIT 5"):
        print(f"  {s:+.3f}  {t}")


if __name__ == "__main__":
    import sys
    if "--score-only" in sys.argv:
        score_only()
    else:
        main()
