#!/usr/bin/env bash
# 新闻采集流水线: 实时抓取 -> 情感打分 -> 影响分析
# 用法: news_pipeline.sh [collect|retrain]
set -euo pipefail
cd /root/code/data_mining
UV=/root/.local/bin/uv
mode="${1:-collect}"

echo "[$(date '+%F %T')] fetch live ..."
"$UV" run python data/fetch_news.py live

if [ "$mode" = "retrain" ]; then
  echo "[$(date '+%F %T')] retrain embedding ..."
  "$UV" run python data/news_embed.py
else
  echo "[$(date '+%F %T')] score new items ..."
  "$UV" run python data/news_embed.py --score-only
fi

echo "[$(date '+%F %T')] impact analysis ..."
"$UV" run python data/news_impact.py
echo "[$(date '+%F %T')] wordcloud ..."
"$UV" run python data/news_wordcloud.py
echo "[$(date '+%F %T')] done ($mode)"
