"""生成交互式可视化网页。

输出:
- web/index.html   自包含交互式页面(内嵌 JSON + ECharts)
- web/data.json    图表原始数据
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
RAW = PROJ / "data" / "raw"
OUT = PROJ / "web"
OUT.mkdir(exist_ok=True)
DIST = PROJ / "web" / "data.json"

SYMBOLS = ["RB0", "CU0", "AU0", "SC0", "IF0", "T0"]
NAMES = {
    "RB0": "螺纹钢", "CU0": "沪铜", "AU0": "沪金",
    "SC0": "原油", "IF0": "沪深300", "T0": "10年国债",
}
# ---------- 数据 ----------

def load(kind: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / kind / f"{symbol}.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def analyse(kind: str, symbol: str) -> dict:
    d = load(kind, symbol)
    ret = ((d["close"] / d["close"].shift(1) - 1) * 100).values
    base = d["close"]
    mdd = float((base / base.cummax() - 1).min())
    ann_vol = float(ret[ret == ret].std() * np.sqrt(252 if kind == "daily" else 252 * 4))

    return {
        "symbol": symbol,
        "name": NAMES[symbol],
        "kind": kind,
        "n": int(len(d)),
        "span": [d["date"].iloc[0].isoformat(), d["date"].iloc[-1].isoformat()],
        "ann_vol": ann_vol,
        "mdd": mdd,
        # 图表数据
        "kline": [
            [row.date.timestamp() * 1000, row.open, row.close, row.low, row.high, row.volume]
            for row in d.itertuples()
        ],
        "dates": [row.date.isoformat() for row in d.itertuples()],
    }


def main() -> None:
    payload: dict = {"symbols": [], "note": "日线跨期>1年; 小时级为免费源历史窗口"}
    for kind in ("daily", "hourly"):
        for sym in SYMBOLS:
            try:
                payload["symbols"].append(analyse(kind, sym))
            except Exception as exc:  # noqa: BLE001
                print(f"! {kind}/{sym}: {type(exc).__name__}: {exc}")

    DATA = json.dumps(payload, ensure_ascii=False)
    DIST.write_text(DATA, encoding="utf-8")

    html = TEMPLATE.replace("%%DATA%%", DATA)
    (OUT / "index.html").write_text(html, encoding="utf-8")

    # 汇总打印
    print(f"[web] 文件: {OUT / 'index.html'} ({len(html)/1024:.0f}KB), {DIST.name}")
    print(f"{'品种':<6}{'粒度':<7}{'样本':>6}{'年化波动':>10}{'最大回撤':>10}")
    for s in payload["symbols"]:
        print(f"{s['name']:<6}{('日' if s['kind']=='daily' else '时'):<6}{s['n']:>6}"
              f"{s['ann_vol']:>9.2f}%{s['mdd'] * 100:>9.2f}%")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>期货数据 — 可视化</title>
<link rel="icon" href="data:,"/>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts-wordcloud@2.1.0/dist/echarts-wordcloud.min.js"></script>
<style>
  :root{--bg:#0f1420;--card:#171e2e;--line:#2a3550;--txt:#e6ebf5;--sub:#8b98b3;--ac:#12b8ff;--warn:#ff5c5c;}
  *{box-sizing:border-box}
  body{margin:0;font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif;background:var(--bg);color:var(--txt)}
  header{padding:18px 26px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:16px;flex-wrap:wrap}
  header h1{font-size:18px;margin:0;font-weight:600}
  header .sub{color:var(--sub);font-size:12px}
  .controls{margin-left:auto;display:flex;gap:10px;align-items:center}
  select,button{background:#0b1018;color:var(--txt);border:1px solid var(--line);padding:6px 10px;border-radius:6px;font-size:13px}
  button{cursor:pointer;background:var(--ac);border:none;color:#04121c;font-weight:600}
  .wrap{padding:20px 26px}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
  .kpi .v{font-size:22px;font-weight:700;margin-top:4px}
  .kpi .l{color:var(--sub);font-size:12px}
  .kpi .v.ok{color:var(--ac)} .kpi .v.bad{color:var(--warn)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .grid2{display:grid;grid-template-columns:1fr;gap:14px;margin-top:14px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
  .card h3{margin:2px 6px 8px;font-size:14px;color:var(--sub);font-weight:600}
  .chart{width:100%;height:360px}
  .foot{color:var(--sub);font-size:12px;margin-top:14px;line-height:1.7}
  .modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:999;display:flex;align-items:center;justify-content:center}
  .modal-box{background:var(--card);border:1px solid var(--line);border-radius:10px;max-width:760px;width:90%;max-height:80vh;overflow:auto;padding:16px 18px}
  .modal-head{display:flex;justify-content:space-between;align-items:center;font-weight:600;margin-bottom:10px}
  .modal-head button{background:transparent;color:var(--sub);border:none;font-size:16px;cursor:pointer}
  @media(max-width:1000px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div>
    <h1>期货数据可视化</h1>
    <div class="sub">主力连续合约 K线 / 成交量 / 收益 · akshare/新浪源</div>
  </div>
  <div class="controls">
    <label>品种
      <select id="symbol"></select>
    </label>
    <label>粒度
      <select id="kind">
        <option value="daily">日线</option>
        <option value="hourly">60分钟</option>
      </select>
    </label>
    <button id="btn">刷新</button>
  </div>
</header>
<div class="wrap">
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div class="card"><h3>K线价格 + 成交量</h3><div id="kline" class="chart"></div></div>
    <div class="card"><h3>单期收益(%, 时间序列)</h3><div id="ret" class="chart"></div></div>
  </div>
  <div class="card" style="margin-top:14px">
    <h3>新闻情绪 vs 未来2周收益(自建embedding情感)</h3>
    <div id="newsStats" class="kpis" style="margin:0 0 10px"></div>
    <div id="newsChart" class="chart" style="height:320px"></div>
    <div class="sub" id="newsNote" style="color:var(--sub);font-size:12px;margin:6px 6px 0"></div>
  </div>
  <div class="card" style="margin-top:14px">
    <h3>影响因素词云(字号=出现频次 · 红=偏多/蓝=偏空 · 悬停看关联|未来2周收益|)</h3>
    <div id="wordcloud" class="chart" style="height:380px"></div>
    <div class="sub" id="wcNote" style="color:var(--sub);font-size:12px;margin:6px 6px 0"></div>
  </div>
  <div class="card" style="margin-top:14px">
    <h3>点击 K线 / 收益图 · 查看当日相关新闻(橙色标记=与收益率显著相关日)</h3>
    <div id="clickNews"><span style="color:var(--sub);font-size:13px">点击上方任一图表的数据点查看该日新闻</span></div>
  </div>
  <div class="card" style="margin-top:14px">
    <h3>相关新闻 · Top 20(优先原文, 其余按情感强度; 无原文则按标题搜索)</h3>
    <div id="newsList"></div>
  </div>
  <div class="foot" id="foot"></div>
</div>
<div id="wcModal" class="modal-overlay" style="display:none">
  <div class="modal-box">
    <div class="modal-head"><span id="wcModalTitle"></span><button id="wcModalClose" title="关闭">✕</button></div>
    <div id="wcModalBody"></div>
  </div>
</div>
<script>
const DATA = %%DATA%%;
function shortDate(v){
  if(v.endsWith('T00:00:00')) return v.slice(0,10);          // 日线
  if(v.length>10) return v.slice(0,16).replace('T',' ');      // 小时线
  return v;
}
// 从点击事件解析出日期(兼容 K线/成交量/markPoint/散点)
function clickedDay(p, s){
  if(p.componentType==='markPoint' && p.data && p.data.coord) return String(p.data.coord[0]).slice(0,10);
  if(p.seriesType==='scatter' && Array.isArray(p.value)) return String(p.value[0]).slice(0,10);
  const d = s.dates[p.dataIndex];
  return d ? String(d).slice(0,10) : null;
}

function pick(){return DATA.symbols.find(s=>s.symbol==document.getElementById('symbol').value && s.kind==document.getElementById('kind').value)||DATA.symbols[0];}

function render(){
  const s = pick();
  document.getElementById('symbol').value = s.symbol;
  document.getElementById('kind').value = s.kind;
  // KPI
  const kpis = [
    ['样本量', s.n, 'ok'],
    ['年化波动率', (s.ann_vol).toFixed(2) + '%', 'ok'],
    ['最大回撤', (s.mdd * 100).toFixed(2) + '%', 'bad'],
  ];
  document.getElementById('kpis').innerHTML = kpis.map(k=>`<div class="kpi"><div class="l">${k[0]}</div><div class="v ${k[2]}">${k[1]}</div></div>`).join('');
  document.getElementById('foot').innerHTML = DATA.note;

  // 显著相关新闻日(橙色标记)
  const nInfo = NEWS ? NEWS.symbols.find(x=>x.symbol===s.symbol) : null;
  const sigSet = new Set(nInfo ? nInfo.sig_dates : []);
  const sigK = [], sigR = [];
  s.dates.forEach((d,i)=>{ const day=d.slice(0,10); if(sigSet.has(day)){ sigK.push({coord:[d, s.kline[i][2]]}); sigR.push([d, s.retSeries[i]]); } });

  // K线
  const kdata = s.kline.map(r=>[r[1], r[2], r[3], r[4]]);
  const volume = s.kline.map(r=>r[5]);
  const kchart = echarts.init(document.getElementById('kline'));
  kchart.setOption({
    backgroundColor:'transparent',
    tooltip:{trigger:'axis',axisPointer:{type:'cross'},backgroundColor:'#0b1018',borderColor:'#2a3550',textStyle:{color:'#e6ebf5'}},
    axisPointer:{link:[{xAxisIndex:'all'}]},
    grid:[{left:'55px',right:'20px',top:20,height:'62%'},{left:'55px',right:'20px',top:'76%',height:'18%'}],
    xAxis:[{type:'category',data:s.dates,axisLine:{lineStyle:{color:'#2a3550'}},axisLabel:{color:'#8b98b3',formatter:shortDate}},
           {type:'category',gridIndex:1,data:s.dates,axisLabel:{show:false},axisLine:{lineStyle:{color:'#2a3550'}}}],
    yAxis:[{scale:true,splitLine:{lineStyle:{color:'#1c2436'}},axisLabel:{color:'#8b98b3'}},
           {gridIndex:1,splitNumber:2,splitLine:{show:false},axisLabel:{color:'#8b98b3'}}],
    dataZoom:[{type:'inside',xAxisIndex:[0,1],start:60,end:100},{type:'slider',xAxisIndex:[0,1],bottom:2,height:16}],
    series:[
      {name:'K线',type:'candlestick',data:kdata,
        itemStyle:{color:'#ff5c5c',color0:'#12b8ff',borderColor:'#ff5c5c',borderColor0:'#12b8ff'},
        markPoint:{symbol:'pin',symbolSize:34,label:{show:false},itemStyle:{color:'#ffb84d'},
          data:sigK.map(p=>({coord:p.coord, tooltip:{formatter:()=>`显著相关日 ${p.coord[0].slice(0,10)}`}}))}
      },
      {name:'成交量',type:'bar',xAxisIndex:1,yAxisIndex:1,data:volume,itemStyle:{color:'#3a4a6a'}}
    ]
  });
  kchart.off('click');
  kchart.on('click', p=>{ const day=clickedDay(p, s); if(day) showNewsForDate(s.symbol, day); });
  // 收益
  const rchart = echarts.init(document.getElementById('ret'));
  rchart.setOption({
    backgroundColor:'transparent',
    tooltip:{trigger:'axis',backgroundColor:'#0b1018',borderColor:'#2a3550',textStyle:{color:'#e6ebf5'}},
    grid:{left:'55px',right:'20px',top:20,bottom:40},
    xAxis:{type:'category',data:s.dates,axisLine:{lineStyle:{color:'#2a3550'}},axisLabel:{color:'#8b98b3',formatter:shortDate}},
    yAxis:{splitLine:{lineStyle:{color:'#1c2436'}},axisLabel:{color:'#8b98b3'},axisLine:{lineStyle:{color:'#2a3550'}}},
    dataZoom:[{type:'inside',start:60,end:100}],
    series:[
      {name:'收益%',type:'line',data:s.retSeries,showSymbol:false,lineStyle:{width:0.8,color:'#12b8ff'},areaStyle:{opacity:0.25,color:'#12b8ff'}},
      {name:'显著相关日',type:'scatter',symbolSize:11,itemStyle:{color:'#ffb84d'},data:sigR,
        tooltip:{formatter:p=>`显著相关日 ${p.value[0].slice(0,10)}`}}
    ]
  });
  rchart.off('click');
  rchart.on('click', p=>{ const day=clickedDay(p, s); if(day) showNewsForDate(s.symbol, day); });
  renderNews(s);
  renderWordcloud(s);
}
window.addEventListener('resize',()=>{[].forEach.call(document.querySelectorAll('.chart'),c=>echarts.getInstanceByDom(c)&&echarts.getInstanceByDom(c).resize());});
</script>
<script>
// 在DATA加载后用 close 差分构造收益序列
function buildRetSeries(s){
  const clos=s.kline.map(r=>r[2]);
  return clos.map((c,i)=> i===0?null:+( (c/clos[i-1]-1)*100 ).toFixed(3));
}

// 文章链接: 原文优先(经济日历已由后端构造为当日真实页面), 否则回退来源主页
const linkA=(href,txt)=>href?`<a href="${href}" target="_blank" rel="noopener" style="color:#12b8ff;text-decoration:none">${txt}</a>`:txt;
function artHref(x){ return x.url || x.source_url || ''; }
function artLabel(x){ return x.url ? '原文链接 ↗' : '来源主页 ↗'; }
function artTag(x){ const h=artHref(x); return h ? (' · '+linkA(h, artLabel(x))) : ''; }

let NEWS=null;
function renderNews(s){
  const note=document.getElementById('newsNote');
  const stats=document.getElementById('newsStats');
  const list=document.getElementById('newsList');
  if(!NEWS){
    note.textContent='（新闻数据未生成：请运行 data/fetch_news.py 与 data/news_impact.py）';
    stats.innerHTML=''; list.innerHTML=''; return;
  }
  const n=NEWS.symbols.find(x=>x.symbol===s.symbol);
  const f=(x,d=3)=>x==null?'—':x.toFixed(d);
  const kpis=[['相关(同期)',f(n.corr_same)],['相关(次日)',f(n.corr_next)],['相关(未来2周)',f(n.corr_fwd)],['宏观相关(2周)',f(n.macro_corr_fwd)],
              ['OLS斜率',f(n.slope)],['t值',f(n.tstat,2)],['R²',f(n.r2)],['有新闻天数',n.n_days]];
  stats.innerHTML=kpis.map(k=>`<div class="kpi"><div class="l">${k[0]}</div><div class="v">${k[1]}</div></div>`).join('');
  note.textContent='情感分∈[-1,1](自建Word2Vec词向量在"多空轴"上的投影)。影响时效=2周: 情绪为近2周衰减加权, 收益为未来2周累计; 相关/OLS 衡量其解释力, |t|>1.96 显著。';

  const ser=n.series||[];
  const chart=echarts.init(document.getElementById('newsChart'));
  chart.setOption({
    backgroundColor:'transparent',
    tooltip:{trigger:'axis',backgroundColor:'#0b1018',borderColor:'#2a3550',textStyle:{color:'#e6ebf5'}},
    legend:{data:['新闻情绪','未来2周收益%'],textStyle:{color:'#e6ebf5'},top:0},
    grid:{left:'55px',right:'55px',top:34,bottom:40},
    xAxis:{type:'category',data:ser.map(r=>r.date),axisLine:{lineStyle:{color:'#2a3550'}},axisLabel:{color:'#8b98b3',formatter:v=>v.slice(5)}},
    yAxis:[{name:'情绪',splitLine:{lineStyle:{color:'#1c2436'}},axisLabel:{color:'#8b98b3'}},
           {name:'收益%',splitLine:{show:false},axisLabel:{color:'#8b98b3'}}],
    series:[
      {name:'新闻情绪',type:'bar',data:ser.map(r=>r.sent),itemStyle:{color:p=>p.value>=0?'#ff5c5c':'#12b8ff'}},
      {name:'未来2周收益%',type:'line',yAxisIndex:1,data:ser.map(r=>r.ret_fwd),showSymbol:false,lineStyle:{color:'#ffb84d',width:1.2}}
    ]
  });

  const show=(n.top_news && n.top_news.length ? n.top_news : NEWS.news).slice(0,20);
  const link=(href,txt)=>href?`<a href="${href}" target="_blank" rel="noopener" style="color:#12b8ff;text-decoration:none">${txt}</a>`:txt;
  list.innerHTML='<table><thead><tr><th>时间</th><th>来源</th><th>标题</th><th>品种</th><th>情绪</th></tr></thead><tbody>'+
    show.map(x=>{const c=x.sent==null?'#8b98b3':(x.sent>=0?'#ff5c5c':'#12b8ff');
      const src=link(x.url||x.source_url, x.source);
      const title=linkA(artHref(x), x.title);
      return `<tr><td>${x.dt}</td><td>${src}</td><td style="white-space:normal">${title}</td><td>${x.symbols||''}</td><td style="color:${c}">${x.sent==null?'—':x.sent.toFixed(3)}</td></tr>`;}).join('')+
    '</tbody></table>';
}

let WC=null;
function renderWordcloud(s){
  const note=document.getElementById('wcNote');
  if(!WC){ note.textContent='（词云数据未生成：请运行 data/news_wordcloud.py）'; return; }
  const info=WC.symbols[s.symbol];
  if(!info){ note.textContent='无该品种词云数据'; return; }
  const words=info.words||[];
  const color=x=>x.sent>=0.05?'#ff5c5c':(x.sent<=-0.05?'#12b8ff':'#8b98b3');
  note.textContent='基于该品种相关新闻关键词统计, 共 '+words.length+' 个影响因素。';
  const chart=echarts.init(document.getElementById('wordcloud'));
  chart.setOption({
    backgroundColor:'transparent',
    tooltip:{backgroundColor:'#0b1018',borderColor:'#2a3550',textStyle:{color:'#e6ebf5'},
      formatter:p=>`${p.name}<br/>出现 ${p.data.value} 次<br/>平均情绪 ${p.data.sent==null?'—':p.data.sent.toFixed(3)}<br/>关联|未来2周收益| ${p.data.impact==null?'—':p.data.impact.toFixed(2)+'%'}`},
    series:[{
      type:'wordCloud', shape:'circle', width:'100%', height:'100%',
      left:'center', top:'center', sizeRange:[12,58], rotationRange:[-45,45], rotationStep:15,
      gridSize:8, drawOutOfBound:false, layoutAnimation:true,
      textStyle:{color:()=>'#8b98b3'},
      emphasis:{textStyle:{fontWeight:'bold'}},
      data:words.map(x=>({name:x.name, value:x.value, sent:x.sent, impact:x.impact, news:x.news, textStyle:{color:color(x)}}))
    }]
  });
  chart.off('click');
  chart.on('click', p=>{ showWordNews(p.name, p.data.news, s.symbol); });
}

function showWordNews(word, news, symbol){
  const m=document.getElementById('wcModal');
  document.getElementById('wcModalTitle').textContent=`${symbol} · 关键词「${word}」 · Top ${news?news.length:0} 新闻`;
  const link=(href,txt)=>href?`<a href="${href}" target="_blank" rel="noopener" style="color:#12b8ff;text-decoration:none">${txt}</a>`:txt;
  const body=document.getElementById('wcModalBody');
  if(!news || !news.length){ body.innerHTML='<div style="color:var(--sub)">无相关新闻</div>'; }
  else{
    body.innerHTML=news.map((n,i)=>{
      const c=n.sent==null?'#8b98b3':(n.sent>=0?'#ff5c5c':'#12b8ff');
      return `<div style="margin-bottom:10px;padding:8px 10px;border-left:3px solid #ffb84d;background:#0b1018;border-radius:6px">`
        + `<div>${i+1}. ${linkA(artHref(n), n.title)}</div>`
        + `<div style="font-size:12px;color:#8b98b3;margin-top:3px">${n.date} · ${linkA(n.source_url, n.source)}`
        + ` · 情绪 <span style="color:${c}">${n.sent==null?'—':n.sent.toFixed(3)}</span>`
        + `${artTag(n)}</div></div>`;
    }).join('');
  }
  m.style.display='flex';
}

function showNewsForDate(symbol, day){
  const box=document.getElementById('clickNews');
  if(!NEWS){ box.innerHTML='<span style="color:var(--sub)">新闻数据未加载</span>'; return; }
  const n=NEWS.symbols.find(x=>x.symbol===symbol);
  if(!n){ box.innerHTML='<span style="color:var(--sub)">无该品种新闻</span>'; return; }
  const items=(n.sig_news && n.sig_news[day]) || [];
  const cnt=(n.news_count && n.news_count[day]) || 0;
  const link=(href,txt)=>href?`<a href="${href}" target="_blank" rel="noopener" style="color:#12b8ff;text-decoration:none">${txt}</a>`:txt;
  let html=`<div style="margin:4px 6px 10px;color:var(--sub);font-size:13px">${symbol} · ${day}：`
    + (items.length?`<b style="color:#ffb84d">${items.length} 条显著相关新闻</b>`:'<span>无显著相关新闻</span>')
    + (cnt?`（当日相关新闻共 ${cnt} 条）`:'') + '</div>';
  if(items.length){
    html+=items.map(x=>{
      const c=x.sent==null?'#8b98b3':(x.sent>=0?'#ff5c5c':'#12b8ff');
      return `<div style="margin:0 6px 10px;padding:8px 10px;border-left:3px solid #ffb84d;background:#0b1018;border-radius:6px">`
        + `<div style="font-weight:600">${linkA(artHref(x), x.title)}</div>`
        + (x.summary?`<div style="color:#8b98b3;font-size:12px;margin:4px 0;line-height:1.5">${x.summary}</div>`:'')
        + `<div style="font-size:12px;color:#8b98b3">来源: ${linkA(x.source_url, x.source)}`
        + ` · 情绪 <span style="color:${c}">${x.sent==null?'—':x.sent.toFixed(3)}</span>`
        + `${artTag(x)}</div></div>`;
    }).join('');
  }
  box.innerHTML=html;
}

(function init(){
  const sel=document.getElementById('symbol');
  const syms=[...new Set(DATA.symbols.map(s=>s.symbol))];
  sel.innerHTML=syms.map(x=>`<option value="${x}">${DATA.symbols.find(s=>s.symbol===x).name} (${x})</option>`).join('');
  // 修正收益序列用 close 差分
  DATA.symbols.forEach(s=>{s.retSeries=buildRetSeries(s);});
  // 覆盖 ret chart 的收益来源
  render();
  document.getElementById('btn').addEventListener('click',render);
  document.getElementById('symbol').addEventListener('change',render);
  document.getElementById('kind').addEventListener('change',render);
  // 新闻情绪数据(运行时加载), 加载后重绘以叠加"显著相关日"标记
  fetch('news_impact.json').then(r=>r.ok?r.json():null).then(j=>{if(j){NEWS=j;render();}})
    .catch(()=>renderNews(pick()));
  // 影响因素词云数据
  fetch('news_wordcloud.json').then(r=>r.ok?r.json():null).then(j=>{if(j){WC=j;renderWordcloud(pick());}})
    .catch(()=>renderWordcloud(pick()));
  // 弹窗关闭
  document.getElementById('wcModalClose').addEventListener('click',()=>{document.getElementById('wcModal').style.display='none';});
  document.getElementById('wcModal').addEventListener('click',e=>{ if(e.target.id==='wcModal') e.currentTarget.style.display='none'; });
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
