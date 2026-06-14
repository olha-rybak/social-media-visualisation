#!/usr/bin/env python3
"""
build_explore_data.py
Pulls lingbow/tiktok-video-engagement-200k from HuggingFace (auto-cached) and
emits a compact explore_data.json shaped for the word-river / archetype viz.

Run:  pip install datasets pandas pyarrow numpy scikit-learn   (already present)
      python3 build_explore_data.py
"""
import json, re, collections
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

DS = "lingbow/tiktok-video-engagement-200k"
N_TOPICS   = 6        # top topics by video volume
WORDS_WK   = 10       # top words per (topic, week)
TAGS_DAY   = 5        # top hashtags stored per (topic, day)
LIFELINES  = 22       # sampled trajectories per topic
OUT        = "explore_data.json"
MONTH_MIN  = 5000     # only offer months with at least this many videos
EMO = ["joy","surprise","anger","sadness","disgust","fear"]
PALETTE = ["#e84a8a","#d99100","#0c9e6e","#e2731a","#e0354f","#7a5cc4","#2f7fd0","#6a9a00"]
BAD_TOPICS = {"Others","Other","other","Unknown","None"}
# generic platform tags/words to suppress (so we surface TOPICAL language)
TIKTOK_STOP = {"fyp","fypシ","foryou","foryoupage","fory","foru","viral","viralvideo","trending",
    "trend","tiktok","capcut","explore","fypage","edit","xyzbca","video","like","follow","page",
    "comment","share","new","day","time","just","love","real","good","funny","goodvibes","fypp"}
STOP_TAGS = {t.lower() for t in TIKTOK_STOP}
STOPW = list(ENGLISH_STOP_WORDS | TIKTOK_STOP)

def log(*a): print(*a, flush=True)

# ---------------------------------------------------------------- load videos
log("Loading videos subset …")
vids = load_dataset(DS, "videos", split="train").to_pandas()
log(f"  videos: {len(vids):,} rows")
vids = vids[["video_id","topic","create_date","desc","gpt_summary","hashtags","created_by_ai",
             "duration","question_count","emoji_count"]+EMO].copy()
vids = vids.dropna(subset=["topic","create_date"])
vids["create_date"] = pd.to_datetime(vids["create_date"], errors="coerce")
vids = vids.dropna(subset=["create_date"])

# ---------------------------------------------------------------- engagement
log("Loading engagement_daily subset (this is the big one) …")
eng = load_dataset(DS, "engagement_daily", split="train").to_pandas()
log(f"  engagement_daily: {len(eng):,} rows")
eng = eng[["video_id","days_since_post","play_count","like_count"]].copy()
# play_count/like_count are CUMULATIVE → final per video = max (no sort needed)
final = eng.groupby("video_id", as_index=False).agg(views=("play_count","max"),
                                                    likes=("like_count","max"))
vids = vids.merge(final, on="video_id", how="left")
vids["views"] = vids["views"].fillna(0); vids["likes"] = vids["likes"].fillna(0)

# ---------------------------------------------------------------- topics
counts = vids[~vids.topic.isin(BAD_TOPICS)]["topic"].value_counts()
top_topics = counts.head(N_TOPICS).index.tolist()
log("Top topics:", top_topics)
vids = vids[vids["topic"].isin(top_topics)].copy()

# ---------------------------------------------------------------- continuous day axis (full range)
d0 = vids["create_date"].min().normalize(); d1 = vids["create_date"].max().normalize()
all_days = pd.date_range(d0, d1, freq="D")
nDays = len(all_days)
day_index = {d.normalize(): i for i, d in enumerate(all_days)}
vids["day"]  = vids["create_date"].dt.normalize().map(day_index)
vids["week"] = (vids["day"] // 7).astype(int)
NWK = int(vids["week"].max()) + 1
total_by_day = vids.groupby("day").size().reindex(range(nDays), fill_value=0)
gap_days = [int(d) for d in range(nDays) if total_by_day[d] == 0]
log(f"{nDays} days ({d0.date()}→{d1.date()}), {NWK} weeks, gap days: {gap_days}")

# months metadata for the UI month filter
adf = pd.DataFrame({"date": all_days}); adf["i"] = range(nDays); adf["m"] = adf["date"].dt.to_period("M")
mcount = vids.groupby(vids["create_date"].dt.to_period("M")).size()
months = []
for per, grp in adf.groupby("m"):
    c = int(mcount.get(per, 0))
    if c < MONTH_MIN: continue
    months.append({"key": str(per), "label": per.to_timestamp().strftime("%B %Y"),
                   "start": int(grp["i"].min()), "end": int(grp["i"].max()), "count": c})
default_month = max(months, key=lambda m: m["count"])["key"]
log("Months:", [(m["label"], m["count"]) for m in months], "| default", default_month)

def clean_text(s):
    s = re.sub(r"http\S+", " ", str(s).lower())
    return s.replace("#", " ").replace("@", " ")
def clean_cap(s):
    return re.sub(r"\s+", " ", str(s)).strip()

# ---------------------------------------------------------------- tf-idf words per (topic, week)
docs, doc_keys = [], []
for tp in top_topics:
    sub = vids[vids.topic == tp]
    for wk in range(NWK):
        docs.append(" ".join(clean_text(x) for x in sub[sub.week == wk]["desc"].fillna("")))
        doc_keys.append((tp, wk))
vec = TfidfVectorizer(stop_words=STOPW, token_pattern=r"[a-zA-Z][a-zA-Z]{2,}", max_features=4000)
X = vec.fit_transform(docs); vocab = np.array(vec.get_feature_names_out())
top_words_by_key = {}
for r,(tp,wk) in enumerate(doc_keys):
    row = X[r].toarray().ravel(); idx = row.argsort()[::-1][:WORDS_WK]
    top_words_by_key[(tp,wk)] = [vocab[i] for i in idx if row[i] > 0]

# ---------------------------------------------------------------- per-topic build
topics_out, series, daily, words, lifelines = [], {}, {}, {}, {}
def smooth(s): return s.rolling(3, center=True, min_periods=1).mean()

for ti, tp in enumerate(top_topics):
    sub = vids[vids.topic == tp]
    color = PALETTE[ti % len(PALETTE)]
    g = sub.groupby("day")
    cnt   = g.size().reindex(range(nDays), fill_value=0).astype(float)
    views = g["views"].sum().reindex(range(nDays), fill_value=0).astype(float)
    likes = g["likes"].sum().reindex(range(nDays), fill_value=0).astype(float)
    for ser in (cnt, views, likes):            # bridge global gap days (e.g. missing Oct 5)
        ser.iloc[gap_days] = np.nan; ser.interpolate(limit_direction="both", inplace=True)
    series[tp] = {"count": smooth(cnt).round(2).tolist(),
                  "views": smooth(views).round(1).tolist(),
                  "likes": smooth(likes).round(1).tolist()}
    # per-day archetype
    dd = [None]*nDays
    for day, sd in sub.groupby("day"):
        order = sd["views"].to_numpy().argsort()
        topv = sd.iloc[int(order[-1])]; medv = sd.iloc[int(order[len(order)//2])]
        tags = collections.Counter()
        for hl in sd["hashtags"]:
            if isinstance(hl,(list,np.ndarray)):
                for h in hl:
                    n = h.get("hashtag_name") if isinstance(h,dict) else None
                    if n and n.lower() not in STOP_TAGS: tags["#"+n] += 1
        def summ_of(rec):   # full GPT summary (fallback to desc)
            s = rec["gpt_summary"]
            return clean_cap(s) if isinstance(s,str) and str(s).strip() else clean_cap(rec["desc"])
        def desc_of(rec):   # real video description (fallback to summary)
            s = rec["desc"]
            return clean_cap(s) if isinstance(s,str) and str(s).strip() else clean_cap(rec["gpt_summary"])
        dd[int(day)] = {
            "emo": [float(sd[e].mean()) for e in EMO],
            "ai": float(sd["created_by_ai"].mean()),
            "dur": float(sd["duration"].mean()),
            "avgViews": float(sd["views"].mean()),
            "tags": [t for t,_ in tags.most_common(TAGS_DAY)],
            "cap": desc_of(medv), "topCap": summ_of(topv), "topViews": float(topv["views"]),
        }
    # fill empty/gap days from nearest neighbour so the card never breaks
    last = next((x for x in dd if x), None)
    for day in range(nDays):
        if dd[day] is None: dd[day] = dict(last) if last else {}
        else: last = dd[day]
    daily[tp] = dd
    words[tp] = [top_words_by_key.get((tp,wk), []) for wk in range(NWK)]
    # lifelines
    samp = sub.sort_values("views", ascending=False).head(LIFELINES*3).sample(
        min(LIFELINES, len(sub)), random_state=ti) if len(sub) else sub
    traj_rows = eng[eng.video_id.isin(samp.video_id)]
    lines = []
    for vid, grp in traj_rows.groupby("video_id"):
        pts = [{"d": int(x), "v": float(max(1, y))} for x,y in zip(grp.days_since_post, grp.play_count) if x <= 29]
        pts.sort(key=lambda p: p["d"])
        if len(pts) >= 3: lines.append({"fv": pts[-1]["v"], "pts": pts})
    lifelines[tp] = lines
    topics_out.append({"key": tp, "color": color, "emoBase": [float(sub[e].mean()) for e in EMO]})
    log(f"  built {tp}: {len(sub):,} videos, {len(lines)} lifelines")

out = {"dateStart": str(d0.date()), "nDays": nDays, "emoOrder": EMO,
       "months": months, "defaultMonth": default_month,
       "topics": topics_out, "series": series, "daily": daily,
       "words": words, "lifelines": lifelines}
with open(OUT, "w") as f: json.dump(out, f)
import os
log(f"\nWrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)")
