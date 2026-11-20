"""第5回 発展: 取得方法ごとに、正解ページ（根拠）をどれだけ取れたかを集計する。

回答生成（LLM）は使わず、取得結果だけを比べる。質問の埋め込みは1問1回だけ呼び、
同じベクトルを全方式で使う（方式ごとの差が埋め込みの揺れに左右されないように）。

    uv run python -m scripts.eval_retrieval                       # 既定の評価セットすべて
    uv run python -m scripts.eval_retrieval --eval lectures/05/retrieval_eval_hard.json
    uv run python -m scripts.eval_retrieval --detail              # 質問ごとの取得ページも表示する

- 質問は "type"（同語・言い換え・推論）ごとに集計する。type がなければ「同語」
- 埋め込みを使わない比較相手として、文字2-gramのTF-IDF検索（キーワード検索）を入れる
- 件数をそろえるのではなく、渡す本文の量（文字数）と再現率の関係で比べる
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import retrieval

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVALS = [ROOT / "lectures/05/retrieval_eval.json", ROOT / "lectures/05/retrieval_eval_holdout.json",
                 ROOT / "lectures/05/retrieval_eval_hard.json"]
TYPES = ["同語", "言い換え", "推論"]


def bigrams(text: str) -> list[str]:
    text = re.sub(r"\s+", "", text.lower())
    return [text[i:i + 2] for i in range(len(text) - 1)]


class KeywordIndex:
    """文字2-gramのTF-IDF（日本語の分かち書きなしで動く、単純なキーワード検索）。"""

    def __init__(self, pages):
        self.pages = pages
        self.tf = [Counter(bigrams(p["title"] + "\n" + p["content"])) for p in pages]
        df = Counter(g for tf in self.tf for g in tf)
        self.idf = {g: math.log(len(pages) / n) for g, n in df.items()}
        self.norm = [math.sqrt(sum((c * self.idf[g]) ** 2 for g, c in tf.items())) or 1.0 for tf in self.tf]

    def search(self, query: str, top_k: int) -> list[dict]:
        q = Counter(bigrams(query))
        scores = []
        for page, tf, norm in zip(self.pages, self.tf, self.norm):
            score = sum(c * self.idf.get(g, 0) ** 2 * tf[g] for g, c in q.items() if g in tf) / norm
            scores.append((-score, page["page_id"], page))
        return [page for _, _, page in sorted(scores)[:top_k]]


def rrf(*rankings: list[str], k: int = 60) -> list[str]:
    """複数の順位を Reciprocal Rank Fusion で1つにまとめる。"""
    score = defaultdict(float)
    for ranking in rankings:
        for rank, page_id in enumerate(ranking):
            score[page_id] += 1 / (k + rank + 1)
    return sorted(score, key=lambda p: (-score[p], p))


def page_ids_of(result) -> list[str]:
    if isinstance(result, list):
        return [hit["page_id"] for hit in result]
    return [hit["page_id"] for hit in result["seeds"] + result["expanded"]]


def build_methods(keyword: KeywordIndex) -> dict:
    def vector(k):
        return lambda q: page_ids_of(retrieval.search(q, top_k=k))

    def kw(k):
        return lambda q: [p["page_id"] for p in keyword.search(q, k)]

    def hybrid(k):
        return lambda q: rrf(page_ids_of(retrieval.search(q, top_k=30)),
                             [p["page_id"] for p in keyword.search(q, 30)])[:k]

    methods = {}
    for k in (3, 5, 10, 15):
        methods[f"ベクトル@{k}"] = vector(k)
    for k in (5, 10, 15):
        methods[f"キーワード@{k}"] = kw(k)
    for k in (5, 10):
        methods[f"ハイブリッド(RRF)@{k}"] = hybrid(k)
    methods["Graph（元）"] = lambda q: page_ids_of(retrieval.search_graph(q))
    methods["terms（関連用語・距離で絞り込み）"] = lambda q: page_ids_of(retrieval.search_graph_terms(q, bridge=False))
    methods["terms（＋回ごとの橋渡し）"] = lambda q: page_ids_of(retrieval.search_graph_terms(q))
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, nargs="+", default=DEFAULT_EVALS)
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    # 質問の埋め込みを使い回す（API呼び出しを1問1回にする）
    retrieval.embed_text = lru_cache(maxsize=None)(retrieval.embed_text)
    pages = {page["page_id"]: page for page in retrieval.load_pages()}
    keyword = KeywordIndex(list(pages.values()))
    items = [item for path in args.eval for item in json.loads(path.read_text(encoding="utf-8"))]
    for item in items:
        item.setdefault("type", "同語")
    types = [t for t in TYPES if any(item["type"] == t for item in items)]

    # 質問と正解ページの言葉の重なり：質問の2-gramのうち、正解ページ（タイトル＋本文）に出てくる割合
    print("## 質問と正解ページの言葉の重なり（2-gramの一致率、高いほど資料と同じ言葉）")
    for t in types:
        overlaps = []
        for item in (i for i in items if i["type"] == t):
            q = set(bigrams(item["question"]))
            gold_text = set(g for gid in item["gold"] for g in bigrams(pages[gid]["title"] + pages[gid]["content"]))
            overlaps.append(len(q & gold_text) / len(q))
        print(f"- {t}（{len(overlaps)}問）: 平均 {sum(overlaps) / len(overlaps):.0%}")

    print("\n## 正解ページの順位（ベクトル／キーワード、214ページ中）")
    for item in items:
        vec = [hit["page_id"] for hit in retrieval.search(item["question"], top_k=None)]
        kw = [p["page_id"] for p in keyword.search(item["question"], len(pages))]
        ranks = ", ".join(f"{g[3:5]}回p.{pages[g]['page']}={vec.index(g) + 1}/{kw.index(g) + 1}位" for g in item["gold"])
        print(f"- [{item['type']}] {item['id']}: {ranks}")

    methods = build_methods(keyword)
    stats = {}  # (方式, type) → [見つけた数, 正解数, 全部そろった質問数, 質問数, 文字数]
    for name, method in methods.items():
        for item in items:
            ids = method(item["question"])
            hit = [g for g in item["gold"] if g in ids]
            for key in ((name, item["type"]), (name, "全体")):
                s = stats.setdefault(key, [0, 0, 0, 0, 0])
                s[0] += len(hit)
                s[1] += len(item["gold"])
                s[2] += len(hit) == len(item["gold"])
                s[3] += 1
                s[4] += sum(len(pages[i]["content"]) for i in ids)
            if args.detail:
                missing = [f"{g[3:5]}回p.{pages[g]['page']}" for g in item["gold"] if g not in ids]
                print(f"  [{name}] {item['id']}: {len(hit)}/{len(item['gold'])} 不足={missing or 'なし'}")

    print("\n## 方式ごとの根拠ページの再現率（type別）と、平均本文文字数")
    header = " | ".join(f"{t}（{stats[(next(iter(methods)), t)][3]}問）" for t in types)
    print(f"| 方式 | {header} | 全体 | 全部そろった質問 | 平均本文文字数 |")
    print("|---|" + "---|" * (len(types) + 3))
    for name in methods:
        cells = []
        for t in types:
            found, total = stats[(name, t)][:2]
            cells.append(f"{found}/{total}（{found / total:.0%}）")
        found, total, full, n, chars = stats[(name, "全体")]
        print(f"| {name} | {' | '.join(cells)} | {found}/{total}（{found / total:.0%}） | {full}/{n} | {chars / n:,.0f} |")


if __name__ == "__main__":
    main()
