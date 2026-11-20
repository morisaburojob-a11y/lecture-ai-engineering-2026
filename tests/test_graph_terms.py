"""第5回 発展: search_graph_terms のテスト（APIキーを使わない）。"""

from app import retrieval
from app.config import GRAPH_MAX_RESULTS


def fake_ranking(order):
    """指定した順に距離を付け、残りのページは遠い距離で返す偽の search。"""
    pages = retrieval.load_pages()
    rank = {page_id: i for i, page_id in enumerate(order)}
    def search(query, top_k=None):
        hits = [{**p, "distance": 0.1 + 0.01 * rank.get(p["page_id"], 50)} for p in pages]
        hits.sort(key=lambda h: (h["distance"], h["page_id"]))
        return hits if top_k is None else hits[:top_k]
    return search


def test_expands_by_terms_not_next_page(monkeypatch):
    order = ["lec02-main-p052", "lec02-main-p051", "lec02-main-p060", "lec02-main-p047", "lec02-main-p053"]
    monkeypatch.setattr(retrieval, "search", fake_ranking(order))
    result = retrieval.search_graph_terms("LoRAとQLoRA", bridge=False)
    seed_ids = [s["page_id"] for s in result["seeds"]]
    assert seed_ids == order[:3]
    expanded = [e["page_id"] for e in result["expanded"]]
    # p.47 は p.52 の前後ページではないが、term:lora 経由で届く
    assert "lec02-main-p047" in expanded
    for e in result["expanded"]:
        assert e["path"][0]["id"] in seed_ids and e["path"][1]["id"].startswith("term:")
    assert len(set(seed_ids + expanded)) == len(seed_ids + expanded)


def test_margin_and_limit(monkeypatch):
    monkeypatch.setattr(retrieval, "search", fake_ranking(["lec03-main-p039", "lec03-main-p040", "lec03-main-p038"]))
    result = retrieval.search_graph_terms("メモリ", margin=0.0, bridge=False)
    assert result["expanded"] == []  # seedより遠い候補は margin=0 ですべて落ちる
    result = retrieval.search_graph_terms("メモリ")
    assert len(result["seeds"]) + len(result["expanded"]) <= GRAPH_MAX_RESULTS


def test_bridge_adds_page_from_other_lecture(monkeypatch):
    monkeypatch.setattr(retrieval, "search", fake_ranking(["lec02-main-p047", "lec02-main-p046", "lec02-main-p057"]))
    result = retrieval.search_graph_terms("RAGの仕組み", margin=0.0)
    bridged = [e for e in result["expanded"] if e["path"][0]["id"] == "term:rag"]
    assert any(e["lecture"] == 1 for e in bridged)
