"""Geminiモデルを3つの質問で比較し、Markdownに結果を出力する。"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from google import genai
from google.genai import errors, types

from app.config import get_api_key


MODELS = {
    "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite",
    "Gemini 3.5 Flash Lite": "gemini-3.5-flash-lite",
    "Gemini 3.6 Flash": "gemini-3.6-flash",
    "Gemini 3.7 Flash": "gemini-3.7-flash",
    "Gemini 3.8 Flash": "gemini-3.8-flash",
}

QUESTIONS = {
    "①": "LoRAとは何ですか？",
    "②": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "③": "この世界は一つの舞台にすぎず、人間は男女を問わず役者にすぎぬって、誰の発言で、どのような心理様態から生まれた？",
}

DEFAULT_OUTPUT = Path("lectures/01/model_benchmark.md")


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def usage_counts(response) -> tuple[int, int, int]:
    usage = response.usage_metadata
    if usage is None:
        return 0, 0, 0
    prompt = usage.prompt_token_count or 0
    candidates = usage.candidates_token_count or 0
    total = usage.total_token_count
    if total is None:
        total = prompt + candidates
    return prompt, candidates, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Geminiモデル比較ベンチマーク")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    api_key = get_api_key()
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30_000),
    )

    rows = []
    results = []

    print("比較実行中: 質問ごとに5モデルを順番に実行します")
    for question_number, question in QUESTIONS.items():
        for display_name, model_id in MODELS.items():
            print(f"{question_number} {display_name} ({model_id})")
            started = time.perf_counter()
            answer = None
            error = None

            try:
                response = client.models.generate_content(model=model_id, contents=question)
                answer = response.text or "（空の応答）"
                input_tokens, output_tokens, total_tokens = usage_counts(response)
            except errors.APIError as exc:
                input_tokens = output_tokens = total_tokens = 0
                error = f"APIエラー（{exc.code}）: {exc.message}"
            except Exception as exc:
                input_tokens = output_tokens = total_tokens = 0
                error = f"実行エラー: {type(exc).__name__}: {exc}"

            elapsed = time.perf_counter() - started
            if error is not None:
                input_tokens = output_tokens = total_tokens = 0

            result = {
                "question_number": question_number,
                "question": question,
                "model": display_name,
                "model_id": model_id,
                "elapsed": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "answer": answer,
                "error": error,
            }
            results.append(result)
            rows.append(result)

    lines = [
        "# Geminiモデル比較ベンチマーク（第1回）",
        "",
        f"実行日時: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 実行対象モデル",
        "",
    ]
    for display_name, model_id in MODELS.items():
        lines.append(f"- {display_name}: `{model_id}`")

    lines.extend(["", "## 比較サマリー", ""])
    lines.append("| 質問 | モデル | 応答時間（秒） | 入力 | 出力 | 合計 | 結果 |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for result in rows:
        result_status = "成功" if result["error"] is None else "エラー"
        lines.append(
            "| {question} | {model} | {elapsed:.3f} | {input_tokens} | {output_tokens} | "
            "{total_tokens} | {status} |".format(
                question=result["question_number"],
                model=result["model"],
                elapsed=result["elapsed"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                total_tokens=result["total_tokens"],
                status=result_status,
            )
        )

    lines.extend(["", "## 詳細結果", ""])
    for result in results:
        lines.extend([
            f"### {result['question_number']} {result['model']} (`{result['model_id']}`)",
            "",
            "**質問:**",
            "",
            result["question"],
            "",
            "**計測:**",
            "",
            f"- 応答時間: {result['elapsed']:.3f} 秒",
            f"- 入力トークン: {result['input_tokens']}",
            f"- 出力トークン: {result['output_tokens']}",
            f"- 合計トークン: {result['total_tokens']}",
            "",
        ])
        if result["error"] is not None:
            lines.extend(["**エラー:**", "", f"```text\n{result['error']}\n```", ""])
        else:
            lines.extend(["**回答:**", "", "```text", result["answer"], "```", ""])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"結果を {args.output} に出力しました")


if __name__ == "__main__":
    main()
