"""Collect requests from people outside the project into a second held-out set for the agent benchmark.

People get situation cards (collect/cards.jsonl), never the tool list, and write what they would type
to an audio assistant. Two annotators label the expected plan independently; only rows they agree on,
or that were settled by hand, become tasks. Protocol: collect/README.md.

    uv run python benchmarks/agent/collect.py questionnaire --participant P01 --lang ru > p01.md
    uv run python benchmarks/agent/collect.py import responses.csv        # -> collect/drafts/
    uv run python benchmarks/agent/collect.py check collect/drafts/label_B.csv    # an annotator checks their own sheet
    uv run python benchmarks/agent/collect.py merge collect/drafts/label_A.csv collect/drafts/label_B.csv \\
        [--resolved resolved.csv]                                          # -> tasks/heldout2.jsonl

Calibration before the real labelling: collect/calibration/ has 20 requests and blank sheets; merge them
with --drafts collect/calibration/calibration.jsonl --out collect/calibration/merged.jsonl.

responses.csv columns: participant, card, request. Raw responses and drafts stay out of git
(collect/drafts is ignored); only the merged, anonymised task file is committed.

Label sheet syntax, one row per request:
  accept     separate>denoise | separate>restore   ("-" no tools, "*" any plan up to max_steps)
  inputs     2=1.vocals; 3=2.denoised|2.restored
  params     1:end_s=197.4,silence=true
  needs      key,chords          unsupported  yes      reply_any  124|A minor
  forbid     separate,analyze    max_steps    2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ah.agent import PlanError, reply_language, validate
from ah.registry import TOOLS

HERE = Path(__file__).parent
COLLECT = HERE / "collect"
DRAFTS = COLLECT / "drafts"
CARDS_PER_PERSON = 12
# Words that name a tool or its action. A card must describe a situation and a goal, not the move to make.
TOOL_WORDS = re.compile(r"(отдел|раздел|денойз|шумодав|реставр|обреж|обрез|анализ|separ|denois|restor|trim|analy|stem|стем|"
                        r"вокал|vocal|acapella|акапелл|bpm|минус|inspect)", re.IGNORECASE)
LABEL_FIELDS = ["id", "card", "fixture", "situation", "request", "accept", "inputs", "params", "needs", "unsupported",
                "reply_any", "forbid", "max_steps", "notes"]
GOLD_FIELDS = LABEL_FIELDS[5:-1]

INTRO = {
    "ru": """# Как бы ты попросил?

Мы делаем ассистента для работы со звуком: загружаешь запись и пишешь словами, что нужно сделать.
Ниже несколько ситуаций. Для каждой напиши **одну-две фразы так, как написал бы их ассистенту в чате**.
Правильных и неправильных ответов нет: нам важны твои собственные слова, с опечатками, сленгом и сокращениями.
Если в такой ситуации ты бы ничего не просил или не знаешь, что попросить, так и напиши.

- Пиши на том языке, на котором тебе удобно.
- Не указывай имён, телефонов и других личных данных.
- Ответы анонимны: мы сохраняем только текст запроса и номер ситуации.

Отправляя ответы, ты соглашаешься, что текст запросов будет использован для оценки ассистента
и может быть опубликован в открытом наборе данных без указания авторства.
""",
    "en": """# How would you ask?

We are building an assistant for audio: you upload a recording and say in words what you need done.
Below are a few situations. For each, write **a sentence or two the way you would type it to the assistant
in a chat**. There are no right or wrong answers: we want your own words, typos, slang and shortcuts included.
If you would not ask for anything, or don't know what to ask, say so.

- Write in whatever language you are comfortable with.
- Do not include names, phone numbers or other personal details.
- Answers are anonymous: we keep only the request text and the situation number.

By sending your answers you agree that the request texts are used to evaluate the assistant and may be
published in an open dataset without attribution.
""",
}


def load_cards() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (COLLECT / "cards.jsonl").read_text().splitlines() if line.strip()]


def leaks(card: dict[str, Any]) -> list[str]:
    return sorted({m.lower() for lang in ("ru", "en") for m in TOOL_WORDS.findall(card[lang])})


def questionnaire(participant: str, lang: str) -> str:
    """A participant's own random subset of cards in a random order, so card order does not shape the answers."""
    cards = load_cards()
    rng = random.Random(int(hashlib.sha256(participant.encode()).hexdigest(), 16))
    picked = rng.sample(cards, min(CARDS_PER_PERSON, len(cards)))
    lines = [INTRO[lang], f"Участник / Participant: {participant}\n"]
    for card in picked:
        lines.append(f"## {card['card']}\n\n{card[lang]}\n\n> \n")
    return "\n".join(lines)


def language(text: str) -> str:
    named = reply_language(text)
    return {"Russian": "ru", "Ukrainian": "uk", "English": "en"}.get(named, "other")


def import_responses(path: Path) -> list[dict[str, Any]]:
    cards = {c["card"]: c for c in load_cards()}
    pseudonyms: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    tasks = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            request = " ".join((row.get("request") or "").split())
            card = (row.get("card") or "").strip()
            if not request or card not in cards or len(request) > 2000:
                continue
            key = (card, request.lower())
            if key in seen:
                continue
            seen.add(key)
            # Whatever the sheet used to tell people apart becomes P01, P02...; the mapping is not kept.
            person = pseudonyms.setdefault(row.get("participant", ""), f"P{len(pseudonyms) + 1:02d}")
            tasks.append({"id": f"h2-{len(tasks) + 1:03d}", "participant": person, "card": card,
                          "fixture": cards[card]["fixture"], "situation": cards[card]["ru"], "request": request,
                          "lang": language(request)})
    return tasks


# --- gold sheets ------------------------------------------------------------------

def parse_gold(row: dict[str, str]) -> dict[str, Any]:
    accept = (row.get("accept") or "").strip()
    if not accept:
        raise ValueError(f"{row.get('id')}: accept is empty")
    gold: dict[str, Any] = {}
    if accept == "*":
        gold["accept"] = None
        gold["max_steps"] = int(row.get("max_steps") or 2)
    else:
        gold["accept"] = sorted([] if alt.strip() == "-" else [t.strip() for t in alt.split(">")] for alt in accept.split("|"))
    for part in filter(None, (p.strip() for p in (row.get("inputs") or "").split(";"))):
        n, _, refs = part.partition("=")
        options = [r.strip() for r in refs.split("|")]
        gold.setdefault("inputs", {})[n.strip()] = options[0] if len(options) == 1 else options
    for part in filter(None, (p.strip() for p in (row.get("params") or "").split(";"))):
        n, _, pairs = part.partition(":")
        values = {}
        for pair in filter(None, (x.strip() for x in pairs.split(","))):
            k, _, v = pair.partition("=")
            values[k.strip()] = json.loads(v.strip().lower()) if v.strip().lower() in ("true", "false") else float(v)
        gold.setdefault("params", {})[n.strip()] = values
    if (row.get("needs") or "").strip():
        gold["needs"] = sorted(x.strip() for x in row["needs"].split(","))
    if (row.get("unsupported") or "").strip().lower() in ("yes", "да", "1", "true"):
        gold["must_say_unsupported"] = True
    if (row.get("reply_any") or "").strip():
        gold["reply_any"] = [x.strip() for x in row["reply_any"].split("|")]
    if (row.get("forbid") or "").strip():
        gold["forbid"] = sorted(x.strip() for x in row["forbid"].split(","))
    return gold


def check_gold(gold: dict[str, Any]) -> None:
    """Every accepted plan must be one the agent's own validate would take, with the gold's inputs and params."""
    for tools in gold["accept"] or []:
        steps = []
        for n, tool in enumerate(tools, 1):
            source = gold.get("inputs", {}).get(str(n), "track")
            steps.append({"tool": tool, "input": source if isinstance(source, str) else source[0], "why": "x",
                          "params": gold.get("params", {}).get(str(n), {})})
        validate({"reply": "ok", "steps": steps})


def check_sheet(path: Path) -> dict[str, Any]:
    """What an annotator should fix before handing the sheet in: unlabelled rows, errors, and times past the end."""
    report: dict[str, Any] = {"rows": 0, "labelled": 0, "unlabelled": [], "errors": [], "warnings": []}
    for task_id, row in read_sheet(path).items():
        report["rows"] += 1
        if not (row.get("accept") or "").strip():
            report["unlabelled"].append(task_id)
            continue
        report["labelled"] += 1
        fixture = HERE / "fixtures" / f"{row.get('fixture', '')}.json"
        try:
            gold = parse_gold(row)
            check_gold(gold)
        except (ValueError, PlanError) as err:
            report["errors"].append(f"{task_id}: {err}")
            continue
        unknown = set(gold.get("forbid", [])) - set(TOOLS)
        if unknown:
            report["errors"].append(f"{task_id}: forbid has unknown tools {sorted(unknown)}")
        if set(gold.get("needs", [])) - {"key", "chords"}:
            report["errors"].append(f"{task_id}: needs takes only key and chords")
        if not fixture.is_file():
            report["errors"].append(f"{task_id}: no fixture {fixture.name}")
            continue
        duration = json.loads(fixture.read_text())["selected_version"]["duration_s"]
        for n, values in gold.get("params", {}).items():
            for key in ("start_s", "end_s"):
                if isinstance(values.get(key), float) and values[key] > duration + 0.1:
                    report["warnings"].append(f"{task_id}: step {n} {key}={values[key]} is past the end of the open version ({duration} s)")
    return report


def kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa: agreement on the plan class beyond what two random labellers with these habits would reach."""
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return round((observed - expected) / (1 - expected), 4) if expected < 1 else 1.0


def read_sheet(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def merge(a_path: Path, b_path: Path, resolved_path: Path | None, drafts_path: Path | None = None) -> dict[str, Any]:
    drafts = {t["id"]: t for t in map(json.loads, (drafts_path or DRAFTS / "tasks.draft.jsonl").read_text().splitlines())}
    a, b = read_sheet(a_path), read_sheet(b_path)
    resolved = read_sheet(resolved_path) if resolved_path else {}
    labelled = sorted(set(a) & set(b) & set(drafts))
    tasks, disagreements, classes_a, classes_b, invalid, agreed = [], [], [], [], [], 0
    for task_id in labelled:
        try:
            gold_a, gold_b = parse_gold(a[task_id]), parse_gold(b[task_id])
        except ValueError as err:
            invalid.append(f"{task_id}: {err}")
            continue
        classes_a.append(json.dumps(gold_a["accept"]))
        classes_b.append(json.dumps(gold_b["accept"]))
        same = json.dumps(gold_a, sort_keys=True) == json.dumps(gold_b, sort_keys=True)
        agreed += same
        if task_id in resolved:
            gold = parse_gold(resolved[task_id])
        elif same:
            gold = gold_a
        else:
            disagreements.append({"id": task_id, "request": drafts[task_id]["request"],
                                  **{f"A_{k}": a[task_id].get(k, "") for k in GOLD_FIELDS},
                                  **{f"B_{k}": b[task_id].get(k, "") for k in GOLD_FIELDS}})
            continue
        try:
            check_gold(gold)
        except PlanError as err:
            invalid.append(f"{task_id}: {err}")
            continue
        d = drafts[task_id]
        tasks.append({"id": task_id, "category": d["card"], "lang": d["lang"], "request": d["request"],
                      "track_data": f"fixtures/{d['fixture']}.json", "participant": d["participant"], "gold": gold})
    parsed = len(classes_a)
    return {"tasks": tasks, "disagreements": disagreements, "invalid": invalid,
            "stats": {"labelled_by_both": parsed, "full_agreement": round(agreed / parsed, 4) if parsed else None,
                      "plan_class_kappa": kappa(classes_a, classes_b), "resolved_by_hand": len(set(resolved) & set(labelled)),
                      "tasks": len(tasks), "open_disagreements": len(disagreements), "invalid_gold": len(invalid)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("questionnaire")
    q.add_argument("--participant", required=True)
    q.add_argument("--lang", choices=["ru", "en"], default="ru")
    i = sub.add_parser("import")
    i.add_argument("responses", type=Path)
    m = sub.add_parser("merge")
    m.add_argument("label_a", type=Path)
    m.add_argument("label_b", type=Path)
    m.add_argument("--resolved", type=Path)
    m.add_argument("--drafts", type=Path, help="draft tasks (default collect/drafts/tasks.draft.jsonl)")
    m.add_argument("--out", type=Path, help="merged task file (default tasks/heldout2.jsonl)")
    c = sub.add_parser("check")
    c.add_argument("sheet", type=Path)
    args = parser.parse_args()

    if args.cmd == "questionnaire":
        bad = {c["card"]: leaks(c) for c in load_cards() if leaks(c)}
        if bad:
            sys.exit(f"cards name tools, fix them first: {bad}")
        print(questionnaire(args.participant, args.lang))
    elif args.cmd == "import":
        tasks = import_responses(args.responses)
        DRAFTS.mkdir(parents=True, exist_ok=True)
        (DRAFTS / "tasks.draft.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))
        for name in ("label_A.csv", "label_B.csv"):
            with (DRAFTS / name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
                writer.writeheader()
                writer.writerows({k: t.get(k, "") for k in LABEL_FIELDS} for t in tasks)
        print(json.dumps({"requests": len(tasks), "participants": len({t["participant"] for t in tasks}),
                          "by_lang": Counter(t["lang"] for t in tasks), "by_card": Counter(t["card"] for t in tasks)},
                         ensure_ascii=False, indent=1))
    elif args.cmd == "check":
        report = check_sheet(args.sheet)
        print(f"\n{args.sheet.name}: {report['labelled']} of {report['rows']} rows labelled")
        for kind in ("errors", "warnings"):
            for line in report[kind]:
                print(f"{kind[:-1]}: {line}")
        if report["unlabelled"]:
            print(f"unlabelled: {', '.join(report['unlabelled'])}")
        print("ok" if not report["errors"] else f"{len(report['errors'])} errors")
        sys.exit(1 if report["errors"] else 0)
    else:
        drafts = args.drafts or DRAFTS / "tasks.draft.jsonl"
        out = merge(args.label_a, args.label_b, args.resolved, drafts)
        disagreements = drafts.parent / "disagreements.csv"
        if out["disagreements"]:
            with disagreements.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(out["disagreements"][0]))
                writer.writeheader()
                writer.writerows(out["disagreements"])
        target = args.out or HERE / "tasks" / "heldout2.jsonl"
        target.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in out["tasks"]))
        print(json.dumps(out["stats"] | {"invalid": out["invalid"][:10], "written": str(target)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
