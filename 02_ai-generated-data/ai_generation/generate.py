"""Generate the AI-assisted and AI-generated text for each human sample.

For every human sample two files are written, both named <seed_sample_id>.txt:
  data/raw/ai_assisted/   the human text rewritten and polished by the model
  data/raw/ai_generated/  a new text on the same topic, written by the model

Existing files are skipped, so a run can be stopped and resumed at any time.
Every new file is also recorded in data/raw/generation_log.jsonl.
"""

import datetime as dt
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config as cfg
from . import prompts as P
from .clients import get_client

_log_lock = threading.Lock()


def count_words(text):
    return len(text.split())


def load_human_ids():
    meta = json.loads(cfg.HUMAN_META.read_text(encoding="utf-8"))
    return {int(sample["sample_id"]) for sample in meta["samples"]}


def call_with_retries(complete, instructions, text, min_words=0):
    """Call the model; retry on errors, empty answers and too-short answers."""
    for attempt in range(1, cfg.MAX_RETRIES + 1):
        try:
            result = complete(instructions, text)
            if not result:
                raise RuntimeError("empty answer")
            if count_words(result) < min_words:
                raise RuntimeError(f"answer too short ({count_words(result)} words)")
            return result
        except Exception as error:
            print(f"  attempt {attempt}/{cfg.MAX_RETRIES} failed: {error}")
            if attempt == cfg.MAX_RETRIES:
                raise
            time.sleep(2**attempt)


def length_range(target_words, strategy):
    """Allowed length (±20%) that is written into the prompt."""
    if strategy == "topics":
        min_words = max(1, int(target_words * 0.8))
        max_words = max(min_words, int(target_words * 1.2))
    else:
        min_words = int(target_words * 0.8)
        max_words = int(target_words * 1.2)
    return min_words, max_words


def make_assisted(complete, text):
    min_words = int(count_words(text) * cfg.MIN_LENGTH_RATIO)
    return call_with_retries(complete, P.ASSISTED_INSTRUCTIONS, text, min_words), None


def make_generated(complete, text, strategy):
    """Returns (new_text, topics). topics is only set for the "topics" strategy."""
    target = count_words(text)
    min_words, max_words = length_range(target, strategy)
    min_accept = int(target * cfg.MIN_LENGTH_RATIO)

    if strategy == "source_based":
        prompt = P.SOURCE_BASED_TEMPLATE.format(
            target_words=target, min_words=min_words, max_words=max_words, text=text
        )
        return call_with_retries(complete, P.SOURCE_BASED_INSTRUCTIONS, prompt, min_accept), None

    if strategy == "topics":
        topics = call_with_retries(complete, P.TOPICS_EXTRACT_INSTRUCTIONS, text)
        prompt = P.TOPICS_TEMPLATE.format(
            target_words=target, min_words=min_words, max_words=max_words, notes=topics
        )
        return call_with_retries(complete, P.TOPICS_INSTRUCTIONS, prompt, min_accept), topics

    raise ValueError(f"Unknown strategy: {strategy}")


def log(record):
    with _log_lock:
        cfg.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with cfg.LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_sample(sample_id, model_name, complete, model_id):
    spec = cfg.MODELS[model_name]
    source_file = cfg.HUMAN_DIR / f"{sample_id}.txt"

    if not source_file.exists():
        return f"{sample_id}: missing human text"

    text = source_file.read_text(encoding="utf-8").strip()
    if not text:
        return f"{sample_id}: empty human text"

    jobs = [
        ("ai_assisted", cfg.ASSISTED_DIR, lambda: make_assisted(complete, text)),
        (
            "ai_generated",
            cfg.GENERATED_DIR,
            lambda: make_generated(complete, text, spec["strategy"]),
        ),
    ]
    messages = [f"human={count_words(text)}"]

    for kind, folder, make in jobs:
        path = folder / f"{sample_id}.txt"
        if path.exists():
            messages.append(f"{kind} skipped")
            continue

        result, topics = make()
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(result, encoding="utf-8")

        record = {
            "seed_sample_id": sample_id,
            "kind": kind,
            "file": f"{folder.name}/{sample_id}.txt",
            "model": model_name,
            "model_id": model_id,
            "company": spec["company"],
            "strategy": spec["strategy"] if kind == "ai_generated" else "rewrite",
            "seed_words": count_words(text),
            "words": count_words(result),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        if topics:
            record["topics"] = topics
        log(record)
        messages.append(f"{kind}={count_words(result)}")

    return f"{sample_id}: " + " | ".join(messages)


def run(model_name, start=None, end=None, workers=None):
    """Generate both texts for every human sample in the model's ID range."""
    spec = cfg.MODELS[model_name]
    first, last = spec["seed_ids"]
    first, last = max(first, start or first), min(last, end or last)

    human_ids = load_human_ids()
    ids = [i for i in range(first, last + 1) if i in human_ids]
    workers = workers or spec["workers"]
    complete, model_id = get_client(model_name)

    print(
        f"{model_name} ({model_id}) | IDs {first}-{last} | {len(ids)} samples | {workers} workers"
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_sample, sample_id, model_name, complete, model_id): sample_id
            for sample_id in ids
        }
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as error:
                result = f"{futures[future]}: FAILED -> {error}"
            print(f"[{done}/{len(ids)}] {result}")

    print("Finished.")


def find_missing():
    """Human samples that still lack an AI-assisted or AI-generated text, per model."""
    human_ids = load_human_ids()
    missing = {}
    for model_name, spec in cfg.MODELS.items():
        first, last = spec["seed_ids"]
        missing[model_name] = {
            kind: [
                i
                for i in range(first, last + 1)
                if i in human_ids and not (folder / f"{i}.txt").exists()
            ]
            for kind, folder in [
                ("ai_assisted", cfg.ASSISTED_DIR),
                ("ai_generated", cfg.GENERATED_DIR),
            ]
        }
    return missing


def show_example(sample_id, max_chars=700):
    """Print the human text next to its two AI versions."""
    for title, folder in [
        ("HUMAN", cfg.HUMAN_DIR),
        ("AI-ASSISTED", cfg.ASSISTED_DIR),
        ("AI-GENERATED", cfg.GENERATED_DIR),
    ]:
        path = folder / f"{sample_id}.txt"
        text = path.read_text(encoding="utf-8") if path.exists() else "(missing)"
        print(f"\n===== {title} ({count_words(text)} words) =====")
        print(text[:max_chars] + (" [...]" if len(text) > max_chars else ""))
