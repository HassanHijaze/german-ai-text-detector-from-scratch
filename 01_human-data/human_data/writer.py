"""Document-level train/validation/test split and the dataset writer (meta.json + text files)."""

import datetime as dt
import hashlib
import json
import os
from collections import Counter

from . import config as cfg
from .chunking import chunk_document
from .pdf_text import clean_plain_text, document_damage
from .quality import is_good_academic_chunk, modernize_spelling, uses_old_spelling
from .text_utils import character_count, is_german, parse_date, sha256_text, word_count


def assign_split(document_id):
    value = (int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16) % 1000) / 1000

    cumulative = 0.0

    for name, ratio in cfg.SPLIT_RATIOS.items():
        cumulative += ratio

        if value < cumulative:
            return name

    return "train"


class DatasetWriter:
    def __init__(self, data_dir, cutoff):
        self.data_dir = data_dir

        self.human_dir = data_dir / "human"

        self.meta_path = data_dir / "meta.json"

        self.cutoff = cutoff

        self.human_dir.mkdir(parents=True, exist_ok=True)

        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

        else:
            self.meta = {
                "schema_version": 4,
                "language": "de",
                "domain": "academic",
                "publication_date_range": [
                    cfg.MIN_PUBLICATION_DATE.isoformat(),
                    cutoff.isoformat(),
                ],
                "target_chunk_words": list(cfg.TARGET_WORDS),
                "max_length_deviation": cfg.MAX_LENGTH_DEVIATION,
                "sentence_aware_chunking": True,
                "quality_filter": True,
                "whitespace": "flattened to single spaces",
                "split_by": "document_id",
                "max_chunks_per_document": cfg.MAX_CHUNKS_PER_DOCUMENT,
                "counts": {"total": 0, "human": 0},
                "samples": [],
            }

        self.samples = self.meta.setdefault("samples", [])

        self.hashes = {sample["hash"] for sample in self.samples if sample.get("hash")}

        self.chunk_keys = {
            (sample.get("document_id"), sample.get("chunk_number")) for sample in self.samples
        }

        self.seen_documents = {sample.get("document_id") for sample in self.samples}

        ids = [
            int(sample["sample_id"])
            for sample in self.samples
            if sample.get("sample_id") is not None
        ]

        self.next_id = max(ids, default=0) + 1

        # Why documents / chunks were skipped (test runs),
        # with a few example URLs per reason to look at
        self.rejections = Counter()
        self.rejection_examples = {}

    def reject(self, reason, example=None):
        self.rejections[reason] += 1

        if example:
            examples = self.rejection_examples.setdefault(reason, [])

            if len(examples) < 3 and example not in examples:
                examples.append(example)

        return 0

    def source_count(self, source):
        return sum(1 for sample in self.samples if sample.get("source") == source)

    def add_document(
        self,
        *,
        source,
        title,
        document_id,
        document_type,
        text,
        publication_date,
        date_basis,
        source_url,
        authors,
        license_name,
        license_url,
        target_total,
    ):

        if self.source_count(source) >= target_total:
            return 0

        if document_id in self.seen_documents:
            return self.reject("document_already_used")

        if not text:
            return self.reject("document_no_text", source_url)

        text = clean_plain_text(text)

        if word_count(text) < cfg.MIN_DOCUMENT_WORDS:
            return self.reject("document_too_short_after_cleaning", source_url)

        if not is_german(text):
            return self.reject("document_not_german", source_url)

        # 1990s (and early 2000s) texts: convert old ß spelling
        text, spelling_changes = modernize_spelling(text)

        if uses_old_spelling(text):
            return self.reject("document_old_spelling", source_url)

        damage = document_damage(text)

        if damage:
            return self.reject(damage, source_url)

        parsed_date = parse_date(publication_date)

        if parsed_date is None:
            return self.reject("document_no_date")

        if not (cfg.MIN_PUBLICATION_DATE <= parsed_date <= self.cutoff):
            return self.reject("document_outside_date_range")

        if not license_name:
            return self.reject("document_no_open_license")

        original_document_length = word_count(text)

        chunks = chunk_document(text, document_id)

        if not chunks:
            return self.reject("document_no_valid_chunks")

        split = assign_split(document_id)

        added = 0

        for chunk_info in chunks:
            if self.source_count(source) >= target_total:
                break

            chunk_number = chunk_info["chunk_number"]

            key = (document_id, chunk_number)

            if key in self.chunk_keys:
                continue

            chunk = chunk_info["text"]

            # =================================================
            # QUALITY FILTER
            # =================================================

            ok, reason = is_good_academic_chunk(chunk)

            if not ok:
                self.reject(reason)
                continue

            digest = sha256_text(chunk)

            if digest in self.hashes:
                self.reject("chunk_duplicate")
                continue

            sample_id = self.next_id

            relative_file = f"human/" f"{sample_id}.txt"

            output_file = self.data_dir / relative_file

            output_file.write_text(chunk.rstrip() + "\n", encoding="utf-8")

            source_info = cfg.SOURCE_INFO[source]

            sample = {
                "sample_id": sample_id,
                "file": relative_file,
                "label": "human",
                "language": "de",
                "split": split,
                "source": source,
                "source_name": source_info["name"],
                "source_type": source_info["source_type"],
                "document_type": document_type,
                "title": title,
                "document_id": document_id,
                "chunk_number": chunk_number,
                "target_length": chunk_info["target_length"],
                "actual_word_count": word_count(chunk),
                "character_count": character_count(chunk),
                "publication_date": parsed_date.isoformat(),
                "date_basis": date_basis,
                "retrieval_date": dt.datetime.now(dt.timezone.utc).isoformat(),
                "license_name": license_name,
                "license_url": license_url,
                "source_url": source_url,
                "author": authors,
                "hash": digest,
                "original_document_length": original_document_length,
                # Text right before the chunk (for generating
                # the matching AI sample in the same context)
                "context_before": chunk_info["context_before"],
                # Words converted from old spelling in the
                # whole document ("daß" -> "dass"), 0 if none
                "spelling_modernized": spelling_changes,
            }

            self.samples.append(sample)

            self.hashes.add(digest)

            self.chunk_keys.add(key)

            self.next_id += 1

            added += 1

        if added:
            self.seen_documents.add(document_id)
            self.save()

        return added

    def save(self):
        self.meta["counts_by_source"] = {
            source: self.source_count(source) for source in cfg.SOURCE_INFO
        }

        self.meta["counts_by_split"] = dict(Counter(sample.get("split") for sample in self.samples))

        self.meta["counts"] = {"total": len(self.samples), "human": len(self.samples)}

        temporary = self.meta_path.with_suffix(".json.tmp")

        temporary.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        os.replace(temporary, self.meta_path)
