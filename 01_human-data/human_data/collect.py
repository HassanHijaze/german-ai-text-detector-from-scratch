"""Collection loop over the repositories, plus merge and summary helpers."""

import datetime as dt
import json
import shutil
from collections import Counter
from pathlib import Path

from . import config as cfg
from .pdf_text import PdfSkipped
from .sources import (
    OAI_NS,
    choose_publication_date,
    dc_values,
    detect_open_license,
    download_pdf_text,
    find_pdf_url,
    metadata_is_german,
    oai_pages,
)
from .text_utils import detect_language, sha256_text
from .writer import DatasetWriter

# Document types that are not running academic prose
EXCLUDED_TYPES = (
    "researchdata",
    "dataset",
    "image",
    "video",
    "sound",
    "audio",
    "software",
    "lecture",
    "patent",
)


def record_is_deleted(record):
    header = record.find(f"{{{OAI_NS}}}header")

    return header is not None and header.get("status") == "deleted"


def record_identifier(record, fallback):
    header = record.find(f"{{{OAI_NS}}}header")

    if header is None:
        return fallback

    return header.findtext(f"{{{OAI_NS}}}identifier") or fallback


def metadata_looks_german(record):
    """
    SSOAR has no dc:language, so fall back to detecting the
    language of title + abstract before downloading the PDF.
    """

    languages = dc_values(record, "language")

    if languages:
        return metadata_is_german(languages)

    sample = " ".join(dc_values(record, "title")[:1] + dc_values(record, "description")[:1])

    if len(sample) < 40:
        return True

    return detect_language(sample) == "de"


def record_pdf_url(identifiers, landing_host):
    # peDOCS lists the PDF directly
    direct = next(
        (
            value
            for value in identifiers
            if value.lower().endswith(".pdf") and landing_host in value
        ),
        None,
    )

    if direct:
        return direct, direct

    landing_url = next(
        (
            value
            for value in identifiers
            if landing_host in value and value.startswith("http") and "/pdf/" not in value
        ),
        None,
    )

    if not landing_url:
        return None, None

    return find_pdf_url(landing_url), landing_url


def record_candidate(writer, record):
    """
    Cheap metadata checks (no download). Returns a small dict
    with what is needed later, or None if the record is not
    usable.
    """

    if record_is_deleted(record):
        return None

    types = dc_values(record, "type")

    if any(value in " ".join(types).lower() for value in EXCLUDED_TYPES):
        writer.reject("record_excluded_type")
        return None

    publication_date = choose_publication_date(dc_values(record, "date"))

    if (
        not publication_date
        or publication_date < cfg.MIN_PUBLICATION_DATE
        or publication_date > cfg.CUTOFF_DATE
    ):
        writer.reject("record_outside_date_range")
        return None

    licence = detect_open_license(dc_values(record, "rights"))

    if not licence:
        writer.reject("record_no_open_license")
        return None

    if not metadata_looks_german(record):
        writer.reject("record_not_german")
        return None

    identifiers = dc_values(record, "identifier")

    return {
        "identifier": record_identifier(record, identifiers[0] if identifiers else ""),
        "identifiers": identifiers,
        "titles": dc_values(record, "title"),
        "creators": dc_values(record, "creator"),
        "types": types,
        "publication_date": publication_date,
        "licence": licence,
    }


def process_candidate(writer, source, candidate, target):
    """Download one candidate document and add its chunks."""

    info = cfg.SOURCE_INFO[source]

    label = info["name"]

    document_id = f"{source}-" + sha256_text(candidate["identifier"])[:20]

    if document_id in writer.seen_documents:
        return 0

    pdf_url, landing_url = record_pdf_url(candidate["identifiers"], info["landing_host"])

    if not pdf_url:
        return writer.reject("record_no_pdf")

    try:
        text = download_pdf_text(pdf_url)
    except PdfSkipped as skipped:
        return writer.reject(skipped.reason, pdf_url)
    except Exception:
        return writer.reject("record_pdf_download_failed", pdf_url)

    titles = candidate["titles"]

    doc_types = [
        value
        for value in candidate["types"]
        if value.lower()
        not in {"text", "publishedversion", "acceptedversion", "updatedversion", "submittedversion"}
    ]

    publication_date = candidate["publication_date"]

    added = writer.add_document(
        source=source,
        title=(titles[0] if titles else candidate["identifier"]),
        document_id=document_id,
        document_type=(doc_types[0] if doc_types else None),
        text=text,
        publication_date=publication_date.isoformat(),
        date_basis=f"{label} publication metadata",
        source_url=landing_url,
        authors=candidate["creators"],
        license_name=candidate["licence"][0],
        license_url=candidate["licence"][1],
        target_total=target,
    )

    if added:
        print(
            f"[{label}] "
            f"{writer.source_count(source)}"
            f"/{target} "
            f"(+{added}) "
            f"{publication_date.year} | "
            f"{(titles[0] if titles else '')[:70]}"
        )

    return added


def collect_repository(writer, source, target=None):
    """
    Pass 1: walk the repository in random year windows (by
            record datestamp) and take only a few documents per
            page / window -> a broad mix of years, topics and
            series. Every other eligible record is remembered.
    Pass 2: if the target is not reached yet, use all remembered
            records (random order) until the target is reached
            or the repository has nothing left.
    """
    if target is None:
        target = cfg.TARGET_PER_SOURCE

    info = cfg.SOURCE_INFO[source]

    label = info["name"]

    years = list(range(info["first_year"], dt.date.today().year + 1))

    cfg.RANDOM.shuffle(years)

    deferred = []

    # ---------------- Pass 1 ----------------
    for year in years:
        if writer.source_count(source) >= target:
            break

        docs_in_window = 0
        scanned = 0
        candidates_in_window = 0

        try:
            for records in oai_pages(
                info["oai"], date_from=f"{year}-01-01", date_until=f"{year}-12-31"
            ):

                records = list(records)

                cfg.RANDOM.shuffle(records)

                docs_on_page = 0

                for record in records:
                    if writer.source_count(source) >= target:
                        break

                    scanned += 1

                    candidate = record_candidate(writer, record)

                    if candidate is None:
                        continue

                    candidates_in_window += 1

                    # Enough from this page / window for now:
                    # keep it for pass 2 instead of dropping it
                    if (
                        docs_on_page >= cfg.MAX_DOCS_PER_OAI_PAGE
                        or docs_in_window >= cfg.MAX_DOCS_PER_YEAR_WINDOW
                    ):
                        deferred.append(candidate)
                        continue

                    if process_candidate(writer, source, candidate, target):
                        docs_on_page += 1
                        docs_in_window += 1

                if writer.source_count(source) >= target:
                    break

        except KeyboardInterrupt:
            raise

        except Exception as error:
            print(f"[{label}] {year} window failed:", error)

        print(
            f"[{label}] records added {year}: "
            f"{scanned} checked, "
            f"{candidates_in_window} eligible, "
            f"{docs_in_window} used now, "
            f"{len(deferred)} waiting for pass 2"
        )

    # ---------------- Pass 2 ----------------
    if writer.source_count(source) < target and deferred:
        print(f"\n[{label}] pass 2: " f"{len(deferred)} more eligible documents")

        cfg.RANDOM.shuffle(deferred)

        for candidate in deferred:
            if writer.source_count(source) >= target:
                break

            try:
                process_candidate(writer, source, candidate, target)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                print(f"[{label}] document failed:", error)

    if writer.source_count(source) < target:
        print(
            f"\n[{label}] repository exhausted: "
            f"{writer.source_count(source)}/{target} chunks "
            f"(no more eligible documents in "
            f"{cfg.MIN_PUBLICATION_DATE.year}–{cfg.CUTOFF_DATE.year})"
        )


# ---- Main ----


def collect_all():
    """Collect chunks from every source until cfg.TARGET_PER_SOURCE is reached.

    Resumes from an existing meta.json in cfg.DATA_DIR.
    """
    writer = DatasetWriter(cfg.DATA_DIR, cfg.CUTOFF_DATE)

    for source in cfg.SOURCE_INFO:
        current = writer.source_count(source)

        print("\n" + "=" * 65)
        print(f"{source.upper()}: " f"{current}/{cfg.TARGET_PER_SOURCE}")
        print("=" * 65)

        if current >= cfg.TARGET_PER_SOURCE:
            print("Already complete.")
            continue

        try:
            collect_repository(writer, source, cfg.TARGET_PER_SOURCE)

        except KeyboardInterrupt:
            writer.save()
            print("\nStopped. Progress saved.")
            raise

        except Exception as error:
            writer.save()
            print(f"{source} failed: {error}")

    writer.save()

    print("\n" + "=" * 65)
    print("FINISHED")
    print("=" * 65)

    print(
        json.dumps(
            {
                "by_source": writer.meta["counts_by_source"],
                "by_split": writer.meta["counts_by_split"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nSkipped (reason: count):")

    for reason, count in writer.rejections.most_common():
        print(f"  {reason}: {count}")

        for example in writer.rejection_examples.get(reason, []):
            print(f"      e.g. {example}")

    print("\nTotal human samples:", len(writer.samples))

    return writer


def merge_datasets(folders, output):
    """Combine several download folders into one dataset.

    Duplicate chunks (same hash) are skipped and samples are renumbered.
    The output folder is deleted first, so it must not be one of the inputs.
    """
    output = Path(output)
    if any(output.resolve() == Path(folder).resolve() for folder in folders):
        raise ValueError("output must be a new folder, not one of the input folders")

    shutil.rmtree(output, ignore_errors=True)
    (output / "human").mkdir(parents=True)

    merged_meta = None
    samples = []
    hashes = set()

    for folder in folders:
        folder = Path(folder)

        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))

        if merged_meta is None:
            merged_meta = {key: value for key, value in meta.items() if key != "samples"}

        for sample in meta["samples"]:
            # Same chunk downloaded twice
            if sample["hash"] in hashes:
                continue

            sample_id = len(samples) + 1

            text = (folder / sample["file"]).read_text(encoding="utf-8")

            sample = dict(
                sample,
                sample_id=sample_id,
                file=f"human/{sample_id}.txt",
                downloaded_in=folder.name,
            )

            (output / sample["file"]).write_text(text, encoding="utf-8")

            samples.append(sample)
            hashes.add(sample["hash"])

    dates = sorted(sample["publication_date"] for sample in samples)

    merged_meta["publication_date_range"] = [dates[0], dates[-1]]
    merged_meta["merged_from"] = [str(folder) for folder in folders]
    merged_meta["counts"] = {"total": len(samples), "human": len(samples)}
    merged_meta["counts_by_source"] = dict(Counter(s["source"] for s in samples))
    merged_meta["counts_by_split"] = dict(Counter(s["split"] for s in samples))
    merged_meta["samples"] = samples

    (output / "meta.json").write_text(
        json.dumps(merged_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Merged {len(samples)} samples into {output}/")
    print("By source:", merged_meta["counts_by_source"])
    print("By split: ", merged_meta["counts_by_split"])
    print("Years:    ", merged_meta["publication_date_range"])


def summarize(data_dir=None, n_examples=3):
    """Print counts, length statistics and a few random samples."""
    import random
    import statistics

    data_dir = Path(data_dir or cfg.DATA_DIR)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    samples = meta["samples"]

    print("Samples:          ", len(samples))
    print("Documents:        ", len({s["document_id"] for s in samples}))
    print("By source:        ", dict(Counter(s["source"] for s in samples)))
    print("By split:         ", dict(Counter(s["split"] for s in samples)))
    years = Counter(s["publication_date"][:4] for s in samples)
    print("Years:            ", f"{min(years)}–{max(years)}")
    print("Document types:   ", Counter(s.get("document_type") for s in samples).most_common(5))

    print("\nTarget length -> actual word counts:")
    for target in cfg.TARGET_WORDS:
        counts = [s["actual_word_count"] for s in samples if s["target_length"] == target]
        if counts:
            print(
                f"  {target:>5}: n={len(counts):>5}  min={min(counts)}  "
                f"median={statistics.median(counts)}  max={max(counts)}"
            )

    for sample in random.sample(samples, min(n_examples, len(samples))):
        text = (data_dir / sample["file"]).read_text(encoding="utf-8")
        print("\n" + "-" * 65)
        print(
            f"#{sample['sample_id']} | {sample['source']} | "
            f"{sample['publication_date'][:4]} | {sample['title'][:60]}"
        )
        print(f"{sample['actual_word_count']} words | split: {sample['split']}\n")
        print(text[:800] + (" [...]" if len(text) > 800 else ""))
