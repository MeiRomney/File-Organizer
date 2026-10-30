"""Processing-time benchmark: how long do scan, extraction and classification take?

Follows ``docs/dataset_plan.md``: folders of 100, 1,000 and 10,000 files are
sampled *with repetition* from the dataset, and every stage is timed
separately:

* ``scan``        - finding the files (``os.walk``),
* ``extraction``  - reading PDF/DOCX/TXT (sum of the per-file extraction time),
* ``other``       - file-system metadata and record building (loop time minus
                    extraction),
* ``classify``    - ``predict_many`` of a trained classifier, per approach.

What is deliberately kept out of the per-file numbers:

* training time (``fit``) is reported once per classifier, not per file;
* the first prediction is a warm-up (for embeddings it loads the model); its
  time is reported separately as ``warmup``.

Every measurement is repeated ``repeats`` times; tables show the median and
the JSON keeps every run. The sampled files are real copies on disk, so the
numbers include real file reads (the operating-system cache will make repeats
faster than the first run: compare ``runs[0]`` with the median if that matters).
Embeddings are timed without any cache. Peak memory is not measured.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from fileorg.classifiers.base import Classifier
from fileorg.models import FileRecord
from fileorg.pipeline import build_record
from fileorg.scanner import scan

DEFAULT_SIZES = (100, 1_000, 10_000)


# --------------------------------------------------------------- environment
def environment() -> dict:
    """Hardware and library versions, so the numbers can be interpreted later."""
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        pass
    for dist in ("scikit-learn", "sentence-transformers", "torch", "pypdf", "python-docx"):
        try:
            info[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    if "torch" in info:
        try:
            import torch

            info["gpu"] = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            pass
    return info


# ------------------------------------------------------------------ sampling
def build_sample_folder(sources: Sequence[Path], dest: Path, n: int, seed: int) -> list[Path]:
    """Copy ``n`` files sampled with repetition from ``sources`` into a fresh folder."""
    if not sources:
        raise ValueError("no source files to sample from")
    rng = random.Random(f"{seed}/{n}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    picks = rng.choices(list(sources), k=n)
    for i, src in enumerate(picks):
        shutil.copyfile(src, dest / f"{i:06d}_{src.name}")
    return picks


# --------------------------------------------------------------- measurements
def _repeat(fn: Callable[[], object], repeats: int) -> list[float]:
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - start)
    return runs


def _extract_all(root: Path, paths: Sequence[Path]) -> tuple[list[FileRecord], float, float]:
    """(records, extraction seconds, other seconds) for one pass over the folder."""
    start = time.perf_counter()
    records = [build_record(p, root) for p in paths]
    wall = time.perf_counter() - start
    extraction = sum(r.extraction_seconds for r in records)
    return records, extraction, max(0.0, wall - extraction)


@dataclass
class SizeResult:
    n: int
    avg_size_kb: float
    statuses: dict[str, int]
    scan: list[float]
    extraction: list[float]
    other: list[float]
    classify: dict[str, list[float]] = field(default_factory=dict)

    def median(self, stage: str, name: Optional[str] = None) -> float:
        runs = self.classify[name] if stage == "classify" else getattr(self, stage)
        return statistics.median(runs)

    def total(self, name: str) -> float:
        """Scan + extraction + other + classification for one approach (medians)."""
        return sum(self.median(s) for s in ("scan", "extraction", "other")) + self.median("classify", name)


@dataclass
class BenchmarkResult:
    environment: dict
    repeats: int
    seed: int
    fit_seconds: dict[str, float]
    warmup_seconds: dict[str, float]
    sizes: list[SizeResult]

    def to_dict(self) -> dict:
        out = {"environment": self.environment, "repeats": self.repeats, "seed": self.seed,
               "fit_seconds": self.fit_seconds, "warmup_seconds": self.warmup_seconds, "sizes": []}
        for s in self.sizes:
            out["sizes"].append({
                "n": s.n, "avg_size_kb": s.avg_size_kb, "statuses": s.statuses,
                "runs": {"scan": s.scan, "extraction": s.extraction, "other": s.other, "classify": s.classify},
                "median": {"scan": s.median("scan"), "extraction": s.median("extraction"), "other": s.median("other"),
                           "classify": {k: s.median("classify", k) for k in s.classify},
                           "total": {k: s.total(k) for k in s.classify}},
            })
        return out


def run_benchmark(
    source_root: str | Path,
    sizes: Sequence[int],
    makers: dict[str, Callable[[], Classifier]],
    train_records: Sequence[FileRecord],
    work_dir: str | Path,
    repeats: int = 3,
    seed: int = 42,
    keep: bool = False,
    log: Callable[[str], None] = lambda message: None,
) -> BenchmarkResult:
    """Train every classifier once, then time every stage on every folder size."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    source_root, work_dir = Path(source_root), Path(work_dir)
    sources = scan(source_root)
    if not sources:
        raise ValueError(f"no PDF/DOCX/TXT files under {source_root}")

    classifiers: dict[str, Classifier] = {}
    fit_seconds: dict[str, float] = {}
    for name, make in makers.items():
        log(f"training {name} on {len(train_records)} files ...")
        clf = make()
        start = time.perf_counter()
        clf.fit(train_records)
        fit_seconds[name] = time.perf_counter() - start
        classifiers[name] = clf

    warmup: dict[str, float] = {}
    results: list[SizeResult] = []
    for n in sizes:
        folder = work_dir / f"n{n}"
        total_mb = sum(p.stat().st_size for p in sources) / len(sources) * n / 2 ** 20
        log(f"building the {n}-file folder (about {total_mb:.0f} MB) ...")
        picks = build_sample_folder(sources, folder, n, seed)
        try:
            log(f"  scan x{repeats}")
            scan_runs = _repeat(lambda: scan(folder), repeats)
            paths = scan(folder)
            log(f"  extraction x{repeats}")
            ext_runs, other_runs, records = [], [], []
            for _ in range(repeats):
                records, extraction, other = _extract_all(folder, paths)
                ext_runs.append(extraction)
                other_runs.append(other)
            statuses: dict[str, int] = {}
            for r in records:
                statuses[r.extraction_status] = statuses.get(r.extraction_status, 0) + 1
            result = SizeResult(n, statistics.mean(p.stat().st_size for p in picks) / 1024, statuses,
                                scan_runs, ext_runs, other_runs)
            for name, clf in classifiers.items():
                if name not in warmup:  # first prediction: model loading etc.
                    start = time.perf_counter()
                    clf.predict_many(records[:1])
                    warmup[name] = time.perf_counter() - start
                log(f"  classify {name} x{repeats}")
                result.classify[name] = _repeat(lambda c=clf: c.predict_many(records), repeats)
            results.append(result)
        finally:
            if not keep:
                shutil.rmtree(folder, ignore_errors=True)
    return BenchmarkResult(environment(), repeats, seed, fit_seconds, warmup, results)


# ----------------------------------------------------------------- reporting
def format_benchmark(result: BenchmarkResult) -> str:
    env = result.environment
    lines = ["Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()), "",
             f"Medians of {result.repeats} run(s); sampling seed {result.seed}", "",
             "One-time costs: " + "; ".join(
                 f"{n}: fit {result.fit_seconds[n]:.2f}s, warm-up {result.warmup_seconds.get(n, 0):.2f}s"
                 for n in result.fit_seconds)]
    for name in result.fit_seconds:
        lines += ["", name,
                  f"{'files':>7}  {'scan s':>8}  {'extract s':>9}  {'other s':>8}  {'classify s':>10}  "
                  f"{'total s':>8}  {'ms/file':>8}  {'files/s':>8}"]
        for s in result.sizes:
            total = s.total(name)
            lines.append(f"{s.n:7d}  {s.median('scan'):8.2f}  {s.median('extraction'):9.2f}  {s.median('other'):8.2f}  "
                         f"{s.median('classify', name):10.2f}  {total:8.2f}  {total / s.n * 1000:8.2f}  {s.n / total:8.1f}")
    statuses = ", ".join(f"{s.n}: {s.statuses}" for s in result.sizes)
    lines += ["", f"Extraction statuses: {statuses}"]
    return "\n".join(lines)