"""Evaluation: stratified split + CV folds, metrics, and a common runner."""
from fileorg.evaluation.metrics import (
    ClassMetrics, Report, agreement, classification_report, cohen_kappa,
    format_confusion, format_report,
)
from fileorg.evaluation.runner import CVResult, EvalResult, cross_validate, evaluate, final_test
from fileorg.evaluation.significance import (
    Comparison, Interval, McNemarResult, PairedResult, bootstrap_ci, compare_classifiers, format_comparison,
    mcnemar, paired_bootstrap,
)
from fileorg.evaluation.timing import BenchmarkResult, SizeResult, build_sample_folder, format_benchmark, run_benchmark
from fileorg.evaluation.tuning import TuneResult, expand_grid, grid_search
from fileorg.evaluation.split import (
    Split, apply_split, assign_folds, load_groups, make_split, write_split,
)

__all__ = [
    "ClassMetrics", "Report", "agreement", "classification_report", "cohen_kappa",
    "format_confusion", "format_report", "CVResult", "EvalResult", "cross_validate",
    "evaluate", "final_test", "Split", "apply_split", "assign_folds", "load_groups",
    "make_split", "write_split", "BenchmarkResult", "SizeResult", "build_sample_folder", "format_benchmark", "run_benchmark",
    "TuneResult", "expand_grid", "grid_search", "Comparison", "Interval", "McNemarResult", "PairedResult",
    "bootstrap_ci", "compare_classifiers", "format_comparison", "mcnemar", "paired_bootstrap",
]