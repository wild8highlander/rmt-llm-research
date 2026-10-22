"""evaluation — метрики, интервалы, утечки."""

from .confidence_intervals import (  # noqa: F401
    bootstrap_metric_ci,
    cluster_bootstrap_delta,
    cluster_permutation_paired,
    label_permutation_auc_p,
)
from .leakage_checks import (  # noqa: F401
    assert_no_task_overlap,
    duplicate_prompt_check,
    family_overlap_report,
)
from .metrics import average_precision, brier, log_loss, roc_auc  # noqa: F401
