"""One declared selection objective; training metrics are diagnostic only."""
import math

PRIMARY_METRIC='balanced_accuracy'
METRICS=('balanced_accuracy','macro_f1','macro_auprc','macro_auroc','accuracy')


def validation_selection_score(validation_metrics,metric=PRIMARY_METRIC):
    if metric not in METRICS:raise ValueError('Unsupported validation selection metric')
    score=validation_metrics.get(metric)
    if score is None or not math.isfinite(score):raise ValueError(f'Undefined validation selection metric: {metric}')
    return float(score)


def generalization_gaps(training,validation):
    return {key:float(training[key]-validation[key]) for key in METRICS
            if training.get(key) is not None and validation.get(key) is not None}


def validate_objective(spec):
    expected={'metric':PRIMARY_METRIC,'split':'validation','direction':'maximize'}
    if spec!=expected:raise ValueError('The new protocol must maximize validation balanced_accuracy consistently')
    return PRIMARY_METRIC
