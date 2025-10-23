import pandas as pd
from birdnet_custom_classifier_suite.eval_toolkit import schema, signature, review


def test_discover_schema_metrics_prefix():
    df = pd.DataFrame({
        'metrics.ood.best_f1.f1': [0.5],
        'metrics.ood.best_f1.precision': [0.9],
        'metrics.iid.auroc': [0.8],
    })
    sch = schema.discover_schema(df)
    assert 'metrics.ood.best_f1' in sch
    assert 'metrics.iid.auroc'.split('.')[-2] not in sch  # ensure keys are prefixed


def test_discover_schema_legacy_prefix():
    df = pd.DataFrame({
        'ood.best_f1.f1': [0.5],
        'iid.auroc': [0.8],
    })
    sch = schema.discover_schema(df)
    # We normalize to canonical 'metrics.' prefixes
    assert 'metrics.ood.best_f1' in sch
    assert any(k.startswith('metrics.iid') for k in sch.keys())


def test_signature_no_config_columns():
    df = pd.DataFrame({
        'metrics.ood.best_f1.f1': [0.5],
        'metrics.iid.auroc': [0.8],
    })
    df2 = signature.add_signatures(df.copy())
    assert '__signature' in df2.columns
    assert (df2['__signature'] == '').all()


def test_summarize_grouped_basic():
    df = pd.DataFrame({
        '__signature': ['a', 'a', 'b'],
        'metrics.ood.best_f1.f1': [0.5, 0.6, 0.55],
        'metrics.ood.best_f1.precision': [0.9, 0.92, 0.88],
    })
    summ = review.summarize_grouped(df, metric_prefix='metrics.ood.best_f1')
    assert 'metrics.ood.best_f1.f1_mean' in summ.columns
    assert 'metrics.ood.best_f1.precision_std' in summ.columns
