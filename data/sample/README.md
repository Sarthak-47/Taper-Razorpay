# Sample period

Three CSV files in the shape a merchant actually receives them: a settlement
report from the gateway, a bank statement, and the merchant's own ledger. They
disagree with each other, on purpose.

Reconcile them:

```bash
python -m taper.cli ingest --settlement data/sample/settlement.csv \
    --bank data/sample/bank.csv --ledger data/sample/ledger.csv
```

**This data is synthetic and reproducible.** It was written by:

```bash
python -m taper.cli --no-llm export --seed 7 --batches 25 --out data/sample
```

Regenerating with that seed reproduces these files byte for byte. No real
merchant data appears anywhere in this repository — the track brief permits
synthetic data, and a reconciliation engine is a poor reason to handle somebody
else's settlement history.

The defects in here are injected at declared rates and labelled, which is what
lets [`taper reconcile`](../../src/taper/metrics/harness.py) report precision
and recall against ground truth rather than against its own opinion.
