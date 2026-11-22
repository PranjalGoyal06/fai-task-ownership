# Reproduce the grounded replay

Run from the repository root on the experiment branch. The recorded environment uses Python 3.12. Install the pinned packages into a local virtual environment and check the shared runner first.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r reports/e1/environment.txt
python -m pytest -q
```

Fetch the 33 MB corpus and Django history. The dataset adapter verifies the corpus checksum and fetches any missing base commits by hash. The saved public pull-request merge dates support an offline rerun after caching the corpus and source objects. If the merge cache is omitted, the adapter queries the public pull requests through an authenticated GitHub CLI.

```sh
mkdir -p results/inputs results/regenerated
cp reports/e1/merges.json results/inputs/
cp reports/e1/environment.txt results/regenerated/
curl -fLsS https://huggingface.co/datasets/SWE-bench/SWE-bench/resolve/c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2/data/test-00000-of-00001.parquet -o results/inputs/test.parquet
git clone --bare --single-branch --no-tags https://github.com/django/django.git results/django.git
export MPLCONFIGDIR="$PWD/results/mpl"
export XDG_CACHE_HOME="$PWD/results/cache"
python -m src.e1.dataset results/inputs results/django.git
python -m src.e1.evidence results/inputs results/django.git
python -m src.e1.replay results/inputs results/run
python -m src.e1.summary results/run results/regenerated
python -m src.e1.source_check results/inputs results/django.git results/regenerated/source-check.json
python -m src.e1.verify results/inputs results/django.git results/regenerated/leakage-check.json
python -m src.e1.sensitivity results/inputs results/regenerated/seed-sensitivity.csv
python -m src.e1.figures results/regenerated
python -m src.e1.package results/inputs results/run results/regenerated
```

Compare the generated scientific files using the saved digests. A renderer or font change can alter SVG bytes across platforms. Cohort identifiers, split sizes, predictions and numerical results are the substantive comparisons.

```sh
python - <<'PY'
import hashlib
import json
from pathlib import Path
expected = json.loads(Path("reports/e1/manifest.json").read_text())["output_sha256"]
for name, digest in expected.items():
    observed = hashlib.sha256((Path("results/regenerated") / name).read_bytes()).hexdigest()
    assert observed == digest, name
print(f"{len(expected)} files match")
PY
```

For a quick settlement and selection check, run `python -m pytest -q tests/test_e1.py`. For one real patch perturbation using the prepared feature cache, use the verification command above. Cached evidence includes source revision, extraction signature, module features and citations. Compressed CSV assignment and ownership fields are integer masks with bit positions in manifest module order. The diagnostics file gives every configuration's full parameter values and solver status. Source-pilot citations identify a real revision, file and line.
