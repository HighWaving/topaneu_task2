from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from .e01_common import N_CLASSES, SEED, sha256_file, write_json


def fit_locked(run: Path, data: dict) -> dict:
    model = ExtraTreesClassifier(n_estimators=512, max_depth=16, min_samples_leaf=2,
                                 max_features=0.5, bootstrap=False, class_weight=None,
                                 n_jobs=4, random_state=SEED)
    model.fit(data["X"], data["y"], sample_weight=data["sample_weight"])
    model_dir = run / "model"; model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "classifier.joblib"
    joblib.dump(model, model_path, compress=3)
    support = {str(i): int(np.sum(data["y"] == i)) for i in range(1, N_CLASSES + 1)}
    learned = [int(x) for x in model.classes_]
    info = {"classes": learned, "class_support_rows": support,
            "unsupported_classes": [i for i in range(1, N_CLASSES + 1) if i not in learned],
            "params": model.get_params(), "fit_once": True, "seed": SEED,
            "model_sha256": sha256_file(model_path)}
    write_json(model_dir / "class_support.json", info)
    write_json(model_dir / "LOCKED.json", {"status": "LOCKED", **info})
    return {"model": model, "path": model_path, "info": info}
