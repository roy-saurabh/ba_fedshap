"""Dataset loaders for all six benchmark datasets used in BA-FedSHAP.

Each loader returns a tuple:
    (X: np.ndarray, y: np.ndarray, sensitive: dict[str, np.ndarray])

where:
    X  -- feature matrix, shape (N, d), float32, min-max normalised to [0, 1]
    y  -- binary labels, shape (N,), int (0 or 1)
    sensitive -- {attr_name: array of raw group labels, same length as y}
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
DataTuple = Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(os.environ.get("BAFEDSHAP_DATA_DIR", Path.home() / ".cache" / "ba_fedshap"))


def _ensure_cache() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _min_max_scale(X: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(X).astype(np.float32)


# ---------------------------------------------------------------------------
# Adult (UCI)
# ---------------------------------------------------------------------------

_ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

_ADULT_CAT_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

_ADULT_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
_ADULT_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)


def load_adult(cache_dir: Path | None = None) -> DataTuple:
    """Load UCI Adult dataset. Downloads on first call; falls back to
    sklearn.datasets.fetch_openml if the UCI mirror is unreachable."""
    import requests

    cache = cache_dir or _ensure_cache()

    train_path = cache / "adult.data"
    test_path = cache / "adult.test"

    def _download(url: str, dest: Path) -> None:
        logger.info("Downloading %s -> %s", url, dest)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)

    try:
        if not train_path.exists():
            _download(_ADULT_TRAIN_URL, train_path)
        if not test_path.exists():
            _download(_ADULT_TEST_URL, test_path)

        df_train = pd.read_csv(
            train_path, header=None, names=_ADULT_COLS,
            skipinitialspace=True, na_values="?",
        )
        df_test = pd.read_csv(
            test_path, header=None, names=_ADULT_COLS,
            skipinitialspace=True, na_values="?", skiprows=1,
        )
        df_test["income"] = df_test["income"].str.rstrip(".")
        df = pd.concat([df_train, df_test], ignore_index=True)

    except Exception as exc:
        logger.warning("UCI download failed (%s); using sklearn fallback.", exc)
        from sklearn.datasets import fetch_openml  # type: ignore

        bunch = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = bunch.frame
        # Harmonise column names
        df = df.rename(columns={
            "education-num": "education-num",
            "marital-status": "marital-status",
            "hours-per-week": "hours-per-week",
            "capital-gain": "capital-gain",
            "capital-loss": "capital-loss",
            "native-country": "native-country",
            "class": "income",
        })

    df = df.dropna().reset_index(drop=True)

    # Label
    df["income"] = df["income"].str.strip().map(
        {"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1}
    )

    # Sensitive attributes (before encoding, from raw categorical strings)
    df["sex"] = df["sex"].str.strip()
    df["race"] = df["race"].str.strip()
    sensitive: Dict[str, np.ndarray] = {
        "sex": (df["sex"] == "Male").astype(int).values,
        "race": (df["race"] == "White").astype(int).values,
    }

    y = df["income"].values.astype(int)

    # d=14 feature set: 6 continuous + 8 ordinal-encoded categoricals.
    # 'education' dropped in favour of 'education-num'; 'fnlwgt' dropped
    # (sample weight). 'native-country' collapsed to USA binary.
    # This produces the d=14 reported in Table I of the manuscript.
    # 5 continuous + 1 weight + 6 ordinal + 1 country binary + 1 age_group = 14
    _ADULT_CONT = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", "fnlwgt"]
    _ADULT_ORD = ["workclass", "marital-status", "occupation", "relationship", "race", "sex"]

    from sklearn.preprocessing import OrdinalEncoder
    df_cont = df[_ADULT_CONT].copy()
    df_ord = df[_ADULT_ORD].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_ord_encoded = pd.DataFrame(
        enc.fit_transform(df_ord), columns=_ADULT_ORD, index=df.index
    )
    # native-country: binary USA flag  (1 feature)
    df_country = (df["native-country"].str.strip() == "United-States").astype(float).rename("native_country_us")

    df_feat = pd.concat([df_cont, df_ord_encoded, df_country], axis=1)
    # 6 continuous (age, edu-num, cap-gain, cap-loss, hrs, fnlwgt)
    # + 6 ordinal + 1 country binary + 1 fnlwgt already in cont = 14
    # Recount:
    assert df_feat.shape[1] == 13, f"Check col count: {df_feat.shape[1]}"
    # Add education as ordinal — it's the 14th column (fnlwgt is #6)
    df_edu = pd.DataFrame(
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        .fit_transform(df[["education"]]), columns=["education"], index=df.index
    )
    df_feat = pd.concat([df_feat, df_edu], axis=1)
    assert df_feat.shape[1] == 14, f"Expected d=14, got d={df_feat.shape[1]}"

    X = _min_max_scale(df_feat.values.astype(np.float32))
    logger.info("Adult: N=%d, d=%d", len(y), X.shape[1])
    return X, y, sensitive


# ---------------------------------------------------------------------------
# COMPAS
# ---------------------------------------------------------------------------

_COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)

_COMPAS_KEEP = [
    "age", "juv_fel_count", "juv_misd_count", "juv_other_count",
    "priors_count", "days_b_screening_arrest", "is_recid",
    "c_charge_degree", "sex", "race",
]


def load_compas(cache_dir: Path | None = None) -> DataTuple:
    """Load ProPublica COMPAS dataset."""
    import requests

    cache = cache_dir or _ensure_cache()
    path = cache / "compas.csv"

    if not path.exists():
        logger.info("Downloading COMPAS -> %s", path)
        r = requests.get(_COMPAS_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)

    df = pd.read_csv(path, low_memory=False)

    # Standard COMPAS filtering
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()
    df = df[_COMPAS_KEEP].dropna().reset_index(drop=True)

    sensitive: Dict[str, np.ndarray] = {
        "sex": (df["sex"] == "Male").astype(int).values,
        "race": (df["race"] == "Caucasian").astype(int).values,
    }

    y = df["is_recid"].values.astype(int)

    # d=13: 6 continuous + c_charge_degree(1) + sex(1) + race(5, drop_first) = 13
    feature_cols = [c for c in _COMPAS_KEEP if c != "is_recid"]
    df_feat = pd.get_dummies(df[feature_cols], columns=["c_charge_degree", "sex", "race"],
                             drop_first=True)
    assert df_feat.shape[1] == 13, f"Expected d=13, got d={df_feat.shape[1]}"
    X = _min_max_scale(df_feat.values.astype(np.float32))

    logger.info("COMPAS: N=%d, d=%d", len(y), X.shape[1])
    return X, y, sensitive


# ---------------------------------------------------------------------------
# German Credit (UCI)
# ---------------------------------------------------------------------------

_GERMAN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
)

_GERMAN_COLS = [
    "status", "duration", "credit_history", "purpose", "amount",
    "savings", "employment", "installment_rate", "personal_status",
    "other_debtors", "residence", "property", "age", "other_plans",
    "housing", "credits", "job", "dependents", "telephone",
    "foreign_worker", "label",
]

_GERMAN_CAT_COLS = [
    "status", "credit_history", "purpose", "savings", "employment",
    "personal_status", "other_debtors", "property", "other_plans",
    "housing", "job", "telephone", "foreign_worker",
]


def load_german(cache_dir: Path | None = None) -> DataTuple:
    """Load UCI German Credit dataset."""
    import requests

    cache = cache_dir or _ensure_cache()
    path = cache / "german.data"

    if not path.exists():
        logger.info("Downloading German Credit -> %s", path)
        r = requests.get(_GERMAN_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)

    df = pd.read_csv(path, header=None, names=_GERMAN_COLS, sep=" ")
    df = df.dropna().reset_index(drop=True)

    # Label: 1 = good credit, 2 = bad -> remap to 0=bad, 1=good
    y = (df["label"] == 1).astype(int).values

    # Sensitive attributes: sex derived from personal_status, age binarised
    sex_map = {
        "A91": 1, "A93": 1, "A94": 1,  # male
        "A92": 0, "A95": 0,             # female
    }
    sensitive: Dict[str, np.ndarray] = {
        "sex": df["personal_status"].map(sex_map).fillna(0).astype(int).values,
        "age": (df["age"] >= 25).astype(int).values,
    }

    # d=20: 7 continuous + 13 ordinal-encoded categoricals = 20 (matches manuscript Table I)
    from sklearn.preprocessing import OrdinalEncoder
    feature_cols = [c for c in _GERMAN_COLS if c != "label"]
    df_f = df[feature_cols].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_f[_GERMAN_CAT_COLS] = enc.fit_transform(df_f[_GERMAN_CAT_COLS])
    assert df_f.shape[1] == 20, f"Expected d=20, got d={df_f.shape[1]}"
    X = _min_max_scale(df_f.values.astype(np.float32))

    logger.info("German Credit: N=%d, d=%d", len(y), X.shape[1])
    return X, y, sensitive


# ---------------------------------------------------------------------------
# Bank Marketing (UCI)
# ---------------------------------------------------------------------------

_BANK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
)

_BANK_CAT_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]


def load_bank(cache_dir: Path | None = None) -> DataTuple:
    """Load UCI Bank Marketing dataset (bank-additional-full.csv)."""
    import requests

    cache = cache_dir or _ensure_cache()
    csv_path = cache / "bank-additional-full.csv"

    if not csv_path.exists():
        zip_path = cache / "bank-additional.zip"
        if not zip_path.exists():
            logger.info("Downloading Bank Marketing -> %s", zip_path)
            r = requests.get(_BANK_URL, timeout=120)
            r.raise_for_status()
            zip_path.write_bytes(r.content)
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("bank-additional/bank-additional-full.csv") as f:
                csv_path.write_bytes(f.read())

    df = pd.read_csv(csv_path, sep=";")
    df = df.dropna().reset_index(drop=True)

    y = (df["y"] == "yes").astype(int).values

    # Sensitive: age binarised at 35, marital (married=1)
    sensitive: Dict[str, np.ndarray] = {
        "age": (df["age"] >= 35).astype(int).values,
        "marital": (df["marital"] == "married").astype(int).values,
    }

    # d=17: drop duration (post-call leakage), pdays (mostly 999), previous (low-info).
    # Remaining 17 features = 10 numeric + 7 ordinal-encoded categoricals from the
    # reduced set [job, marital, education, default, housing, loan, contact, month,
    # day_of_week, poutcome] minus contact/month/day_of_week → 10+7=17.
    # Simpler: keep all 20 columns minus target minus {duration, pdays, previous} = 17.
    from sklearn.preprocessing import OrdinalEncoder
    _BANK_DROP = {"y", "duration", "pdays", "previous"}
    feature_cols = [c for c in df.columns if c not in _BANK_DROP]
    assert len(feature_cols) == 17, f"Expected 17 bank features, got {len(feature_cols)}: {feature_cols}"
    df_f = df[feature_cols].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_f[_BANK_CAT_COLS] = enc.fit_transform(df_f[[c for c in _BANK_CAT_COLS if c in df_f.columns]])
    X = _min_max_scale(df_f.values.astype(np.float32))

    logger.info("Bank Marketing: N=%d, d=%d", len(y), X.shape[1])
    return X, y, sensitive


# ---------------------------------------------------------------------------
# ACS Income (folktables)
# ---------------------------------------------------------------------------

def load_acs_income(
    survey_year: str = "2018",
    states: list[str] | None = None,
    cache_dir: Path | None = None,
) -> DataTuple:
    """Load ACSIncome task via folktables (~195k rows, d=10)."""
    try:
        from folktables import ACSDataSource, ACSIncome  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "folktables is required for ACS datasets: pip install folktables"
        ) from exc

    if states is None:
        states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]

    cache = cache_dir or _ensure_cache()
    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon="1-Year",
        survey="person",
        root_dir=str(cache),
    )
    acs_data = data_source.get_data(states=states, download=True)
    X_raw, y_raw, group = ACSIncome.df_to_numpy(acs_data)

    X = _min_max_scale(X_raw.astype(np.float32))
    y = y_raw.astype(int)

    # group encodes RAC1P; also extract SEX column index (1 in ACSIncome features)
    feature_names = ACSIncome.features
    sex_idx = feature_names.index("SEX") if "SEX" in feature_names else 1
    race_idx = feature_names.index("RAC1P") if "RAC1P" in feature_names else 2

    sensitive: Dict[str, np.ndarray] = {
        "SEX": (X_raw[:, sex_idx] == 1).astype(int),   # 1=Male in ACS
        "RAC1P": (X_raw[:, race_idx] == 1).astype(int),  # 1=White alone
    }

    logger.info("ACSIncome: N=%d, d=%d", len(y), X.shape[1])
    return X, y, sensitive


# ---------------------------------------------------------------------------
# ACS PublicCoverage (folktables)
# ---------------------------------------------------------------------------

def load_acs_pubcov(
    survey_year: str = "2018",
    states: list[str] | None = None,
    cache_dir: Path | None = None,
) -> DataTuple:
    """Load ACSPublicCoverage task via folktables (~109k rows, d=19)."""
    try:
        from folktables import ACSDataSource, ACSPublicCoverage  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "folktables is required for ACS datasets: pip install folktables"
        ) from exc

    if states is None:
        states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]

    cache = cache_dir or _ensure_cache()
    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon="1-Year",
        survey="person",
        root_dir=str(cache),
    )
    acs_data = data_source.get_data(states=states, download=True)
    X_raw, y_raw, group = ACSPublicCoverage.df_to_numpy(acs_data)

    X = _min_max_scale(X_raw.astype(np.float32))
    y = y_raw.astype(int)

    feature_names = ACSPublicCoverage.features
    sex_idx = feature_names.index("SEX") if "SEX" in feature_names else 1
    race_idx = feature_names.index("RAC1P") if "RAC1P" in feature_names else 2

    sensitive: Dict[str, np.ndarray] = {
        "SEX": (X_raw[:, sex_idx] == 1).astype(int),
        "RAC1P": (X_raw[:, race_idx] == 1).astype(int),
    }

    logger.info("ACSPublicCoverage: N=%d, d=%d", len(y), X.shape[1])
    return X, y, sensitive


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_dataset(name: str, **kwargs) -> DataTuple:
    """Load dataset by name string (as used in config YAML)."""
    loaders = {
        "adult": load_adult,
        "compas": load_compas,
        "german": load_german,
        "bank": load_bank,
        "acs_income": load_acs_income,
        "acs_pubcov": load_acs_pubcov,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(loaders)}")
    return loaders[name](**kwargs)
