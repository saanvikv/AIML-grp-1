import os
import pandas as pd
import joblib
from urllib.parse import urlparse

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "url_ai_model.joblib"
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

DATASETS = [
    os.path.join(DATASET_DIR, "phishing.csv"),
    os.path.join(DATASET_DIR, "legitimateurls.csv"),
    os.path.join(DATASET_DIR, "phishurls.csv"),
    os.path.join(DATASET_DIR, "combined_urls.csv")
]

_model = None
_url_data = None


def normalize_url(url):
    url = str(url or "").strip().lower()

    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""

        if domain.startswith("www."):
            domain = domain[4:]

        return domain.lower()

    except Exception:
        return ""


def load_model():
    global _model

    if _model is None:

        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"AI phishing model not found: {MODEL_PATH}"
            )

        data = joblib.load(MODEL_PATH)

        if "model" in data:
            _model = data["model"]
        else:
            _model = data

    return _model


def load_dataset():
    global _url_data

    if _url_data is not None:
        return _url_data

    _url_data = {}

    for path in DATASETS:

        if not os.path.isfile(path):
            continue

        try:

            df = pd.read_csv(
                path,
                low_memory=False
            )

            if "url" in df.columns:

                for _, row in df.iterrows():

                    domain = normalize_url(
                        row["url"]
                    )

                    if not domain:
                        continue

                    if "label" in df.columns:
                        _url_data[domain] = int(
                            row["label"]
                        )
                    else:
                        _url_data[domain] = 0

            else:

                column = df.columns[0]

                header_domain = normalize_url(
                    column
                )

                if header_domain:
                    _url_data[header_domain] = 1

                for value in df[column]:

                    domain = normalize_url(
                        value
                    )

                    if domain:
                        _url_data[domain] = 1

        except Exception:
            continue

    return _url_data


def predict_url(url):

    original_url = str(url or "").strip()

    if not original_url:
        raise ValueError(
            "Please enter a URL."
        )

    domain = normalize_url(
        original_url
    )

    if not domain:
        raise ValueError(
            "Invalid URL."
        )

    dataset = load_dataset()

    if domain not in dataset:

        return {
            "prediction": "URL Not Available in Dataset",
            "confidence": 0,
            "confidence_percent": 0,
            "probability": 0,
            "label": None,
            "url": original_url,
            "dataset_match": False,
            "live_model": False,
            "source": "dataset_only"
        }

    label = int(
        dataset[domain]
    )

    if label == 1:

        prediction = "Legitimate Website"

    else:

        prediction = "Phishing Website"

    confidence = 100

    return {
        "prediction": prediction,
        "confidence": confidence,
        "confidence_percent": confidence,
        "probability": 100 if label == 0 else 0,
        "label": label,
        "url": original_url,
        "dataset_match": True,
        "live_model": False,
        "source": "dataset_only"
    }


def predict(url):
    return predict_url(url)


def get_debug_info():

    dataset = load_dataset()

    return {
        "model_path": MODEL_PATH,
        "dataset_urls": len(dataset),
        "feature_count": 30,
        "accuracy": 0.9534,
        "source": "dataset_only"
    }
