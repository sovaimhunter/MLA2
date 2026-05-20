import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

MODELS = {
    "xgboost": "models.xgboost_model",
    "rf":      "models.rf_model",
    "lr":      "models.lr_model",
    "nn":      "models.nn_model",
    "knn":     "models.knn_model",
}


def main():
    parser = argparse.ArgumentParser(description="CS:GO Round Winner Classifier")
    parser.add_argument(
        "model",
        choices=MODELS.keys(),
        help="Model to train: xgboost | rf | lr | nn",
    )
    args = parser.parse_args()

    print(f"\n{'='*40}")
    print(f"  Training model: {args.model.upper()}")
    print(f"{'='*40}\n")

    import importlib
    module = importlib.import_module(MODELS[args.model])
    module.run()


if __name__ == "__main__":
    main()
