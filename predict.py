"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions

def single_predict(path_to_model_dir, smiles, feature_type):
    args = parse_predict_args(path_to_model_dir)#path to model dir
    return (make_predictions(args, smiles, feature_type)[1:])
