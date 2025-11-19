from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 64,
        "d_model": 512,
        "d_ff": 2048,
        "num_block": 6,
        "num_head": 8,
        "dropout": 0.1,
        "lang_src": "en",
        "lang_tgt": "de",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    } 

def get_weights_file_path(config):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = 'best.pt'
    return str(Path('.')/model_folder/model_filename)