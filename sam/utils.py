import json
import torch

try:
    import yaml
    has_yaml = True
except ImportError:
    has_yaml = False


def read_cfg_file(cfg_fp):
    if cfg_fp.endswith(".json"):
        with open(cfg_fp, "r") as i_fh:
            model_cfg = json.load(i_fh)
    elif cfg_fp.endswith(".yaml"):
        if not has_yaml:
            raise ImportError("Can not read YAML configuration file, the pyyaml"
                              " library is not installed.")
        with open(cfg_fp, 'r') as i_fh:
            model_cfg = yaml.safe_load(i_fh)
    else:
        raise TypeError(
            f"Invalid extension for configuration file: {cfg_fp}. Must be a"
            " json or yaml file.")
    return model_cfg

def print_msg(msg, verbose=True, tag="verbose"):
    if verbose:
        print(f"[{tag}]:", msg)

def to_json_compatible(report):
    _report = {}
    for k in report:
        if isinstance(report[k], (str, int, float)):
            _report[k] = report[k]
        else:
            try:
                _report[k] = float(report[k])
            except TypeError:
                _report[k] = "ERROR"
    return _report