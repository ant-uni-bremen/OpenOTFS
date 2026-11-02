from pathlib import Path
from datetime import datetime
import json
import pickle
import time
import dataclasses


def make_run_id(prefix=None):
    """
    Create a readable unique run id.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if prefix is None:
        return timestamp

    return f"{prefix}_{timestamp}"


def config_to_dict(config):
    """
    Convert SimulationConfig dataclass to a plain dictionary.
    """
    if dataclasses.is_dataclass(config):
        cfg = dataclasses.asdict(config)
    else:
        cfg = dict(config.__dict__)

    # complex numbers are not JSON serializable
    if "pilot_symbol" in cfg:
        cfg["pilot_symbol"] = str(cfg["pilot_symbol"])

    return cfg


def save_config_snapshot(config, output_dir):
    """
    Save the config used for a run as JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config_to_dict(config)

    path = output_dir / "config.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

    return path


def save_results(result_dict, output_dir, filename="results.pkl"):
    """
    Save a result dictionary as pickle.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename

    with open(path, "wb") as f:
        pickle.dump(result_dict, f)

    return path


def load_results(path):
    """
    Load a saved pickle result file.
    """
    path = Path(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def append_result_row(results, row):
    """
    Append one simulation row to a results dictionary.

    Example row:
        {
            "waveform": "otfs",
            "channel_case": "two_path_mild_doppler",
            "use_lmmse": True,
            "csi_mode": "perfect",
            "ebno_db": [...],
            "ber": [...],
            "bler": [...],
            "duration": 12.3,
        }
    """
    for key, value in row.items():
        if key not in results:
            results[key] = []

        results[key].append(value)

    return results


def create_result_row(config, ber, bler, duration, extra=None):
    """
    Build one standardized result row from a config and sim_ber output.
    """
    row = {
        "scenario_name": config.scenario_name,
        "waveform": config.waveform,
        "receiver": config.receiver,
        "channel_case": config.channel_case,
        "channel_backend": config.channel_backend,
        "use_lmmse": config.use_lmmse,
        "csi_mode": config.csi_mode,
        "ebno_db": list(config.ebno_db),
        "ber": ber.tolist() if hasattr(ber, "tolist") else list(ber),
        "bler": bler.tolist() if hasattr(bler, "tolist") else list(bler),
        "duration": duration,
        "num_ofdm_symbols": config.num_ofdm_symbols,
        "fft_size": config.fft_size,
        "cyclic_prefix_length": config.cyclic_prefix_length,
        "num_bits_per_symbol": config.num_bits_per_symbol,
        "coderate": config.coderate,
    }

    if extra is not None:
        row.update(extra)

    return row


class Timer:
    """
    Small context manager for timing simulations.

    Example:
        with Timer() as timer:
            ...
        print(timer.duration)
    """

    def __enter__(self):
        self.start = time.time()
        self.duration = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.duration = self.end - self.start