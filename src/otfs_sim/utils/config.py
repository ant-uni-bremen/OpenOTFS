from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class SimulationConfig:
    # ============================================================
    # Basic simulation identity
    # ============================================================
    waveform: str
    scenario_name: str = "default"
    seed: int = 42

    # ============================================================
    # Grid parameters
    # ============================================================
    num_ofdm_symbols: int = 8
    fft_size: int = 16
    subcarrier_spacing: float = 15000
    cyclic_prefix_length: int = 4

    # ============================================================
    # OFDM pilot pattern
    # ============================================================
    pilot_ofdm_symbol_indices: list = None

    # ============================================================
    # Modulation / coding
    # ============================================================
    num_bits_per_symbol: int = 2
    coderate: float = 0.5
    coded_ber: bool = True

    # ============================================================
    # Channel
    # ============================================================
    channel_case: str = "single"
    channel_backend: str = "manual"

    channel_l_min: int = 0
    channel_l_max: int | None = None
    channel_normalize: bool = False

    # ============================================================
    # Receiver options
    # ============================================================
    receiver: str = None
    
    use_lmmse: bool = False
    csi_mode: str = "perfect"
    
    def __post_init__(self):
        if self.receiver == "no_lmmse":
            self.use_lmmse = False
            self.csi_mode = "estimated"   # doesn't matter, won't be used

        elif self.receiver == "perfect_csi_lmmse":
            self.use_lmmse = True
            self.csi_mode = "perfect"

        elif self.receiver == "estimated_csi_lmmse":
            self.use_lmmse = True
            self.csi_mode = "estimated"

    # ============================================================
    # OTFS-only embedded pilot settings
    # ============================================================
    use_pilots: bool = False
    lmax: int = 0
    kmax: int = 0
    pilot_symbol_real: float = 1.0
    pilot_symbol_imag: float = 0.0
    estimation_threshold_factor: float = 1.732
    estimator_detection_mode: str = "threshold"
    estimator_fp_min_persistence: float | None = None

    # ============================================================
    # Runtime
    # ============================================================
    batch_size: int = 1
    ebno_db: list = None

    max_mc_iter: int = 20
    num_target_block_errors: int = 20

    @property
    def pilot_symbol(self):
        return complex(self.pilot_symbol_real, self.pilot_symbol_imag)


def load_config(config_path: str | Path) -> SimulationConfig:
    """
    Load YAML simulation config and return SimulationConfig object.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    return SimulationConfig(**cfg)