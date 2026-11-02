import numpy as np
import tensorflow as tf

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    ResourceGridDemapper,
    LSChannelEstimator,
    LMMSEEqualizer,
    OFDMModulator,
    OFDMDemodulator,
    RemoveNulledSubcarriers,
)
from sionna.phy.channel import (
    subcarrier_frequencies,
    cir_to_ofdm_channel,
    time_to_ofdm_channel,
    AWGN,
)
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no

from otfs_sim.phy.channel.apply_channel import ApplyCIRTimeChannel
from otfs_sim.phy.channel.generate_channel_cir import (
    DDPathCIRGenerator,
    NTNHardcodedCIRGenerator,
    NTNIntegerDDCIRGenerator,
)
from otfs_sim.phy.channel.channel_utils import (
    get_required_num_cir_time_steps,
    h_time_to_gs,
    dd_paths_to_gs_via_h_time,
)


class OFDMSimulationModel(Block):
    """
    End-to-end OFDM simulation model.

    Supports:
        - no LMMSE
        - Sionna LMMSE equalization
        - perfect CSI
        - LS estimated CSI
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        if self.config.waveform != "ofdm":
            raise ValueError(
                f"OFDMSimulationModel requires waveform='ofdm', "
                f"but got waveform='{self.config.waveform}'."
            )

        if self.config.csi_mode not in ["perfect", "estimated"]:
            raise ValueError(
                f"Unsupported csi_mode='{self.config.csi_mode}'. "
                "Use 'perfect' or 'estimated'."
            )

        # ------------------------------------------------------------
        # Basic single-user SISO setup
        # ------------------------------------------------------------
        self.num_tx = 1
        self.num_streams_per_tx = 1
        self.num_rx = 1

        self.rx_tx_association = np.zeros([self.num_rx, self.num_tx])
        self.rx_tx_association[:, 0] = 1

        self.sm = StreamManagement(
            self.rx_tx_association,
            self.num_streams_per_tx,
        )

        # ------------------------------------------------------------
        # Resource Grid
        # ------------------------------------------------------------
        self.rg = ResourceGrid(
            num_ofdm_symbols=self.config.num_ofdm_symbols,
            fft_size=self.config.fft_size,
            subcarrier_spacing=self.config.subcarrier_spacing,
            num_tx=self.num_tx,
            num_streams_per_tx=self.num_streams_per_tx,
            cyclic_prefix_length=self.config.cyclic_prefix_length,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self.config.pilot_ofdm_symbol_indices,
        )

        # ------------------------------------------------------------
        # Source / coding / mapping
        # ------------------------------------------------------------
        self.binary_source = BinarySource()

        self.n = int(self.rg.num_data_symbols * self.config.num_bits_per_symbol)
        self.k = int(self.n * self.config.coderate)

        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder)

        self.mapper = Mapper("qam", self.config.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.config.num_bits_per_symbol)

        self.rg_mapper = ResourceGridMapper(self.rg)
        self.rg_demapper = ResourceGridDemapper(self.rg, self.sm)

        # ------------------------------------------------------------
        # OFDM waveform
        # ------------------------------------------------------------
        self.modulator = OFDMModulator(self.rg.cyclic_prefix_length)

        self.demodulator = OFDMDemodulator(
            self.rg.fft_size,
            self.config.channel_l_min,
            self.rg.cyclic_prefix_length,
        )

        # ------------------------------------------------------------
        # Channel generation and application
        # ------------------------------------------------------------
        if self.config.channel_backend == "dd":
            self.channel_generator = DDPathCIRGenerator(
                channel_case=self.config.channel_case,
            )

        elif self.config.channel_backend == "cir":
            if self.config.channel_case == "ntn":
                self.channel_generator = NTNHardcodedCIRGenerator()
            elif self.config.channel_case == "ntn_quant":
                self.channel_generator = NTNIntegerDDCIRGenerator()
            else:
                raise ValueError(
                f"Unsupported channel_case='{self.config.channel_case}'. "
                "Use 'ntn' or 'ntn_quant'."
            )
        else:
            raise ValueError(
                f"Unsupported channel_backend='{self.config.channel_backend}'. "
                "Use 'dd' or 'cir'."
            )
            
    
        self.channel = ApplyCIRTimeChannel(
            bandwidth=self.rg.bandwidth,
            l_min=self.config.channel_l_min,
            l_max=self.config.channel_l_max,
            normalize=self.config.channel_normalize,
        )

        self.awgn = AWGN()

        # ------------------------------------------------------------
        # Sionna OFDM receiver blocks
        # ------------------------------------------------------------
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.rg)

        self.ls_estimator = LSChannelEstimator(
            self.rg,
            interpolation_type="lin",
        )

        self.lmmse_equalizer = LMMSEEqualizer(
            self.rg,
            self.sm,
        )

        self.frequencies = subcarrier_frequencies(
            self.rg.fft_size,
            self.rg.subcarrier_spacing,
        )

    def call(self, batch_size, ebno_db):
        # ------------------------------------------------------------
        # Transmitter
        # ------------------------------------------------------------
        no = ebnodb2no(
            ebno_db,
            self.config.num_bits_per_symbol,
            self.config.coderate,
            self.rg,
        )

        b = self.binary_source(
            [batch_size, self.num_tx, self.num_streams_per_tx, self.k]
        )

        c = self.encoder(b)
        x = self.mapper(c)

        # OFDM time-frequency resource grid
        x_rg = self.rg_mapper(x)

        # Time-frequency -> serialized time-domain waveform with CP
        x_time = self.modulator(x_rg)

        # ------------------------------------------------------------
        # Channel + noise
        # ------------------------------------------------------------
        num_time_samples = self.rg.num_time_samples

        num_time_steps = get_required_num_cir_time_steps(
            num_time_samples=num_time_samples,
            l_min=self.config.channel_l_min,
            l_max=self.config.channel_l_max,
        )
        
        a, tau = self.channel_generator(
            batch_size=batch_size,
            num_time_steps=num_time_steps,
            bandwidth=self.rg.bandwidth,
        )
        
        y_time_clean, h_time = self.channel(x_time, a, tau)
        y_time = self.awgn(y_time_clean, no)

        # Demodulate back to OFDM resource grid
        y_rg = self.demodulator(y_time)

        # ------------------------------------------------------------
        # Receiver
        # ------------------------------------------------------------
        if self.config.use_lmmse:
            if self.config.csi_mode == "perfect":
                # Convert the exact time-domain channel used by Sionna
                # into a frequency-domain OFDM channel response.
                h_freq = time_to_ofdm_channel(
                    h_time,
                    self.rg,
                    l_min=self.config.channel_l_min,
                )

                h_hat = self.remove_nulled_scs(h_freq)
                err_var = 0.0

            elif self.config.csi_mode == "estimated":
                h_hat, err_var = self.ls_estimator(y_rg, no)

            else:
                raise ValueError(
                    f"Unsupported csi_mode='{self.config.csi_mode}'. "
                    "Use 'perfect' or 'estimated'."
                )

            x_hat, no_eff = self.lmmse_equalizer(
                y_rg,
                h_hat,
                err_var,
                no,
            )

            llr = self.demapper(x_hat, no_eff)

        else:
            y_for_demapping = self.rg_demapper(y_rg)
            llr = self.demapper(y_for_demapping, no)

        if self.config.coded_ber:
            b_hat = self.decoder(llr)
            return b, b_hat

        return llr, c
    
    def run_frame_debug(self, batch_size, ebno_db):
        """
        Run one OFDM frame and return internal tensors for debugging and plotting.

        This method is intentionally separate from call(), so sim_ber()
        remains clean.
        """

        # ------------------------------------------------------------
        # Transmitter
        # ------------------------------------------------------------
        no = ebnodb2no(
            ebno_db,
            self.config.num_bits_per_symbol,
            self.config.coderate,
            self.rg,
        )

        b = self.binary_source(
            [batch_size, self.num_tx, self.num_streams_per_tx, self.k]
        )

        c = self.encoder(b)
        x = self.mapper(c)

        # OFDM time-frequency grid
        x_rg = self.rg_mapper(x)

        # Time-frequency -> serialized time-domain waveform with CP
        x_time = self.modulator(x_rg)

        # ------------------------------------------------------------
        # Channel + noise
        # ------------------------------------------------------------
        num_time_samples = self.rg.num_time_samples

        num_time_steps = get_required_num_cir_time_steps(
            num_time_samples=num_time_samples,
            l_min=self.config.channel_l_min,
            l_max=self.config.channel_l_max,
        )
        
        a, tau = self.channel_generator(
            batch_size=batch_size,
            num_time_steps=num_time_steps,
            bandwidth=self.rg.bandwidth,
        )
        
        y_time_clean, h_time = self.channel(x_time, a, tau)
        y_time_noisy = self.awgn(y_time_clean, no)

        # Demodulate noisy received waveform before equalization
        y_rg = self.demodulator(y_time_noisy)

        # ------------------------------------------------------------
        # Channel information
        # ------------------------------------------------------------
        h_freq_true = time_to_ofdm_channel(
            h_time,
            self.rg,
            l_min=self.config.channel_l_min,
        )

        h_hat = None
        err_var = None

        if self.config.csi_mode == "perfect":
            h_hat = self.remove_nulled_scs(h_freq_true)
            err_var = 0.0

        elif self.config.csi_mode == "estimated":
            h_hat, err_var = self.ls_estimator(y_rg, no)

        else:
            raise ValueError(
                f"Unsupported csi_mode='{self.config.csi_mode}'. "
                "Use 'perfect' or 'estimated'."
            )

        # ------------------------------------------------------------
        # Optional equalization
        # ------------------------------------------------------------
        if self.config.use_lmmse:
            x_hat, no_eff = self.lmmse_equalizer(
                y_rg,
                h_hat,
                err_var,
                no,
            )

            llr = self.demapper(x_hat, no_eff)

        else:
            x_hat = None
            no_eff = None
            llr = None

        if self.config.coded_ber and llr is not None:
            b_hat = self.decoder(llr)
        else:
            b_hat = None

        return {
            "b": b,
            "c": c,
            "x": x,

            "x_rg": x_rg,
            "x_time": x_time,

            "y_time_clean": y_time_clean,
            "y_time_noisy": y_time_noisy,
            "y_rg": y_rg,

            "h_time_true": h_time,
            "h_freq_true": h_freq_true,

            "h_hat": h_hat,
            "err_var": err_var,

            "x_hat": x_hat,
            "llr": llr,
            "b_hat": b_hat,
            "no": no,
            "no_eff": no_eff,

            "a": a,
            "tau": tau,
        }