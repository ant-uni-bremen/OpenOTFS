import numpy as np
import tensorflow as tf

from sionna.phy import Block
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, ResourceGridDemapper, OFDMModulator, OFDMDemodulator
from sionna.phy.mimo import StreamManagement
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no

from otfs_sim.phy.otfs import OTFSModulator, OTFSDemodulator
from otfs_sim.phy.ofdm import OFDMModulatorCustom, OFDMDemodulatorCustom
from otfs_sim.phy.equalizers import BlockLMMSEEqualizer
from otfs_sim.phy.pilots import EmbeddedOTFSPilotPattern
from otfs_sim.phy.estimators import EmbeddedPilotChannelEstimator

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


class OTFSSimulationModel(Block):
    """
    End-to-end SISO OTFS simulation model.

    This block implements the full transmit-channel-receive chain used for
    BER simulations. It is compatible with Sionna's sim_ber() interface.

    Main pipeline:
        bits
        -> LDPC encoder
        -> QAM mapper
        -> Sionna ResourceGrid mapper
        -> OTFS modulation: DD -> TF
        -> OFDM modulation: TF -> serialized time waveform with CP
        -> time-domain channel
        -> AWGN
        -> optional block-wise LMMSE equalization
        -> OFDM demodulation
        -> OTFS demodulation: TF -> DD
        -> ResourceGrid demapper
        -> QAM demapper
        -> optional LDPC decoder

    System assumptions:
        - Single-user SISO:
              num_tx = 1
              num_rx = 1
              num_streams_per_tx = 1

        - OTFS grid convention:
              num_ofdm_symbols -> Doppler bins
              fft_size         -> delay bins / subcarriers

        - Sionna tensor convention:
              resource grids have shape
              [batch, num_tx, num_streams_per_tx,
               num_ofdm_symbols, fft_size]

        - Time-domain waveforms have shape
              [batch, num_tx, num_tx_ant, num_time_samples]

    Supported CSI modes:
        perfect:
            Uses the true h_time from the applied channel and converts it
            to gs for the custom block LMMSE equalizer.

        estimated:
            Uses the embedded OTFS pilot estimator to estimate DD paths,
            converts the estimated paths to gs, and uses that gs for LMMSE.

    Supported channel backends:
        dd:
            Uses hardcoded delay-Doppler path cases.

        cir:
            Uses Sionna/OpenNTN-style CIR generation, either directly
            or after integer DD-grid quantization.

    Output behavior:
        If coded_ber is True:
            call() returns (b, b_hat), where b is the transmitted
            information bit tensor and b_hat is the decoded estimate.

        If coded_ber is False:
            call() returns (llr, c), where c is the encoded bit tensor and
            llr is the demapper output used by sim_ber().
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        if self.config.waveform != "otfs":
            raise ValueError(
                f"OTFSSimulationModel requires waveform='otfs', "
                f"but got waveform='{self.config.waveform}'."
            )

        if self.config.csi_mode not in ["perfect", "estimated"]:
            raise ValueError(
                f"Unsupported csi_mode='{self.config.csi_mode}'. "
                "Use 'perfect' or 'estimated'."
            )

        if self.config.csi_mode == "estimated" and not self.config.use_pilots:
            raise ValueError(
                "csi_mode='estimated' requires use_pilots=True."
            )

        if self.config.csi_mode == "estimated" and not self.config.use_lmmse:
            raise ValueError(
                "csi_mode='estimated' only affects LMMSE equalization. "
                "Set use_lmmse=True or use csi_mode='perfect'."
            )

        if self.config.lmax < 0 or self.config.kmax < 0:
            raise ValueError("lmax and kmax must be non-negative.")

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
        # OTFS pilot pattern / resource grid
        # ------------------------------------------------------------
        if self.config.use_pilots:
            self.pilot_pattern = EmbeddedOTFSPilotPattern(
                num_ofdm_symbols=self.config.num_ofdm_symbols,
                fft_size=self.config.fft_size,
                subcarrier_spacing=self.config.subcarrier_spacing,
                num_tx=self.num_tx,
                num_streams_per_tx=self.num_streams_per_tx,
                lmax=self.config.lmax,
                kmax=self.config.kmax,
                pilot_symbol=self.config.pilot_symbol,
            )

            self.pilot_info = self.pilot_pattern.pilot_info
            pilot_pattern_for_rg = self.pilot_pattern
            pilot_ofdm_symbol_indices = None

        else:
            self.pilot_pattern = None
            self.pilot_info = None
            pilot_pattern_for_rg = "kronecker"
            pilot_ofdm_symbol_indices = [2]

        self.rg = ResourceGrid(
            num_ofdm_symbols=self.config.num_ofdm_symbols,
            fft_size=self.config.fft_size,
            subcarrier_spacing=self.config.subcarrier_spacing,
            num_tx=self.num_tx,
            num_streams_per_tx=self.num_streams_per_tx,
            cyclic_prefix_length=self.config.cyclic_prefix_length,
            pilot_pattern=pilot_pattern_for_rg,
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
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
        # self.demapper = Demapper("app", "qam", self.config.num_bits_per_symbol)
        self.demapper = Demapper("app", "qam", self.config.num_bits_per_symbol, hard_out=not self.config.coded_ber,
        )

        self.rg_mapper = ResourceGridMapper(self.rg)
        self.rg_demapper = ResourceGridDemapper(self.rg, self.sm)
        
        self.ofdm_modulator = OFDMModulator(
            cyclic_prefix_length=self.config.cyclic_prefix_length,
        )

        self.ofdm_demodulator_with_cp = OFDMDemodulator(
            fft_size=self.config.fft_size,
            l_min=self.config.channel_l_min,
            cyclic_prefix_length=self.config.cyclic_prefix_length,
        )

        self.ofdm_demodulator_no_cp = OFDMDemodulator(
            fft_size=self.config.fft_size,
            l_min=self.config.channel_l_min,
            cyclic_prefix_length=0,
        )

        # ------------------------------------------------------------
        # OTFS / OFDM waveform blocks
        # ------------------------------------------------------------
        self.otfs_modulator = OTFSModulator()
        self.otfs_demodulator = OTFSDemodulator()


        # ------------------------------------------------------------
        # Channel / noise / equalization
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

        self.block_lmmse = BlockLMMSEEqualizer(
            num_blocks=self.config.num_ofdm_symbols,
            fft_size=self.config.fft_size,
            cyclic_prefix_length=self.config.cyclic_prefix_length,
        )

        if self.config.use_pilots:
            self.channel_estimator = EmbeddedPilotChannelEstimator(
                num_ofdm_symbols=self.config.num_ofdm_symbols,
                fft_size=self.config.fft_size,
                pilot_info=self.pilot_info,
                detection_mode=self.config.estimator_detection_mode,
                threshold_factor=self.config.estimation_threshold_factor,
                fp_min_persistence=self.config.estimator_fp_min_persistence,
            )
        else:
            self.channel_estimator = None

    def call(self, batch_size, ebno_db):
        """
        Run one OTFS transmission batch.

        Input:
            batch_size :
                Number of frames simulated in parallel.

            ebno_db :
                Eb/N0 value in dB.

        Important tensor shapes:
            b :
                [batch_size, num_tx, num_streams_per_tx, k]

                Information bits.

            c :
                [batch_size, num_tx, num_streams_per_tx, n]

                Encoded bits.

            x :
                [batch_size, num_tx, num_streams_per_tx, num_data_symbols]

                QAM data symbols before resource-grid mapping.

            x_dd :
                [batch_size, num_tx, num_streams_per_tx,
                num_ofdm_symbols, fft_size]

                OTFS delay-Doppler grid.

            x_tf :
                [batch_size, num_tx, num_streams_per_tx,
                num_ofdm_symbols, fft_size]

                Time-frequency grid after OTFS modulation.

            x_time :
                [batch_size, num_tx, num_tx_ant, num_time_samples]

                Serialized time-domain waveform with CP.

            a :
                [batch_size, rx, rx_ant, tx, tx_ant,
                num_paths, num_time_steps]

                Sionna CIR path coefficients.

            tau :
                [batch_size, rx, tx, num_paths]

                Sionna CIR path delays in seconds.

            h_time :
                [batch_size, rx, rx_ant, tx, tx_ant, time, delay]

                Discrete time-domain channel taps used by ApplyTimeChannel.

            gs_for_lmmse :
                [delay, num_time_samples]

                Delay-time channel response used by BlockLMMSEEqualizer.

            y_dd :
                [batch_size, num_rx, num_streams_per_tx,
                num_ofdm_symbols, fft_size]

                Received delay-Doppler grid after demodulation.

        Processing:
            1. Generate, encode, and map bits to QAM symbols.
            2. Map QAM symbols to the OTFS delay-Doppler grid.
            3. Apply OTFS and OFDM modulation.
            4. Generate and apply a time-varying channel.
            5. Add AWGN according to ebno_db.
            6. Optionally build gs and apply block-wise LMMSE equalization.
            7. Demodulate back to the delay-Doppler grid.
            8. Demap and optionally decode.

        Output:
            If coded_ber is True:
                b :
                    Transmitted information bits.

                b_hat :
                    Decoded information-bit estimates.

            If coded_ber is False:
                llr :
                    Soft bit estimates from the demapper.

                c :
                    Encoded transmitted bits.
        """
        # ------------------------------------------------------------
        # Transmitter
        # ------------------------------------------------------------
        b = self.binary_source([batch_size, self.num_tx, self.num_streams_per_tx, self.k])

        c = self.encoder(b)
        x = self.mapper(c)

        # In OTFS this is the delay-Doppler grid
        x_dd = self.rg_mapper(x)

        # Delay-Doppler -> time-frequency
        x_tf = self.otfs_modulator(x_dd)

        # Time-frequency -> serialized time-domain waveform with CP
        x_time = self.ofdm_modulator(x_tf)

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
        
        y_time, h_time = self.channel(x_time, a, tau)

        no = ebnodb2no(
            ebno_db,
            self.config.num_bits_per_symbol,
            self.config.coderate,
            self.rg,
        )

        y_time_noisy = self.awgn(y_time, no)

        # ------------------------------------------------------------
        # Optional LMMSE equalization
        # ------------------------------------------------------------
        if self.config.use_lmmse:

            if self.config.csi_mode == "perfect":
                gs_for_lmmse = h_time_to_gs(h_time, self.rg.num_time_samples)

            elif self.config.csi_mode == "estimated":
                if self.channel_estimator is None:
                    raise ValueError(
                        "csi_mode='estimated' requires use_pilots=True "
                        "and a valid channel_estimator."
                    )

                # Temporary demodulation without equalization for CE
                y_tf_ce = self.ofdm_demodulator_with_cp(y_time_noisy)
                y_dd_ce = self.otfs_demodulator(y_tf_ce)

                g_hat, l_hat, k_hat, _ = self.channel_estimator(y_dd_ce, no)

                gs_for_lmmse = dd_paths_to_gs_via_h_time(
                    g_i=g_hat,
                    l_i=l_hat,
                    k_i=tf.cast(k_hat, tf.float32),
                    bandwidth=self.rg.bandwidth,
                    num_time_samples=self.rg.num_time_samples,
                    batch_size=batch_size,
                )

            else:
                raise ValueError(
                    f"Unsupported csi_mode='{self.config.csi_mode}'. "
                    "Use 'perfect' or 'estimated'."
                )

            y_time_for_demod = self.block_lmmse(
                y_time_noisy,
                gs_for_lmmse,
                no,
            )

            ofdm_demodulator = self.ofdm_demodulator_no_cp

        else:
            y_time_for_demod = y_time_noisy
            ofdm_demodulator = self.ofdm_demodulator_with_cp

        # ------------------------------------------------------------
        # Receiver
        # ------------------------------------------------------------
        y_tf = ofdm_demodulator(y_time_for_demod)
        y_dd = self.otfs_demodulator(y_tf)

        x_hat = self.rg_demapper(y_dd)

        llr = self.demapper(x_hat, no)

        if self.config.coded_ber:
            b_hat = self.decoder(llr)
            return b, b_hat

        return llr, c

    def run_frame_debug(self, batch_size, ebno_db):
        """
        Run one OTFS batch and return internal tensors for debugging.

        This method follows the same transmit-channel-receive chain as call(),
        but exposes intermediate tensors for inspection, plotting, and sanity
        checks. It should not be used directly with sim_ber().

        Input:
            batch_size :
                Number of frames simulated in parallel.

            ebno_db :
                Eb/N0 value in dB.

        Returned dictionary keys:
            b :
                [batch_size, num_tx, num_streams_per_tx, k]
                Transmitted information bits.

            c :
                [batch_size, num_tx, num_streams_per_tx, n]
                Encoded bits.

            x :
                [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
                QAM data symbols.

            x_dd :
                [batch_size, num_tx, num_streams_per_tx,
                num_ofdm_symbols, fft_size]
                Transmitted delay-Doppler grid.

            x_tf :
                [batch_size, num_tx, num_streams_per_tx,
                num_ofdm_symbols, fft_size]
                Transmitted time-frequency grid.

            x_time :
                [batch_size, num_tx, num_tx_ant, num_time_samples]
                Transmitted serialized time-domain waveform.

            y_time :
                [batch_size, num_rx, num_rx_ant,
                num_time_samples + l_tot - 1]
                Channel output before AWGN.

            y_time_noisy :
                Same shape as y_time.
                Channel output after AWGN.

            y_tf_ce :
                Time-frequency grid used for channel-estimation view before
                equalization.

            y_dd_ce :
                Delay-Doppler grid used for embedded pilot channel estimation.

            g_hat :
                [num_detected_paths] or None.
                Estimated complex path gains.

            l_hat :
                [num_detected_paths] or None.
                Estimated integer delay taps.

            k_hat :
                [num_detected_paths] or None.
                Estimated Doppler taps.

            h_dd_hat :
                [num_ofdm_symbols, fft_size] or None.
                Sparse DD-domain channel estimate from the pilot estimator.

            gs_true :
                [delay, num_time_samples]
                True delay-time response extracted from h_time.

            gs_est :
                Currently unused placeholder.

            gs_for_lmmse :
                [delay, num_time_samples]
                Channel response actually used by the LMMSE equalizer.

            a :
                [batch_size, rx, rx_ant, tx, tx_ant,
                num_paths, num_time_steps]
                CIR path coefficients.

            tau :
                [batch_size, rx, tx, num_paths]
                CIR path delays.

            bandwidth :
                Resource-grid bandwidth in Hz.

            y_time_for_demod :
                Time-domain signal passed to the OFDM demodulator.
                If LMMSE is enabled, this is equalized and CP-free.
                Otherwise, this is the noisy channel output with CP.

            y_tf :
                Time-frequency grid after final OFDM demodulation.

            y_dd :
                Delay-Doppler grid after final OTFS demodulation.

            x_hat :
                Demapped received QAM symbols.

            llr :
                Soft bit estimates from the demapper.

            b_hat :
                Decoded information bits if coded_ber is True, otherwise None.

            no :
                Noise variance used by AWGN, demapper, estimator, and LMMSE.

        Notes:
            - The debug channel-estimation view is computed before equalization.
            - For perfect CSI, gs_for_lmmse is extracted from the true h_time.
            - For estimated CSI, gs_for_lmmse is reconstructed from detected
            embedded-pilot DD paths.
        """

        # ------------------------------------------------------------
        # Transmitter
        # ------------------------------------------------------------
        b = self.binary_source([batch_size, self.num_tx, self.num_streams_per_tx, self.k])

        c = self.encoder(b)
        x = self.mapper(c)

        x_dd = self.rg_mapper(x)
        x_tf = self.otfs_modulator(x_dd)
        x_time = self.ofdm_modulator(x_tf)

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
        
        y_time, h_time = self.channel(x_time, a, tau)

        no = ebnodb2no(
            ebno_db,
            self.config.num_bits_per_symbol,
            self.config.coderate,
            self.rg,
        )

        y_time_noisy = self.awgn(y_time, no)

        # ------------------------------------------------------------
        # Debug CE view before equalization
        # ------------------------------------------------------------
        y_tf_ce = self.ofdm_demodulator_with_cp(y_time_noisy)
        y_dd_ce = self.otfs_demodulator(y_tf_ce)

        g_hat = None
        l_hat = None
        k_hat = None
        h_dd_hat = None
        gs_est = None
        gs_true = h_time_to_gs(h_time, self.rg.num_time_samples)

        if self.config.csi_mode == "estimated":
            if self.channel_estimator is None:
                raise ValueError(
                    "Debug estimated CSI requires use_pilots=True "
                    "and a valid channel_estimator."
                )

            g_hat, l_hat, k_hat, h_dd_hat = self.channel_estimator(y_dd_ce, no)

            gs_for_lmmse = dd_paths_to_gs_via_h_time(
                    g_i=g_hat,
                    l_i=l_hat,
                    k_i=tf.cast(k_hat, tf.float32),
                    bandwidth=self.rg.bandwidth,
                    num_time_samples=self.rg.num_time_samples,
                    batch_size=batch_size,
                )


        elif self.config.csi_mode == "perfect":
            gs_for_lmmse = h_time_to_gs(h_time, self.rg.num_time_samples)

        else:
            raise ValueError(
                f"Unsupported csi_mode='{self.config.csi_mode}'. "
                "Use 'perfect' or 'estimated'."
            )

        # ------------------------------------------------------------
        # Optional equalization
        # ------------------------------------------------------------
        if self.config.use_lmmse:
            y_time_for_demod = self.block_lmmse(
                y_time_noisy,
                gs_for_lmmse,
                no,
            )
            ofdm_demodulator = self.ofdm_demodulator_no_cp
        else:
            y_time_for_demod = y_time_noisy
            ofdm_demodulator = self.ofdm_demodulator_with_cp

        # ------------------------------------------------------------
        # Receiver
        # ------------------------------------------------------------
        y_tf = ofdm_demodulator(y_time_for_demod)
        y_dd = self.otfs_demodulator(y_tf)

        x_hat = self.rg_demapper(y_dd)
        llr = self.demapper(x_hat, no)

        if self.config.coded_ber:
            b_hat = self.decoder(llr)
        else:
            b_hat = None

        return {
            "b": b,
            "c": c,
            "x": x,

            "x_dd": x_dd,
            "x_tf": x_tf,
            "x_time": x_time,

            "y_time": y_time,
            "y_time_noisy": y_time_noisy,

            "y_tf_ce": y_tf_ce,
            "y_dd_ce": y_dd_ce,

            "g_hat": g_hat,
            "l_hat": l_hat,
            "k_hat": k_hat,
            "h_dd_hat": h_dd_hat,

            "gs_true": gs_true,
            "gs_est": gs_est,
            "gs_for_lmmse": gs_for_lmmse,
            
            "a": a,
            "tau": tau,
            "bandwidth": self.rg.bandwidth,

            "y_time_for_demod": y_time_for_demod,
            "y_tf": y_tf,
            "y_dd": y_dd,

            "x_hat": x_hat,
            "llr": llr,
            "b_hat": b_hat,
            "no": no,
        }