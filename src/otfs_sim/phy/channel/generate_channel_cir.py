# otfs_sim/phy/channel_generation.py

import tensorflow as tf

from otfs_sim.phy.channel.channel_utils import dd_paths_to_cir, approximate_dd_paths_from_cir
from sionna.phy.channel.tr38811 import Antenna, AntennaArray
from sionna.phy.channel.tr38811.utils import gen_single_sector_topology as gen_ntn_topology
from sionna.phy.channel.tr38811 import DenseUrban, Urban, SubUrban


"""
Channel generators for OTFS/OFDM simulations.

This module provides generators that produce Sionna-compatible CIR tensors:

    a :
        [batch, rx, rx_ant, tx, tx_ant, num_paths, num_time_steps]

    tau :
        [batch, rx, tx, num_paths]

Supported channel sources:
    - Hardcoded delay-Doppler path models
    - NTN channels generated with Sionna/OpenNTN
    - Integer-grid approximations of NTN channels
"""

class DDPathCIRGenerator:
    """
    Generate Sionna CIR tensors from hardcoded delay-Doppler paths.

    Each channel case defines:
        g_i :
            Complex path gains

        l_i :
            Integer delay taps

        k_i :
            Doppler taps

    Public interface:
        a, tau = generator(
            batch_size,
            num_time_steps,
            bandwidth,
        )

    Output:
        a :
            [batch, rx, rx_ant, tx, tx_ant,
             num_paths, num_time_steps]

        tau :
            [batch, rx, tx, num_paths]
    """

    def __init__(
        self,
        channel_case="single",
        g_i=None,
        l_i=None,
        k_i=None,
    ):
        self.channel_case = channel_case

        if g_i is None and l_i is None and k_i is None:
            self.g_i, self.l_i, self.k_i = self._make_paths(channel_case)
        elif g_i is not None and l_i is not None and k_i is not None:
            self.g_i = tf.constant(g_i, dtype=tf.complex64)
            self.l_i = tf.constant(l_i, dtype=tf.int32)
            self.k_i = tf.constant(k_i, dtype=tf.float32)
        else:
            raise ValueError(
                "Either provide all of g_i, l_i, k_i or none of them."
            )

    def _make_paths(self, channel_case):
        # ----------------------------------------------------------------
        # Case definitions are calibrated for:
        #   N=14 Doppler bins, M=624 delay bins
        #   SCS=240 kHz, BW=149.8 MHz (Ka-band NTN numerology)
        #   kmax=3 (pilot guard covers ±kmax Doppler bins)
        #   lmax=6 (pilot guard covers [0, lmax] delay bins)
        #
        # Doppler tap k_i convention:
        #   k_i is in units of OTFS Doppler bins.
        #   k_i = 1 corresponds to one full phase rotation over N*M
        #   time samples, i.e. a physical Doppler shift of Delta_f / N.
        #   At 240 kHz SCS and N=14: 1 bin = 240e3/14 ≈ 17.1 kHz,
        #   equivalent to v ≈ 270 m/s at 19 GHz carrier.
        #
        # Delay tap l_i convention:
        #   l_i is in units of samples at the system bandwidth.
        #   At BW=149.8 MHz: 1 tap ≈ 6.7 ns.
        #   l_i must satisfy l_i <= lmax and l_i < CP_LEN.
        # ----------------------------------------------------------------

        if channel_case == "single":
            # Single LOS path. No delay, no Doppler.
            # Purpose: AWGN reference. All receivers should converge to
            # the AWGN bound. Used to validate the simulation chain.
            g_i = tf.constant([1.0 + 0.0j], dtype=tf.complex64)
            l_i = tf.constant([0], dtype=tf.int32)
            k_i = tf.constant([0.0], dtype=tf.float32)

        elif channel_case in ("static_multipath", "two_path", "two_path_zero_doppler"):
            # Two paths, delay spread only, no Doppler.
            # Purpose: frequency-selective fading without time variation.
            # OFDM single-tap equalization works well here; OTFS shows no
            # advantage. Baseline for the delay-only case.
            # Note: two_path and two_path_zero_doppler are kept as aliases.
            g_i = tf.constant(
                [1.0 + 0.0j, 0.6 + 0.2j],
                dtype=tf.complex64,
            )
            l_i = tf.constant([0, 3], dtype=tf.int32)
            k_i = tf.constant([0.0, 0.0], dtype=tf.float32)

        elif channel_case in ("low_doppler", "two_path_mild_doppler"):
            # Two paths: LOS at k=0, delayed path at integer k=1.
            # Purpose: mild doubly dispersive channel with integer Doppler.
            # The embedded pilot estimator can resolve this case cleanly.
            # Corresponds to ~270 m/s relative velocity at 19 GHz / 240 kHz.
            # Note: two_path_mild_doppler is kept as an alias.
            g_i = tf.constant(
                [1.0 + 0.0j, 0.6 + 0.2j],
                dtype=tf.complex64,
            )
            l_i = tf.constant([0, 3], dtype=tf.int32)
            k_i = tf.constant([0.0, 2.0], dtype=tf.float32)

        elif channel_case in ("high_doppler", "two_path_strong_doppler"):
            # Two paths: LOS at k=0, delayed path at integer k=3 (= kmax).
            # Purpose: strong doubly dispersive channel, maximum resolvable
            # Doppler for kmax=3. OFDM suffers significant ICI; OTFS with
            # perfect CSI LMMSE recovers cleanly. Main comparison scenario.
            # Corresponds to ~810 m/s relative velocity at 19 GHz / 240 kHz.
            # Note: two_path_strong_doppler is kept as an alias.
            g_i = tf.constant(
                [1.0 + 0.0j, 0.6 + 0.2j],
                dtype=tf.complex64,
            )
            l_i = tf.constant([0, 3], dtype=tf.int32)
            k_i = tf.constant([0.0, 5.0], dtype=tf.float32)

        elif channel_case == "fractional_doppler":
            # Two paths: LOS at k=0, delayed path at fractional k=2.5.
            # Purpose: demonstrates the integer-tap estimator limitation.
            # The embedded pilot estimator produces a persistent NMSE floor
            # regardless of SNR because the path does not land on a grid
            # point. The gap between perfect-CSI and estimated-CSI LMMSE
            # is irreducible. Used in threshold sensitivity analysis.
            g_i = tf.constant(
                [1.0 + 0.0j, 0.6 + 0.2j],
                dtype=tf.complex64,
            )
            l_i = tf.constant([0, 3], dtype=tf.int32)
            k_i = tf.constant([0.0, 3.5], dtype=tf.float32)

        elif channel_case in ("ntn_like", "many_path"):
            # Four paths with near-degenerate Doppler clustering.
            # Purpose: synthetic proxy for NTN channel structure, where
            # satellite orbital velocity dominates and all paths share
            # approximately the same Doppler shift. Tests the estimator
            # under the clustered-Doppler condition that makes individual
            # path resolution difficult.
            # Delay taps within lmax=6; Doppler taps within kmax=3.
            # Note: many_path is kept as an alias (old case was broken:
            # l=[0,12,7,5] exceeded lmax and k=[6,0,3.5,2] exceeded kmax).
            g_i = tf.constant(
                [1.0 + 0.0j, 0.5 + 0.3j, 0.3 + 0.1j, 0.2 + 0.15j],
                dtype=tf.complex64,
            )
            l_i = tf.constant([0, 1, 3, 5], dtype=tf.int32)
            k_i = tf.constant([2.0, 2.0, 3.0, 2.0], dtype=tf.float32)

        else:
            raise ValueError(
                f"Unsupported channel_case='{channel_case}'. "
                "Supported cases: 'single', 'static_multipath', "
                "'low_doppler', 'high_doppler', 'fractional_doppler', "
                "'ntn_like'. Legacy aliases: 'two_path', "
                "'two_path_zero_doppler', 'two_path_mild_doppler', "
                "'two_path_strong_doppler', 'many_path'."
            )
            
        return g_i, l_i, k_i

    def __call__(self, batch_size, num_time_steps, bandwidth):
        return dd_paths_to_cir(
            num_time_steps=num_time_steps,
            bandwidth=bandwidth,
            g_i=self.g_i,
            l_i=self.l_i,
            k_i=self.k_i,
            batch_size=batch_size,
        )


# ============================================================
# Hardcoded NTN CIR generator
# ============================================================

class NTNHardcodedCIRGenerator:
    """
    Generate NTN CIR tensors using OpenNTN TR 38.811 models.

    The generator internally configures a fixed NTN scenario and exposes
    a simplified interface for experiments.

    Public interface:
        a, tau = generator(
            batch_size,
            num_time_steps,
            bandwidth,
        )

    Output:
        a :
            [batch, rx, rx_ant, tx, tx_ant,
             num_paths, num_time_steps]

        tau :
            [batch, rx, tx, num_paths]
    """

    def __init__(
        self,
        scenario="dur",
        carrier_frequency=19.0e9,
        direction="downlink",
        elevation_angle=80.0,
        satellite_height=2000000.0,
        num_ut=1,
        doppler_enabled=True,
    ):

        self.gen_ntn_topology = gen_ntn_topology

        self.scenario = scenario
        self.carrier_frequency = carrier_frequency
        self.direction = direction
        self.elevation_angle = elevation_angle
        self.satellite_height = satellite_height
        self.num_ut = num_ut
        self.doppler_enabled = doppler_enabled

        # User-terminal antenna
        ut_array = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency,
        )

        # Satellite/base-station antenna
        bs_array = AntennaArray(
            num_rows=1,
            num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency,
        )

        self.scenario = scenario.lower()

        if self.scenario == "dur":
            self.channel_model = DenseUrban(
                carrier_frequency=self.carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction=self.direction,
                elevation_angle=self.elevation_angle,
                doppler_enabled=self.doppler_enabled,
            )

        elif self.scenario == "urb":
            self.channel_model = Urban(
                carrier_frequency=self.carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction=self.direction,
                elevation_angle=self.elevation_angle,
                doppler_enabled=self.doppler_enabled,
            )

        elif self.scenario == "sur":
            self.channel_model = SubUrban(
                carrier_frequency=self.carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction=self.direction,
                elevation_angle=self.elevation_angle,
                doppler_enabled=self.doppler_enabled,
            )

        else:
            raise ValueError(
                f"Unsupported scenario '{scenario}'. "
                "Use 'dur', 'urb', or 'sur'."
            )

    def __call__(self, batch_size, num_time_steps, bandwidth):
        topology = self.gen_ntn_topology(
            # batch_size=batch_size,
            batch_size=1,
            num_ut=self.num_ut,
            scenario=self.scenario,
            bs_height=self.satellite_height,
            elevation_angle=self.elevation_angle,
        )

        self.channel_model.set_topology(*topology)

        a, tau = self.channel_model(
            num_time_samples=num_time_steps,
            sampling_frequency=bandwidth,
        )
        
        a = tf.tile(a, [batch_size, 1, 1, 1, 1, 1, 1])
        tau = tf.tile(tau, [batch_size, 1, 1, 1])

        return a, tau
    
class NTNIntegerDDCIRGenerator:
    """
    Generate an NTN channel and approximate it using integer-grid
    delay-Doppler taps.

    Processing:
        1. Generate an NTN CIR
        2. Estimate DD paths from the CIR
        3. Quantize delay and Doppler taps
        4. Rebuild a clean integer-grid CIR

    This is useful for studying the impact of integer DD-grid channel
    assumptions on realistic NTN channels.
    """

    def __init__(
        self,
        delay_rounding="round",
        doppler_rounding="round",
        **ntn_kwargs,
    ):
        self.ntn_generator = NTNHardcodedCIRGenerator(**ntn_kwargs)
        self.delay_rounding = delay_rounding
        self.doppler_rounding = doppler_rounding

    def _quantize(self, x, mode):
        if mode == "round":
            return tf.round(x)
        if mode == "ceil":
            return tf.math.ceil(x)
        if mode == "floor":
            return tf.math.floor(x)

        raise ValueError(
            f"Unsupported quantization mode='{mode}'. "
            "Use 'round', 'ceil', or 'floor'."
        )

    def __call__(self, batch_size, num_time_steps, bandwidth):
        # 1) Generate original OpenNTN/Sionna NTN CIR
        a_ntn, tau_ntn = self.ntn_generator(
            batch_size=batch_size,
            num_time_steps=num_time_steps,
            bandwidth=bandwidth,
        )

        # 2) Approximate DD paths from the first generated CIR
        g_i, l_i, k_i = approximate_dd_paths_from_cir(
            a=a_ntn,
            tau=tau_ntn,
            bandwidth=bandwidth,
            frame_num_time_samples=num_time_steps,
        )

        # 3) Force delay and Doppler to integer taps
        l_i = tf.cast(
            self._quantize(tf.cast(l_i, tf.float32), self.delay_rounding),
            tf.int32,
        )

        k_i = tf.cast(
            self._quantize(k_i, self.doppler_rounding),
            tf.float32,
        )

        # 4) Rebuild clean CIR from integer-grid DD paths
        a, tau = dd_paths_to_cir(
            num_time_steps=num_time_steps,
            bandwidth=bandwidth,
            g_i=g_i,
            l_i=l_i,
            k_i=k_i,
            batch_size=batch_size,
        )

        return a, tau