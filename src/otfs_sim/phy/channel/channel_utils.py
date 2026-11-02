"""Utilities for channel representation conversions.

This module contains stateless helpers for moving between the channel
representations used in the simulator:

- DD paths:     g_i, l_i, k_i
- Sionna CIR:   a, tau
- Sionna taps:  h_time
- textbook DT:  gs[delay, time]

No channel application block should live here. Stateful blocks such as
ManualDelayTimeChannel or SionnaCIRDelayTimeChannel should import these
helpers.
"""

from __future__ import annotations

import tensorflow as tf
from sionna.phy.channel import cir_to_time_channel


# ============================================================
# Basic sizing helpers
# ============================================================

def get_l_tot(l_min: int, l_max: int) -> int:
    """
    Return the number of discrete delay taps.

    Input:
        l_min : smallest delay tap
        l_max : largest delay tap

    Output:
        l_tot = l_max - l_min + 1
    """

    l_tot = int(l_max) - int(l_min) + 1

    if l_tot <= 0:
        raise ValueError(
            f"Invalid channel lags: l_min={l_min}, l_max={l_max}."
        )

    return l_tot


def get_required_num_cir_time_steps(
    num_time_samples: int,
    l_min: int,
    l_max: int,
) -> int:
    """Return the number of CIR snapshots required by ApplyTimeChannel.

    ApplyTimeChannel expects h_time with time dimension
    num_time_samples + l_tot - 1, where l_tot = l_max - l_min + 1.
    Since cir_to_time_channel preserves the CIR time dimension, this is also
    the required last dimension of the Sionna CIR coefficient tensor a.
    """

    l_tot = get_l_tot(l_min=l_min, l_max=l_max)
    return int(num_time_samples) + l_tot - 1


# ============================================================
# DD paths -> textbook delay-time channel
# ============================================================

def dd_paths_to_gs(
    num_time_samples: int,
    g_i,
    l_i,
    k_i,
):
    """Build the textbook discrete delay-time channel gs.

    The convention is

        gs[ell, q] = sum_i g_i z**(k_i * (q - l_i))

    for all paths with delay ell, where

        z = exp(j 2 pi / num_time_samples).

    Args:
        num_time_samples: Number of time samples in the frame.
        g_i: Complex path gains, shape [num_paths].
        l_i: Integer delay taps, shape [num_paths].
        k_i: Normalized Doppler taps, shape [num_paths].

    Returns:
        gs: Tensor with shape [max(l_i)+1, num_time_samples].
    """

    g_i = tf.cast(g_i, tf.complex64)
    l_i = tf.cast(l_i, tf.int32)
    k_i = tf.cast(k_i, tf.float32)

    delay_spread = int(tf.reduce_max(l_i).numpy())
    gs = tf.zeros([delay_spread + 1, num_time_samples], dtype=tf.complex64)

    q = tf.cast(tf.range(num_time_samples), tf.float32)

    z = tf.exp(
        tf.complex(
            tf.constant(0.0, dtype=tf.float32),
            2.0
            * tf.constant(3.141592653589793, dtype=tf.float32)
            / tf.cast(num_time_samples, tf.float32),
        )
    )

    for path_index in range(g_i.shape[0]):
        ell = int(l_i[path_index].numpy())
        k = tf.cast(k_i[path_index], tf.complex64)
        g = tf.cast(g_i[path_index], tf.complex64)

        exponent = k * tf.cast(q - tf.cast(ell, tf.float32), tf.complex64)
        values = g * tf.pow(z, exponent)

        indices = tf.stack(
            [
                tf.fill([num_time_samples], ell),
                tf.range(num_time_samples),
            ],
            axis=1,
        )

        gs = tf.tensor_scatter_nd_add(gs, indices, values)

    return gs


# ============================================================
# DD paths -> Sionna CIR
# ============================================================

def dd_paths_to_cir(
    num_time_steps: int,
    bandwidth,
    g_i,
    l_i,
    k_i,
    batch_size: int,
    num_rx: int = 1,
    num_rx_ant: int = 1,
    num_tx: int = 1,
    num_tx_ant: int = 1,
):
    """Convert DD-style paths into Sionna CIR tensors a,tau.

    Args:
        num_time_steps: Number of CIR time snapshots, i.e., the last axis of a.
            For direct use with ApplyTimeChannel, this should usually be
            num_time_samples + l_tot - 1.
        bandwidth: Sampling frequency / bandwidth W in Hz.
        g_i: Complex path gains, shape [num_paths].
        l_i: Delay taps in samples, shape [num_paths].
        k_i: Normalized Doppler taps, shape [num_paths].
        batch_size: Batch size.
        num_rx: Number of receivers.
        num_rx_ant: Number of receive antennas per receiver.
        num_tx: Number of transmitters.
        num_tx_ant: Number of transmit antennas per transmitter.

    Returns:
        a: Complex CIR coefficients with shape
           [batch, num_rx, num_rx_ant, num_tx, num_tx_ant,
            num_paths, num_time_steps].
        tau: Path delays in seconds with shape
             [batch, num_rx, num_tx, num_paths].
    """

    g_i = tf.cast(g_i, tf.complex64)
    l_i = tf.cast(l_i, tf.float32)
    k_i = tf.cast(k_i, tf.float32)

    num_paths = int(g_i.shape[0])
    q = tf.cast(tf.range(num_time_steps), tf.float32)

    z = tf.exp(
        tf.complex(
            tf.constant(0.0, dtype=tf.float32),
            2.0
            * tf.constant(3.141592653589793, dtype=tf.float32)
            / tf.cast(num_time_steps, tf.float32),
        )
    )

    a_paths = []

    for path_index in range(num_paths):
        g = g_i[path_index]
        ell = l_i[path_index]
        k = tf.cast(k_i[path_index], tf.complex64)

        # Textbook convention:
        # a_i[q] = g_i * z**(k_i * (q - l_i))
        exponent = k * tf.cast(q - ell, tf.complex64)
        a_i = g * tf.pow(z, exponent)

        a_paths.append(a_i)

    # [num_paths, num_time_steps]
    a = tf.stack(a_paths, axis=0)

    # [1, 1, 1, 1, 1, num_paths, num_time_steps]
    a = tf.reshape(a, [1, 1, 1, 1, 1, num_paths, num_time_steps])

    # [batch, rx, rx_ant, tx, tx_ant, num_paths, num_time_steps]
    a = tf.tile(a, [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, 1, 1])

    # tau_i = delay_samples / bandwidth
    tau_paths = l_i / tf.cast(bandwidth, tf.float32)

    # [1, 1, 1, num_paths]
    tau = tf.reshape(tau_paths, [1, 1, 1, num_paths])

    # [batch, rx, tx, num_paths]
    tau = tf.tile(tau, [batch_size, num_rx, num_tx, 1])

    return a, tau


# ============================================================
# CIR / h_time / gs conversions
# ============================================================

def infer_l_max_from_tau(
    tau,
    bandwidth,
    l_min: int = 0,
    margin: int = 0,
) -> int:
    """Infer l_max from physical path delays tau.

    Args:
        tau: Sionna path delays in seconds.
        bandwidth: Sampling frequency / bandwidth W in Hz.
        l_min: Smallest allowed discrete time lag.
        margin: Additional integer delay margin.

    Returns:
        l_max: Largest discrete time lag.
    """

    delay_samples = tau * tf.cast(bandwidth, tf.float32)
    max_delay = tf.reduce_max(delay_samples)

    l_max = int(tf.math.ceil(max_delay).numpy()) + int(margin)

    if l_max < l_min:
        raise ValueError(
            f"Invalid inferred l_max={l_max} for l_min={l_min}."
        )

    return l_max


def cir_to_h_time(
    a,
    tau,
    bandwidth,
    l_min: int = 0,
    l_max: int | None = None,
    l_margin: int = 0,
    normalize: bool = True,
):
    """Convert Sionna CIR tensors a,tau into Sionna h_time taps.

    Args:
        a: CIR coefficients with shape
           [batch, rx, rx_ant, tx, tx_ant, paths, time].
        tau: CIR delays with shape [batch, rx, tx, paths] or full-rank delay
            tensor accepted by Sionna.
        bandwidth: Sampling frequency / bandwidth W in Hz.
        l_min: Smallest discrete time lag.
        l_max: Largest discrete time lag. If None, inferred from tau.
        l_margin: Additional margin used only when l_max is None.
        normalize: Passed to sionna.phy.channel.cir_to_time_channel.

    Returns:
        h_time: Sionna time-domain channel taps with shape
            [batch, rx, rx_ant, tx, tx_ant, time, l_tot].
        l_min: Used l_min.
        l_max: Used l_max.
    """

    if l_max is None:
        l_max = infer_l_max_from_tau(
            tau=tau,
            bandwidth=bandwidth,
            l_min=l_min,
            margin=l_margin,
        )

    h_time = cir_to_time_channel(
        bandwidth=bandwidth,
        a=a,
        tau=tau,
        l_min=l_min,
        l_max=l_max,
        normalize=normalize,
    )

    return h_time, l_min, l_max


def dd_paths_to_h_time(
    g_i,
    l_i,
    k_i,
    bandwidth,
    num_time_samples: int,
    batch_size: int,
    l_min: int = 0,
    l_max: int | None = None,
    l_margin: int = 0,
    normalize: bool = True,
    num_rx: int = 1,
    num_rx_ant: int = 1,
    num_tx: int = 1,
    num_tx_ant: int = 1,
):
    """Convert DD paths into Sionna h_time via a,tau.

    This combines

        DD paths -> Sionna CIR a,tau -> Sionna h_time.

    The CIR time dimension is chosen to be compatible with ApplyTimeChannel:

        num_time_steps = num_time_samples + l_tot - 1.

    Returns:
        h_time: Sionna time-domain channel taps.
        a: Sionna CIR coefficients.
        tau: Sionna CIR delays.
        l_min: Used l_min.
        l_max: Used l_max.
    """

    if l_max is None:
        l_max = int(tf.reduce_max(tf.cast(l_i, tf.int32)).numpy())
        l_max += int(l_margin)

    num_time_steps = get_required_num_cir_time_steps(
        num_time_samples=num_time_samples,
        l_min=l_min,
        l_max=l_max,
    )

    a, tau = dd_paths_to_cir(
        num_time_steps=num_time_steps,
        bandwidth=bandwidth,
        g_i=g_i,
        l_i=l_i,
        k_i=k_i,
        batch_size=batch_size,
        num_rx=num_rx,
        num_rx_ant=num_rx_ant,
        num_tx=num_tx,
        num_tx_ant=num_tx_ant,
    )

    h_time, l_min, l_max = cir_to_h_time(
        a=a,
        tau=tau,
        bandwidth=bandwidth,
        l_min=l_min,
        l_max=l_max,
        l_margin=l_margin,
        normalize=normalize,
    )

    return h_time, a, tau, l_min, l_max


def h_time_to_gs(
    h_time,
    num_time_samples: int,
    batch_index: int = 0,
    rx_index: int = 0,
    rx_ant_index: int = 0,
    tx_index: int = 0,
    tx_ant_index: int = 0,
):
    """Convert Sionna h_time to gs for the custom time-domain LMMSE.

    Args:
        h_time: Tensor with shape
            [batch, rx, rx_ant, tx, tx_ant, time, delay].
        num_time_samples: Number of time samples to keep in gs.
        batch_index: Batch example to extract.
        rx_index: Receiver index to extract.
        rx_ant_index: Receive antenna index to extract.
        tx_index: Transmitter index to extract.
        tx_ant_index: Transmit antenna index to extract.

    Returns:
        gs: Tensor with shape [delay, num_time_samples].
    """

    h_selected = h_time[
        batch_index,
        rx_index,
        rx_ant_index,
        tx_index,
        tx_ant_index,
        :num_time_samples,
        :,
    ]

    return tf.transpose(h_selected, perm=[1, 0])


def dd_paths_to_gs_via_h_time(
    g_i,
    l_i,
    k_i,
    bandwidth,
    num_time_samples: int,
    batch_size: int,
    l_min: int = 0,
    l_max: int | None = None,
    l_margin: int = 0,
    normalize: bool = True,
):
    """Build gs from DD paths through Sionna CIR and h_time.

    This is different from dd_paths_to_gs(), which uses the direct textbook
    formula. This function uses Sionna's sinc discretization first.

    Returns:
        gs: Tensor with shape [delay, num_time_samples].
    """

    h_time, _, _, _, _ = dd_paths_to_h_time(
        g_i=g_i,
        l_i=l_i,
        k_i=k_i,
        bandwidth=bandwidth,
        num_time_samples=num_time_samples,
        batch_size=batch_size,
        l_min=l_min,
        l_max=l_max,
        l_margin=l_margin,
        normalize=normalize,
    )

    return h_time_to_gs(
        h_time=h_time,
        num_time_samples=num_time_samples,
    )


def cir_to_gs(
    a,
    tau,
    bandwidth,
    num_time_samples: int,
    l_min: int = 0,
    l_max: int | None = None,
    l_margin: int = 0,
    normalize: bool = True,
):
    """Build gs from Sionna CIR tensors through h_time.

    Returns:
        gs: Tensor with shape [delay, num_time_samples].
    """

    h_time, _, _ = cir_to_h_time(
        a=a,
        tau=tau,
        bandwidth=bandwidth,
        l_min=l_min,
        l_max=l_max,
        l_margin=l_margin,
        normalize=normalize,
    )

    return h_time_to_gs(
        h_time=h_time,
        num_time_samples=num_time_samples,
    )


# ============================================================
# Debugging approximations
# ============================================================

def approximate_dd_paths_from_cir(
    a,
    tau,
    bandwidth,
    frame_num_time_samples: int,
):
    """Approximate DD paths g_i,l_i,k_i from Sionna CIR tensors.

    This is mainly for debugging and plotting. It is exact only for simple
    sparse paths with approximately linear phase evolution over time.

    Args:
        a: Sionna CIR coefficients.
        tau: Sionna CIR delays.
        bandwidth: Sampling frequency / bandwidth W in Hz.
        frame_num_time_samples: Normalization length used to map phase
            increment to normalized Doppler taps.

    Returns:
        g_i: Approximate complex path gains, shape [num_paths].
        l_i: Approximate integer delay taps, shape [num_paths].
        k_i: Approximate normalized Doppler taps, shape [num_paths].
    """

    # SISO, first batch:
    # a_path_time: [num_paths, num_time_steps]
    a_path_time = a[0, 0, 0, 0, 0, :, :]

    # tau_paths: [num_paths]
    tau_paths = tau[0, 0, 0, :]

    g_i = a_path_time[:, 0]

    delay_samples = tau_paths * tf.cast(bandwidth, tf.float32)
    l_i = tf.cast(tf.round(delay_samples), tf.int32)

    # Estimate normalized Doppler tap from average phase increment.
    phase_ratio = a_path_time[:, 1:] * tf.math.conj(a_path_time[:, :-1])
    phase_increment = tf.math.angle(tf.reduce_mean(phase_ratio, axis=-1))

    k_i = (
        phase_increment
        * tf.cast(frame_num_time_samples, tf.float32)
        / (2.0 * tf.constant(3.141592653589793, dtype=tf.float32))
    )

    return g_i, l_i, k_i
