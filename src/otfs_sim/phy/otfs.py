from sionna.phy import Block
from sionna.phy.signal import fft, ifft

"""
OTFS delay-Doppler <-> time-frequency transforms.

Conventions:
    - Shape: [..., num_ofdm_symbols, fft_size]
    - Last axis: delay/subcarriers
    - Second-to-last axis: Doppler/OFDM symbols

Notes:
    - OTFS modulation:
          FFT(delay) + IFFT(Doppler)
    - OTFS demodulation:
          FFT(Doppler) + IFFT(delay)
"""

class OTFSModulator(Block):
    """
    OTFS modulator.

    Converts delay-Doppler symbols to the time-frequency domain
    using the ISFFT.

    Input/Output:
        [..., num_ofdm_symbols, fft_size]
    """

    def __init__(self):
        super().__init__()

    def call(self, x_dd):
        x_tf = fft(x_dd, axis=-1)
        x_tf = ifft(x_tf, axis=-2)
        return x_tf


class OTFSDemodulator(Block):
    """
    OTFS demodulator.

    Converts time-frequency symbols to the delay-Doppler domain
    using the SFFT.

    Input/Output:
        [..., num_ofdm_symbols, fft_size]
    """

    def __init__(self):
        super().__init__()

    def call(self, x_tf):
        x_dd = fft(x_tf, axis=-2)
        x_dd = ifft(x_dd, axis=-1)
        return x_dd