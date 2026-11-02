import tensorflow as tf
from sionna.phy import Block
from sionna.phy.signal import fft, ifft
from sionna.phy.utils import flatten_last_dims


class OFDMModulatorCustom(Block):
    """
    OFDM-style modulation.

    Input shape:
        [..., num_ofdm_symbols, fft_size]

    Output shape:
        [..., num_ofdm_symbols * (fft_size + cyclic_prefix_length)]
    """

    def __init__(self, cyclic_prefix_length=0):
        super().__init__()
        self.cyclic_prefix_length = cyclic_prefix_length

    def call(self, x_freq):
        x_freq = tf.signal.ifftshift(x_freq, axes=-1)
        x_time = ifft(x_freq, axis=-1)

        if self.cyclic_prefix_length > 0:
            cp = x_time[..., -self.cyclic_prefix_length:]
            x_time = tf.concat([cp, x_time], axis=-1)

        x_time = flatten_last_dims(x_time, 2)
        return x_time


class OFDMDemodulatorCustom(Block):
    """
    OFDM-style demodulation.

    Input shape:
        [..., num_samples]

    Output shape:
        [..., num_ofdm_symbols, fft_size]
    """

    def __init__(self, fft_size, cyclic_prefix_length=0):
        super().__init__()
        self.fft_size = fft_size
        self.cyclic_prefix_length = cyclic_prefix_length

    def call(self, y_time):
        symbol_length = self.fft_size + self.cyclic_prefix_length

        num_complete_symbols = tf.shape(y_time)[-1] // symbol_length
        num_used_samples = num_complete_symbols * symbol_length
        y_time = y_time[..., :num_used_samples]

        new_shape = tf.concat(
            [tf.shape(y_time)[:-1], [num_complete_symbols, symbol_length]],
            axis=0,
        )
        y_time = tf.reshape(y_time, new_shape)

        if self.cyclic_prefix_length > 0:
            y_time = y_time[..., self.cyclic_prefix_length:]

        y_freq = fft(y_time, axis=-1)
        y_freq = tf.signal.fftshift(y_freq, axes=-1)

        return y_freq