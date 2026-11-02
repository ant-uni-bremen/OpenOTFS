import tensorflow as tf
from sionna.phy import Block
from sionna.phy.utils import flatten_last_dims
from sionna.phy.mimo.equalization import lmmse_matrix
from sionna.phy.utils import flatten_last_dims


class BlockLMMSEEqualizer(Block):
    """
    Block-wise time-domain LMMSE equalizer for CP-OTFS and CP-OFDM.

    Conventions:
        Time-domain blocks are processed independently after CP removal.

    Input:
        y_time_serial :
            [..., num_blocks * (fft_size + cyclic_prefix_length)]

            Serialized received waveform including CP.

        gs :
            [delay_spread + 1, num_time_samples]

            Delay-time channel response where:
                axis 0 -> delay tap index
                axis 1 -> time sample index

        no :
            Scalar noise variance.

    Processing:
        1. Split serialized waveform into blocks
        2. Remove CP from each block
        3. Build one channel matrix per block
        4. Apply block-wise LMMSE equalization

    Internal tensor shapes:
        y_blocks :
            [..., num_blocks, fft_size]

        G_blocks :
            [num_blocks, fft_size, fft_size]

    Output:
        x_time_serial_hat :
            [..., num_blocks * fft_size]

            Equalized serialized waveform without CP.

    Notes:
        - Uses one LMMSE filter per block.
        - Channel matrices are constructed from the delay-time response gs.
        - Uses Sionna's lmmse_matrix() implementation internally.
    """

    def __init__(self, num_blocks, fft_size, cyclic_prefix_length):
        super().__init__()
        self.num_blocks = num_blocks
        self.fft_size = fft_size
        self.cyclic_prefix_length = cyclic_prefix_length

    def remove_cp_blocks(self, x_time_serial):
        """
        Split a serialized waveform into blocks and remove the CP.

        Input:
            x_time_serial :
                [..., num_blocks * (fft_size + cyclic_prefix_length)]

        Output:
            x_blocks :
                [..., num_blocks, fft_size]

        Notes:
            - Truncates incomplete trailing samples.
            - Keeps only the expected number of blocks.
        """
        
        block_len = self.fft_size + self.cyclic_prefix_length

        rest = tf.math.mod(tf.shape(x_time_serial)[-1], block_len)

        x_time_serial = tf.cond(
            tf.equal(rest, 0),
            lambda: x_time_serial,
            lambda: x_time_serial[..., :-rest],
        )

        # Keep only the expected OTFS frame blocks
        expected_len = self.num_blocks * block_len
        x_time_serial = x_time_serial[..., :expected_len]

        new_shape = tf.concat(
            [
                tf.shape(x_time_serial)[:-1],
                [self.num_blocks],
                [block_len],
            ],
            axis=0,
        )

        x_blocks_cp = tf.reshape(x_time_serial, new_shape)

        if self.cyclic_prefix_length > 0:
            x_blocks = x_blocks_cp[..., self.cyclic_prefix_length:]
        else:
            x_blocks = x_blocks_cp

        return x_blocks

    def build_block_channel_matrices(self, gs):
        """
        Construct one circulant channel matrix per time block.

        Input:
            gs :
                [delay_spread + 1, num_time_samples]

                Delay-time channel response.

        Output:
            G_blocks :
                [num_blocks, fft_size, fft_size]

                One channel matrix per block.
        """
    
        dtype = gs.dtype
        delay_spread = int(gs.shape[0]) - 1
        block_len = self.fft_size + self.cyclic_prefix_length

        M = self.fft_size
        N = self.num_blocks
        L = delay_spread + 1

        # m: [M]
        m = tf.range(M, dtype=tf.int32)

        # ell: [L]
        ell = tf.range(L, dtype=tf.int32)

        # q_idx[n, m] = n*block_len + cp + m
        n = tf.range(N, dtype=tf.int32)
        q_idx = (
            n[:, None] * block_len
            + self.cyclic_prefix_length
            + m[None, :]
        )  # [N, M]

        # values[n, ell, m] = gs[ell, q_idx[n,m]]
        values = tf.gather(gs, q_idx, axis=1)      # [L, N, M]
        values = tf.transpose(values, [1, 2, 0])   # [N, M, L]

        # row index = m
        row_idx = tf.broadcast_to(
            m[None, :, None],
            [N, M, L],
        )

        # col index = (m - ell) mod M
        col_idx = tf.math.floormod(
            m[None, :, None] - ell[None, None, :],
            M,
        )

        col_idx = tf.broadcast_to(
            col_idx,
            [N, M, L],
        )

        # block index = n
        block_idx = tf.broadcast_to(
            n[:, None, None],
            [N, M, L],
        )

        indices = tf.stack(
            [block_idx, row_idx, col_idx],
            axis=-1,
        )  # [N, M, L, 3]

        G_blocks = tf.scatter_nd(
            indices=tf.reshape(indices, [-1, 3]),
            updates=tf.reshape(values, [-1]),
            shape=[N, M, M],
        )

        return tf.cast(G_blocks, dtype)

    def call(self, y_time_serial, gs, no):
        y_blocks = self.remove_cp_blocks(y_time_serial)     # [..., N, M]
        G_blocks = self.build_block_channel_matrices(gs)    # [N, M, M]

        dtype = y_blocks.dtype
        sigma2 = tf.cast(no, dtype)

        # Noise covariance S = sigma2 * I_M, broadcast over N blocks
        eye = tf.eye(self.fft_size, dtype=dtype)
        S = sigma2 * tf.broadcast_to(
            eye[None, :, :],
            [self.num_blocks, self.fft_size, self.fft_size],
        )                                                   # [N, M, M]

        # Sionna returns W = H^H (H H^H + S)^-1
        W = lmmse_matrix(G_blocks, S)                       # [N, M, M]

        # Apply all N filters in one batched operation
        x_hat_blocks = tf.einsum("nij,...nj->...ni", W, y_blocks)

        return flatten_last_dims(x_hat_blocks, 2)