import tensorflow as tf
import numpy as np
from findpeaks import findpeaks
from sionna.phy import Block


class EmbeddedPilotChannelEstimator(Block):

    def __init__(
        self,
        num_ofdm_symbols,
        fft_size,
        pilot_info,
        detection_mode='threshold',   # 'threshold' | 'peakfinder'
        threshold_factor=1.732,
        # findpeaks kwargs — only used when detection_mode='peakfinder'
        fp_method='topology',         # 'topology' | 'mask'
        fp_denoise=None,              # None = no denoising (raw DD grid)
        fp_window=None,
        fp_scale=False,               # don't rescale — preserve complex magnitude
        fp_min_persistence=0.2,      # float or None: filter by persistence score
    ):
        super().__init__()

        assert detection_mode in ('threshold', 'peakfinder'), \
            "detection_mode must be 'threshold' or 'peakfinder'"

        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.pilot_info = pilot_info
        self.detection_mode = detection_mode
        self.threshold_factor = threshold_factor
        self.fp_min_persistence = fp_min_persistence

        self.mp = pilot_info["pilot_delay_idx"]
        self.np_ = pilot_info["pilot_doppler_idx"]   # renamed to avoid shadowing np
        self.lmax = pilot_info["lmax"]
        self.kmax = pilot_info["kmax"]
        self.pilot_symbol = tf.constant(pilot_info["pilot_symbol"], dtype=tf.complex64)

        # Build findpeaks once at construction time (no limit = keep all peaks)
        if detection_mode == 'peakfinder':
            self._fp = findpeaks(
                method=fp_method,
                limit=None,           # <-- no threshold, return everything
                scale=fp_scale,
                togray=False,         # already a 2D float array, not an image
                denoise=fp_denoise,
                window=fp_window,
                verbose=0,            # silence output
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phase_correction(self, k, ell):
        N  = self.num_ofdm_symbols
        M  = self.fft_size
        NM = N * M

        z = tf.exp(tf.complex(
            tf.constant(0.0, tf.float32),
            2.0 * np.pi / tf.cast(NM, tf.float32),
        ))
        phase_k = tf.pow(z, tf.cast(k * self.mp, tf.complex64))
        phase_l = tf.exp(tf.complex(
            tf.constant(0.0, tf.float32),
            -2.0 * np.pi * float(self.np_ * ell) / N,
        ))
        return phase_k * phase_l

    def _estimate_from_peaks(self, y0, peak_indices):
        """
        peak_indices : list of (doppler_idx, delay_idx) tuples
        Returns g_hat, l_hat, k_hat, h_dd_hat.
        """
        N, M = self.num_ofdm_symbols, self.fft_size
        h_dd_hat = tf.zeros([N, M], dtype=y0.dtype)
        g_list, l_list, k_list = [], [], []

        for (doppler_idx, delay_idx) in peak_indices:
            ell = delay_idx  - self.mp
            k   = doppler_idx - self.np_

            if not (0 <= ell <= self.lmax):
                continue
            if not (0 <= doppler_idx < N and 0 <= delay_idx < M):
                continue

            y_sample = y0[doppler_idx, delay_idx]
            correction = self._phase_correction(k, ell)
            g_hat = y_sample / (self.pilot_symbol * correction)

            g_list.append(g_hat)
            l_list.append(ell)
            k_list.append(k)
            h_dd_hat = tf.tensor_scatter_nd_update(
                h_dd_hat, [[doppler_idx, delay_idx]], [g_hat]
            )

        return g_list, l_list, k_list, h_dd_hat

    # ------------------------------------------------------------------
    # Detection modes
    # ------------------------------------------------------------------

    def _detect_threshold(self, y0, no):
        """Original threshold loop — returns list of (doppler_idx, delay_idx)."""
        sigma_w   = tf.sqrt(tf.cast(no, tf.float32))
        threshold = self.threshold_factor * sigma_w
        N, M = self.num_ofdm_symbols, self.fft_size

        peaks = []
        for ell in range(self.lmax + 1):
            for doppler_idx in range(N):
                delay_idx = self.mp + ell
                if not (0 <= delay_idx < M and 0 <= doppler_idx < N):
                    continue
                if tf.abs(y0[doppler_idx, delay_idx]) >= threshold:
                    peaks.append((doppler_idx, delay_idx))
        return peaks

    def _detect_peakfinder(self, y0, no):
        N, M = self.num_ofdm_symbols, self.fft_size
        mag = np.abs(y0.numpy())

        # compute threshold
        sigma_w   = float(tf.sqrt(tf.cast(no, tf.float32)).numpy())
        threshold = self.threshold_factor * sigma_w

        # crop to estimation region only
        d_lo = self.mp
        d_hi = self.mp + self.lmax + 1
        k_lo = 0
        k_hi = N

        crop = mag[k_lo:k_hi, d_lo:d_hi]

        results = self._fp.fit(crop)
        
        df = results['persistence']

        # amplitude filter
        df = df[df.apply(
            lambda row: crop[int(row['y']), int(row['x'])] >= threshold, axis=1
        )]

        # normalize persistence scores within this frame
        if self.fp_min_persistence is not None and len(df) > 0:
            df = df[df['score'] / df['score'].max() >= self.fp_min_persistence]

        df = df.copy()
        df['score_norm'] = df['score'] / df['score'].max()
        print(df[['x', 'y', 'score', 'score_norm']].sort_values('score_norm', ascending=False))

        peaks = [(int(row['y']) + k_lo, int(row['x']) + d_lo)
                for _, row in df.iterrows()]
        return peaks

    # ------------------------------------------------------------------
    # call
    # ------------------------------------------------------------------

    def call(self, y_dd, no):
        y0 = y_dd[0, 0, 0, :, :]  # [N, M]

        if self.detection_mode == 'threshold':
            peak_indices = self._detect_threshold(y0, no)
        else:
            peak_indices = self._detect_peakfinder(y0, no)

        g_list, l_list, k_list, h_dd_hat = self._estimate_from_peaks(y0, peak_indices)

        # Fallback: LOS if nothing detected
        if len(g_list) == 0:
            N, M = self.num_ofdm_symbols, self.fft_size
            g_hat = tf.constant([1.0 + 0.0j], dtype=tf.complex64)
            l_hat = tf.constant([0], dtype=tf.int32)
            k_hat = tf.constant([0], dtype=tf.int32)
            h_dd_hat = tf.tensor_scatter_nd_update(
                h_dd_hat, [[self.np_, self.mp]],
                [tf.constant(1.0 + 0.0j, dtype=tf.complex64)]
            )
        else:
            g_hat = tf.stack(g_list)
            l_hat = tf.constant(l_list, dtype=tf.int32)
            k_hat = tf.constant(k_list, dtype=tf.int32)

        return g_hat, l_hat, k_hat, h_dd_hat