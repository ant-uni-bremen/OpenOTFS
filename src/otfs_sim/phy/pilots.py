import numpy as np
import tensorflow as tf
from sionna.phy.ofdm import ResourceGrid, PilotPattern

"""
Embedded OTFS pilot pattern compatible with Sionna ResourceGrid.

Conventions:
    - Doppler axis -> OFDM symbol axis
    - Delay axis   -> subcarrier axis

The pilot is placed at the center of the delay-Doppler grid and surrounded
by a rectangular guard region used for embedded OTFS channel estimation.
"""

class EmbeddedOTFSPilotPattern(PilotPattern):
    """
    Embedded OTFS pilot pattern with rectangular guard region.

    The generated mask reserves a rectangular region around a single pilot
    symbol in the delay-Doppler grid:

        Doppler guard size : [-2*kmax, +2*kmax]
        Delay guard size   : [-lmax,  +lmax]

    Grid dimensions:
        num_ofdm_symbols :
            Number of Doppler bins / OFDM symbols.

        fft_size :
            Number of delay bins / subcarriers.

    Internal tensor shapes:
        pilot_mask :
            [num_tx,
             num_streams_per_tx,
             num_ofdm_symbols,
             num_effective_subcarriers]

        pilots :
            [num_tx,
             num_streams_per_tx,
             num_reserved_elements]

    Notes:
        - Only the center pilot location contains a nonzero pilot symbol.
        - All surrounding reserved REs are guards set to zero.
        - The object is directly compatible with
          ResourceGrid(..., pilot_pattern=...).
    """

    def __init__(
        self,
        num_ofdm_symbols,
        fft_size,
        subcarrier_spacing,
        num_tx,
        num_streams_per_tx,
        lmax,
        kmax,
        pilot_symbol=1.0 + 0.0j,
        normalize=False,
        precision=None,
    ):
        rg_tmp = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            pilot_pattern="empty",
        )

        self.pilot_delay_idx = rg_tmp.num_effective_subcarriers // 2
        self.pilot_doppler_idx = rg_tmp.num_ofdm_symbols // 2

        self.lmax = lmax
        self.kmax = kmax
        self.pilot_symbol_value = pilot_symbol

        # self.guard_doppler_left = self.pilot_doppler_idx - 2 * kmax
        # self.guard_doppler_right = self.pilot_doppler_idx + 2 * kmax
        
        # Full Doppler guard: reserve every Doppler bin
        self.guard_doppler_left = 0
        self.guard_doppler_right = rg_tmp.num_ofdm_symbols - 1

        self.guard_delay_left = self.pilot_delay_idx - lmax
        self.guard_delay_right = self.pilot_delay_idx + lmax

        assert self.guard_doppler_left >= 0
        assert self.guard_doppler_right < rg_tmp.num_ofdm_symbols
        assert self.guard_delay_left >= 0
        assert self.guard_delay_right < rg_tmp.num_effective_subcarriers

        pilot_mask = np.zeros(
            [
                rg_tmp.num_tx,
                rg_tmp.num_streams_per_tx,
                rg_tmp.num_ofdm_symbols,
                rg_tmp.num_effective_subcarriers,
            ],
            dtype=bool,
        )

        pilot_mask[
            :,
            :,
            self.guard_doppler_left : self.guard_doppler_right + 1,
            self.guard_delay_left : self.guard_delay_right + 1,
        ] = True

        num_reserved = np.sum(pilot_mask[0, 0])

        pilots = np.zeros(
            [rg_tmp.num_tx, rg_tmp.num_streams_per_tx, num_reserved],
            dtype=np.complex64,
        )

        self.pilot_linear_idx = (
            (self.pilot_doppler_idx - self.guard_doppler_left)
            * (self.guard_delay_right - self.guard_delay_left + 1)
            + (self.pilot_delay_idx - self.guard_delay_left)
        )

        pilots[:, :, self.pilot_linear_idx] = pilot_symbol

        super().__init__(
            mask=tf.constant(pilot_mask, dtype=tf.bool),
            pilots=tf.constant(pilots, dtype=tf.complex64),
            normalize=normalize,
            precision=precision,
        )

    @property
    def pilot_info(self):
        return {
            "pilot_delay_idx": self.pilot_delay_idx,
            "pilot_doppler_idx": self.pilot_doppler_idx,
            "guard_delay_left": self.guard_delay_left,
            "guard_delay_right": self.guard_delay_right,
            "guard_doppler_left": self.guard_doppler_left,
            "guard_doppler_right": self.guard_doppler_right,
            "pilot_symbol": self.pilot_symbol_value,
            "pilot_linear_idx": self.pilot_linear_idx,
            "lmax": self.lmax,
            "kmax": self.kmax,
        }