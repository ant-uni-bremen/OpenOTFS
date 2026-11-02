# otfs_sim/phy/channel/apply_channel.py

from __future__ import annotations

from sionna.phy import Block
from sionna.phy.channel import ApplyTimeChannel

from otfs_sim.phy.channel.channel_utils import (
    cir_to_h_time,
    get_l_tot,
)


class ApplyCIRTimeChannel(Block):
    """
    Apply a time-domain channel from Sionna CIR tensors a,tau.

    Input:
        x_time:
            [batch, num_tx, num_tx_ant, num_time_samples]

        a:
            [batch, num_rx, num_rx_ant, num_tx, num_tx_ant,
             num_paths, num_time_steps]

        tau:
            [batch, num_rx, num_tx, num_paths]

    Returns:
        y_time:
            [batch, num_rx, num_rx_ant,
             num_time_samples + l_tot - 1]

        h_time:
            [batch, num_rx, num_rx_ant, num_tx, num_tx_ant,
             num_time_steps, l_tot]
    """

    def __init__(
        self,
        bandwidth,
        l_min: int = 0,
        l_max: int | None = None,
        l_margin: int = 0,
        normalize: bool = False,
    ):
        super().__init__()

        self.bandwidth = bandwidth

        self.l_min = l_min
        self.l_max = l_max
        self.l_margin = l_margin

        self.normalize = normalize

        self.apply_time_channel = None

        self.last_h_time = None
        self.last_y_time = None

        self.last_l_min = None
        self.last_l_max = None

    def call(self, x_time, a, tau):
        """
        Apply the channel.

        Args:
            x_time:
                [batch, num_tx, num_tx_ant, num_time_samples]

            a:
                CIR coefficients.

            tau:
                CIR delays.

        Returns:
            y_time:
                Channel output.

            h_time:
                Sionna discrete-time channel taps.
        """

        num_time_samples = int(x_time.shape[-1])

        h_time, l_min, l_max = cir_to_h_time(
            a=a,
            tau=tau,
            bandwidth=self.bandwidth,
            l_min=self.l_min,
            l_max=self.l_max,
            l_margin=self.l_margin,
            normalize=self.normalize,
        )

        l_tot = get_l_tot(
            l_min=l_min,
            l_max=l_max,
        )

        if self.apply_time_channel is None:
            self.apply_time_channel = ApplyTimeChannel(
                num_time_samples=num_time_samples,
                l_tot=l_tot,
            )

        y_time = self.apply_time_channel(
            x_time,
            h_time,
        )

        self.last_h_time = h_time
        self.last_y_time = y_time

        self.last_l_min = l_min
        self.last_l_max = l_max

        return y_time, h_time