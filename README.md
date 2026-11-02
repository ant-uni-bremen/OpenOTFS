# OpenOTFS

**OpenOTFS** is an open-source, TensorFlow-based framework for link-level simulation of **Orthogonal Time Frequency Space (OTFS)** modulation, built on [Sionna](https://nvlabs.github.io/sionna/). It provides OTFS processing blocks, channel and receiver components, and examples for investigating transmission over time-varying wireless channels.

The framework was developed for research into OTFS performance and receiver processing, with an emphasis on simulations that can be inspected and modified in Python. It also includes OFDM components for comparison.

## Features

- OTFS modulation and demodulation integrated with a Sionna-based physical-layer simulation chain.
- OFDM modulation and demodulation components.
- Synthetic delay–Doppler multipath channels and channel-representation utilities.
- Pilot-based channel estimation and block LMMSE equalization.
- Channel-generation components for non-terrestrial-network (NTN) scenarios using [OpenNTN](https://github.com/ant-uni-bremen/OpenNTN).
- Utilities for simulation configuration, execution, results, and BER/BLER visualization.
- Jupyter notebooks illustrating key concepts and receiver operations.

## Installation (Ubuntu)

**Prerequisites:** [Conda](https://docs.conda.io/), Git, and `curl`.

Clone the repository:

```bash
git clone https://github.com/ant-uni-bremen/OpenOTFS.git
cd OpenOTFS
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate openotfs
```

Install the Sionna 1.x-compatible version of OpenNTN:

```bash
bash install_openntn.sh
```

> The OpenNTN installer integrates its NTN channel module into the Sionna installation inside the active Conda environment. Run it in the dedicated `openotfs` environment, not a shared environment.

For terminal-based Python sessions, make the repository source directory importable:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Check the installation:

```bash
python -c 'from otfs_sim.models import OTFSSimulationModel; print("OpenOTFS import OK")'
python -c 'from sionna.phy.channel.tr38811 import DenseUrban; print("OpenNTN import OK")'
```

The Python import package is currently named `otfs_sim`. No separate installation of the OpenOTFS source package is required when `src/` is on the Python path. The notebooks also contain local path setup for importing from the repository.

**Tested setup:** Ubuntu, Python 3.11, Sionna 1.2.1, TensorFlow 2.21.0, and OpenNTN 1.0.0. The end-to-end notebook has been run successfully in a newly created Conda environment. Windows installation has not yet been verified.

## Examples

The [`examples/`](examples/) directory contains four notebooks. They can be read in the following order:

| Notebook | Description |
| --- | --- |
| [Channel representations](examples/channel_representations.ipynb) | Visualize the representations of synthetic doubly dispersive channels in different signal-processing domains. |
| [OFDM grid, cyclic prefix, and channel diagonalization](examples/ofdm_grid_cp_channel_diagonalization.ipynb) | Explore the effects of the cyclic prefix and time variation on the effective OFDM channel. |
| [OTFS end-to-end](examples/otfs_end_to_end.ipynb) | Follow an OTFS transmission through modulation, channel application, and receiver processing. **Recommended starting point.** |
| [OTFS channel estimation and block LMMSE equalization](examples/otfs_channel_estimation_block_lmmse.ipynb) | Examine pilot-based channel estimation and equalization in more detail. |

To open the notebooks, activate the environment and launch JupyterLab from the repository root:

```bash
conda activate openotfs
jupyter lab
```

Alternatively, open the repository in VS Code and select the `openotfs` Conda environment as the notebook kernel.

## Repository structure

```text
OpenOTFS/
├── src/otfs_sim/
│   ├── models/          # End-to-end OTFS and OFDM simulation models
│   ├── phy/             # Modulation, pilots, estimation, equalization, channels
│   ├── utils/           # Configuration, execution, analysis, and results
│   └── visualization/   # BER/BLER plotting utilities
├── examples/            # Jupyter notebooks
├── environment.yml      # Conda environment
└── install_openntn.sh   # Compatible OpenNTN integration (Ubuntu)
```

## Compatibility and scope

OpenOTFS currently targets the **Sionna 1.x / TensorFlow** software stack. In particular, the tested NTN integration uses **OpenNTN 1.0.0**; it is not interchangeable with newer OpenNTN versions built for Sionna 2.x.

The present repository focuses on simulation components and illustrative notebooks. Extensive automated testing, a packaged Python distribution, and Windows installation support are not yet provided.

## Citation

If you use OpenOTFS in academic work, please cite the relevant publication when available. Citation metadata will be added here.

## Acknowledgments

OpenOTFS builds on [Sionna](https://nvlabs.github.io/sionna/) for physical-layer simulation and uses [OpenNTN](https://github.com/ant-uni-bremen/OpenNTN) for NTN channel modeling. Please also cite these projects where applicable.
