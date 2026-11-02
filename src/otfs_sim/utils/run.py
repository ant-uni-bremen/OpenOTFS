import numpy as np
from pathlib import Path

from sionna.phy.utils import sim_ber

from otfs_sim.models.otfs_model import OTFSSimulationModel
from otfs_sim.models.ofdm_model import OFDMSimulationModel
from otfs_sim.utils.results import Timer, create_result_row, save_results, load_results


def make_model(config):
    if config.waveform == "otfs":
        return OTFSSimulationModel(config)
    if config.waveform == "ofdm":
        return OFDMSimulationModel(config)
    raise ValueError(f"Unsupported waveform='{config.waveform}'.")


def run_single_simulation(config):
    model = make_model(config)
    ebno_db = np.array(config.ebno_db, dtype=np.float32)

    with Timer() as timer:
        ber, bler = sim_ber(
            model, ebno_db,
            batch_size=config.batch_size,
            max_mc_iter=config.max_mc_iter,
            num_target_block_errors=config.num_target_block_errors,
        )

    return create_result_row(
        config=config, ber=ber, bler=bler, duration=timer.duration,
        extra={"max_mc_iter": config.max_mc_iter,
               "num_target_block_errors": config.num_target_block_errors},
    )


def run_single_simulation_with_checkpointing(config, output_dir):
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.pkl"

    ebno_db_all = np.array(config.ebno_db, dtype=np.float32)

    # --- Skip already-completed SNR points ---
    completed_ebno = []
    if checkpoint_path.exists():
        ckpt = load_results(checkpoint_path)
        completed_ebno = ckpt["ebno_db"]
        print(f"  [checkpoint] {len(completed_ebno)}/{len(ebno_db_all)} points already done, skipping.")

    remaining_ebno = np.array(
        [e for e in ebno_db_all if float(e) not in set(completed_ebno)],
        dtype=np.float32,
    )

    if remaining_ebno.size == 0:
        print("  [checkpoint] All SNR points already completed.")
    else:
        model = make_model(config)

        def checkpoint_callback(mc_iter, snr_idx, ebno_dbs,
                                 bit_errors, block_errors, nb_bits, nb_blocks):
            current_iter = int(mc_iter.numpy())
            current_snr  = int(snr_idx.numpy())

            hit_max_iter     = (current_iter == config.max_mc_iter - 1)
            hit_block_target = (
                config.num_target_block_errors is not None and
                int(block_errors[current_snr].numpy()) >= config.num_target_block_errors
            )

            if not (hit_max_iter or hit_block_target):
                return sim_ber.CALLBACK_CONTINUE

            # This SNR point just finished — check what's already saved
            already_done = []
            already_ber  = []
            already_bler = []
            if checkpoint_path.exists():
                ckpt = load_results(checkpoint_path)
                already_done = ckpt["ebno_db"]
                already_ber  = ckpt["ber"]
                already_bler = ckpt["bler"]

            ebno_val = float(ebno_dbs[current_snr].numpy())

            if ebno_val not in already_done:
                ber_val  = float(bit_errors[current_snr].numpy()) / float(nb_bits[current_snr].numpy())
                bler_val = float(block_errors[current_snr].numpy()) / float(nb_blocks[current_snr].numpy())

                already_done.append(ebno_val)
                already_ber.append(ber_val)
                already_bler.append(bler_val)

                save_results(
                    {"ebno_db": already_done, "ber": already_ber, "bler": already_bler},
                    output_dir, filename="checkpoint.pkl",
                )
                print(f"\n  [checkpoint] {ebno_val:.1f} dB saved ({len(already_done)}/{len(ebno_db_all)} done)")

            return sim_ber.CALLBACK_CONTINUE

        with Timer() as timer:
            ber, bler = sim_ber(
                model, remaining_ebno,
                batch_size=config.batch_size,
                max_mc_iter=config.max_mc_iter,
                num_target_block_errors=config.num_target_block_errors,
                forward_keyboard_interrupt=False,
                callback=checkpoint_callback,
            )

    # Read final checkpoint to build the result row
    ckpt = load_results(checkpoint_path)

    # Reorder to match original ebno_db order
    ebno_to_ber  = dict(zip(ckpt["ebno_db"], ckpt["ber"]))
    ebno_to_bler = dict(zip(ckpt["ebno_db"], ckpt["bler"]))
    ordered_ber  = [ebno_to_ber[float(e)]  for e in ebno_db_all]
    ordered_bler = [ebno_to_bler[float(e)] for e in ebno_db_all]

    return create_result_row(
        config=config,
        ber=np.array(ordered_ber),
        bler=np.array(ordered_bler),
        duration=timer.duration if remaining_ebno.size > 0 else 0.0,
        extra={"max_mc_iter": config.max_mc_iter,
               "num_target_block_errors": config.num_target_block_errors},
    )