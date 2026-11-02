import numpy as np
import tensorflow as tf

from otfs_sim.utils.results import Timer, create_result_row
from otfs_sim.utils.run import make_model

def _pad_delay_dim(gs_a, gs_b):
    """
    Pad two [delay, time] tensors to the same delay dimension.
    """
    delay_a = tf.shape(gs_a)[0]
    delay_b = tf.shape(gs_b)[0]
    max_delay = tf.maximum(delay_a, delay_b)

    gs_a = tf.pad(gs_a, [[0, max_delay - delay_a], [0, 0]])
    gs_b = tf.pad(gs_b, [[0, max_delay - delay_b], [0, 0]])

    return gs_a, gs_b


def _compute_gs_nmse(gs_true, gs_est):
    """
    Compute normalized MSE between two delay-time channel responses.
    """
    gs_true, gs_est = _pad_delay_dim(gs_true, gs_est)

    error_power = tf.reduce_sum(tf.abs(gs_true - gs_est) ** 2)
    true_power = tf.reduce_sum(tf.abs(gs_true) ** 2)

    return error_power / (true_power + 1e-12)


def run_single_analysis_simulation(config):
    """
    Run repeated debug frames and collect estimator-quality metrics.

    This is intended for embedded-pilot threshold tuning. It assumes
    batch_size = 1 and uses model.run_frame_debug() instead of sim_ber().
    """

    NUM_ANALYSIS_FRAMES = 50

    if config.batch_size != 1:
        raise ValueError(
            "run_single_analysis_simulation assumes config.batch_size == 1."
        )

    model = make_model(config)

    analysis_results = []

    with Timer() as timer:
        for ebno_db in config.ebno_db:

            nmse_values = []
            detected_path_counts = []

            for _ in range(NUM_ANALYSIS_FRAMES):

                analysis = model.run_frame_debug(
                    batch_size=1,
                    ebno_db=ebno_db,
                )

                gs_true = analysis["gs_true"]
                gs_est = analysis["gs_for_lmmse"]

                nmse = _compute_gs_nmse(gs_true, gs_est)
                nmse_values.append(float(nmse.numpy()))

                g_hat = analysis["g_hat"]

                if g_hat is None:
                    num_paths = 0
                else:
                    num_paths = int(tf.shape(g_hat)[0].numpy())

                detected_path_counts.append(num_paths)

            analysis_results.append(
                {
                    "ebno_db": float(ebno_db),
                    "nmse_mean": float(np.mean(nmse_values)),
                    "nmse_std": float(np.std(nmse_values)),
                    "nmse_values": nmse_values,
                    "num_detected_paths_mean": float(
                        np.mean(detected_path_counts)
                    ),
                    "num_detected_paths_std": float(
                        np.std(detected_path_counts)
                    ),
                    "num_detected_paths_values": detected_path_counts,
                    "num_analysis_frames": NUM_ANALYSIS_FRAMES,
                }
            )

    dummy_ber = np.zeros(len(config.ebno_db), dtype=np.float32)
    dummy_bler = np.zeros(len(config.ebno_db), dtype=np.float32)

    row = create_result_row(
        config=config,
        ber=dummy_ber,
        bler=dummy_bler,
        duration=timer.duration,
        extra={
            "run_type": "analysis",
            "estimator_detection_mode":
                config.estimator_detection_mode,
            "estimation_threshold_factor":
                config.estimation_threshold_factor,
            "estimator_fp_min_persistence":
                config.estimator_fp_min_persistence,
            "num_analysis_frames": NUM_ANALYSIS_FRAMES,
            "analysis_results": analysis_results,
        },
    )

    return row