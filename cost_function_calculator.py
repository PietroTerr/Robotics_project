from __future__ import annotations

from typing import Iterable

DEFAULT_LAMBDA_S = 50_000
DEFAULT_LAMBDA_D = 7_000

RunResult = tuple[float, int, float]
"""A simulation result tuple: (total_time, stuck_events, final_distance)."""


def compute_run_cost(
    total_time: float,
    stuck_events: int,
    final_distance: float,
    *,
    lambda_s: int = DEFAULT_LAMBDA_S,
    lambda_d: int = DEFAULT_LAMBDA_D,
) -> float:
    """Compute the cost for a single simulation run.

    J_i = T_i + lambda_s * N_i^stuck + lambda_d * D_i^final

    Args:
        total_time: total mission time for the run.
        stuck_events: number of stuck events during the run.
        final_distance: cell distance from the rover's final cell to the target.
        lambda_s: penalty per stuck event.
        lambda_d: penalty per cell distance to the target.

    Returns:
        The scalar cost for the run.
    """
    return total_time + lambda_s * stuck_events + lambda_d * final_distance


def average_cost(
    run_results: Iterable[RunResult],
    *,
    lambda_s: int = DEFAULT_LAMBDA_S,
    lambda_d: int = DEFAULT_LAMBDA_D,
) -> float:
    """Compute the average cost over multiple simulation runs.

    The final score is:
        J = (1 / N) * sum(J_i)

    Args:
        run_results: iterable of run tuples (total_time, stuck_events, final_distance).
        lambda_s: penalty per stuck event.
        lambda_d: penalty per cell distance to the target.

    Returns:
        The average cost across all provided runs.

    Raises:
        ValueError: if run_results is empty.
    """
    results = list(run_results)
    if not results:
        raise ValueError("run_results must contain at least one run.")

    total_cost = 0.0
    for total_time, stuck_events, final_distance in results:
        total_cost += compute_run_cost(
            total_time,
            stuck_events,
            final_distance,
            lambda_s=lambda_s,
            lambda_d=lambda_d,
        )

    return total_cost / len(results)


def score_single_run(
    total_time: float,
    stuck_events: int,
    final_distance: float,
    *,
    lambda_s: int = DEFAULT_LAMBDA_S,
    lambda_d: int = DEFAULT_LAMBDA_D,
) -> float:
    """Compute the score for a single simulation run."""
    return compute_run_cost(
        total_time=total_time,
        stuck_events=stuck_events,
        final_distance=final_distance,
        lambda_s=lambda_s,
        lambda_d=lambda_d,
    )

