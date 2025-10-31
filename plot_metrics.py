from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt

METRICS_PATH = Path("rlp_metrics.pkl")
OUTPUT_DIR = Path("rlp_metric_plots")


def _align_steps(
    steps: List[int],
    values: List[float],
) -> Tuple[List[int], List[float]]:
    """Trim or synthesize step indices to match the length of the metric values."""
    if not values:
        return [], []

    if steps:
        trimmed_len = min(len(steps), len(values))
        return steps[:trimmed_len], values[:trimmed_len]

    synthesized_steps = list(range(1, len(values) + 1))
    return synthesized_steps, values


def _plot_series(
    steps: List[int],
    values: List[float],
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(steps, values, marker="o", markersize=3)
    ax.set_xlabel("Global step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_reward_with_std(
    steps: List[int],
    reward: List[float],
    reward_std: List[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(steps, reward, label="mean reward", color="tab:blue")
    if reward_std:
        trimmed_std = reward_std[: len(steps)]
        ax.fill_between(
            steps,
            [m - s for m, s in zip(reward, trimmed_std)],
            [m + s for m, s in zip(reward, trimmed_std)],
            alpha=0.2,
            color="tab:blue",
            label="reward ± std",
        )
    ax.set_xlabel("Global step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward History")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected metrics format in {path}")
    return data


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

    data = load_metrics(METRICS_PATH)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    steps: List[int] = data.get("step_history", [])

    reward: List[float] = data.get("reward_history", [])
    reward_std: List[float] = data.get("reward_std_history", [])
    cot_length: List[float] = data.get("cot_length_history", [])
    loss: List[float] = data.get("loss_history", [])

    if reward:
        reward_steps, reward_values = _align_steps(steps, reward)
        plot_reward_with_std(
            reward_steps,
            reward_values,
            reward_std,
            output_dir / "reward_history.png",
        )

    if cot_length:
        cot_steps, cot_values = _align_steps(steps, cot_length)
        _plot_series(
            cot_steps,
            cot_values,
            output_dir / "cot_length_history.png",
            title="Chain-of-Thought Length",
            ylabel="Tokens",
        )

    if loss:
        loss_steps, loss_values = _align_steps(steps, loss)
        _plot_series(
            loss_steps,
            loss_values,
            output_dir / "loss_history.png",
            title="PPO Objective Loss",
            ylabel="Loss",
        )

    if reward_std and not reward:
        std_steps, std_values = _align_steps(steps, reward_std)
        _plot_series(
            std_steps,
            std_values,
            output_dir / "reward_std_history.png",
            title="Reward Standard Deviation",
            ylabel="Std Dev",
        )

    clipped_tokens = data.get("clipped_tokens")
    global_step = data.get("global_step")
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_lines = []
        if global_step is not None:
            summary_lines.append(f"global_step: {global_step}")
        if clipped_tokens is not None:
            summary_lines.append(f"clipped_tokens: {clipped_tokens}")
        if not summary_lines:
            summary_lines.append("No scalar summary values found.")
        summary_file.write("\n".join(summary_lines))


if __name__ == "__main__":
    main()
