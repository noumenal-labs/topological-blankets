"""
Headless plotting utilities for Topological Blankets experiments.

Sets Agg backend on import so plots work in headless Ralph iterations.
All figures are saved as PNG to results/; no plt.show() calls.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def save_figure(fig, name: str, experiment_name: str = "",
                dpi: int = 150) -> str:
    """
    Save a matplotlib figure as PNG to results/.

    Args:
        fig: matplotlib Figure object
        name: Short descriptive name (e.g. "strength_sweep_ari")
        experiment_name: Optional prefix for grouping
        dpi: Resolution

    Returns:
        Path to the saved PNG file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if experiment_name:
        filename = f"{timestamp}_{experiment_name}_{name}.png"
    else:
        filename = f"{timestamp}_{name}.png"

    filepath = RESULTS_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {filepath}")
    return str(filepath)
