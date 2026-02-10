"""
Results persistence for Topological Blankets experiments.

Provides save_results() and load_results() for structured JSON output
with consistent schema: {timestamp, experiment_name, config, metrics, notes}.
"""

import json
import os
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def save_results(experiment_name: str,
                 metrics: dict,
                 config: dict = None,
                 notes: str = "") -> str:
    """
    Save experiment results as timestamped JSON to results/.

    Args:
        experiment_name: Short identifier (e.g. "strength_sweep", "spectral_comparison")
        metrics: Dictionary of metric values
        config: Dictionary of configuration parameters
        notes: Free-text notes about this run

    Returns:
        Path to the saved JSON file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{timestamp}_{experiment_name}.json"
    filepath = RESULTS_DIR / filename

    payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "config": config or {},
        "metrics": metrics,
        "notes": notes,
    }

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    print(f"Results saved to {filepath}")
    return str(filepath)


def load_results(filepath: str) -> dict:
    """Load a previously saved results JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def list_results(experiment_name: str = None) -> list:
    """
    List all result files, optionally filtered by experiment name.

    Returns list of (filepath, metadata) tuples sorted by timestamp.
    """
    if not RESULTS_DIR.exists():
        return []

    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        if p.name == "registry.json":
            continue
        if experiment_name and experiment_name not in p.name:
            continue
        try:
            data = load_results(str(p))
            results.append((str(p), data))
        except (json.JSONDecodeError, IOError):
            continue
    return results


def build_registry() -> dict:
    """
    Scan results/ and build a registry summarizing all experiments.

    Saves registry.json and returns the registry dict.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = list_results()

    entries = []
    for filepath, data in all_results:
        entries.append({
            "file": os.path.basename(filepath),
            "timestamp": data.get("timestamp", ""),
            "experiment_name": data.get("experiment_name", ""),
            "config_summary": _summarize_config(data.get("config", {})),
            "metrics_summary": _summarize_metrics(data.get("metrics", {})),
            "notes": data.get("notes", ""),
        })

    registry = {
        "generated": datetime.now().isoformat(),
        "total_experiments": len(entries),
        "entries": entries,
    }

    registry_path = RESULTS_DIR / "registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, default=_json_default)

    print(f"Registry saved to {registry_path} ({len(entries)} experiments)")
    return registry


def _summarize_config(config: dict) -> str:
    """One-line summary of config for the registry."""
    if not config:
        return ""
    parts = [f"{k}={v}" for k, v in list(config.items())[:5]]
    return ", ".join(parts)


def _summarize_metrics(metrics: dict) -> dict:
    """Extract key numeric metrics for the registry, handling nested structures."""
    summary = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            summary[k] = round(v, 4) if isinstance(v, float) else v
        elif isinstance(v, dict):
            # One level deeper: look for mean_ari, mean_f1, etc.
            for k2, v2 in v.items():
                if isinstance(v2, (int, float)) and k2.startswith('mean_'):
                    summary[f"{k}.{k2}"] = round(v2, 4) if isinstance(v2, float) else v2
                elif isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        if isinstance(v3, (int, float)) and k3.startswith('mean_'):
                            summary[f"{k}.{k2}.{k3}"] = round(v3, 4) if isinstance(v3, float) else v3
    # Truncate to avoid huge summaries
    if len(summary) > 10:
        summary = dict(list(summary.items())[:10])
        summary["..."] = "truncated"
    return summary


def _json_default(obj):
    """JSON serializer for objects not natively serializable."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
