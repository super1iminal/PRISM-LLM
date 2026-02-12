"""
Repairer.py - Detect and re-run missing/failed samples in evaluation parquet files.

Scans a results directory for parquet files with missing sample_ids, re-runs
only those samples, and writes complete results to a new *-repaired directory.
"""

import argparse
import os
import shutil
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from Evaluator import EvalModel, get_model, results_to_multiindex_df
from utils.DataLoader import DataLoader
from utils.Logging import setup_logger


def detect_missing_samples(
    parquet_path: str,
    expected_count: int,
    repair_failed: bool = False,
) -> Tuple[Set[int], Set[int]]:
    """
    Detect missing and failed sample_ids in a parquet file.

    Args:
        parquet_path: Path to the parquet file.
        expected_count: Total expected number of samples (0..N-1).
        repair_failed: Whether to also detect failed samples.

    Returns:
        (missing_ids, failed_ids) where:
          missing_ids - sample_ids not present in the parquet at all
          failed_ids  - sample_ids present but with 0 iterations and 0.0 LTL score
    """
    df = pd.read_parquet(parquet_path)
    expected_ids = set(range(expected_count))

    if df.empty:
        return expected_ids, set()

    present_ids = set(df.index.get_level_values("sample_id").unique())
    missing_ids = expected_ids - present_ids

    failed_ids: Set[int] = set()
    if repair_failed:
        # Failed samples: only 1 iteration row, that row has is_final=True,
        # final_ltl_score == 0.0, and iteration_time == 0.0 (the error fallback)
        for sid in present_ids:
            sample_df = df.loc[sid]
            if isinstance(sample_df, pd.Series):
                # Single iteration row
                if sample_df.get("final_ltl_score", 1.0) == 0.0 and sample_df.get("iteration_time", 1.0) == 0.0:
                    failed_ids.add(sid)
            else:
                if len(sample_df) == 1:
                    row = sample_df.iloc[0]
                    if row.get("final_ltl_score", 1.0) == 0.0 and row.get("iteration_time", 1.0) == 0.0:
                        failed_ids.add(sid)

    return missing_ids, failed_ids


def parse_model_type_from_filename(filename: str) -> EvalModel:
    """
    Parse the EvalModel enum from a parquet filename.

    E.g. 'LLM_VANILLA_GPT5_NANO_results.parquet' -> EvalModel.LLM_VANILLA_GPT5_NANO
    """
    name = filename.replace("_results.parquet", "")
    return EvalModel[name]


def results_to_multiindex_df_with_mapping(
    results: List[Dict],
    dataloader: DataLoader,
    sample_id_mapping: Dict[int, int],
) -> pd.DataFrame:
    """
    Convert results to MultiIndex DataFrame, mapping local indices to original sample_ids.

    Same logic as Evaluator.results_to_multiindex_df but uses sample_id_mapping[local_idx]
    for the sample_id column instead of the enumerate index.

    Args:
        results: List of result dicts from model.evaluate().
        dataloader: The subset DataLoader used for re-evaluation.
        sample_id_mapping: Dict mapping local index (0-based) -> original sample_id.

    Returns:
        DataFrame with MultiIndex (sample_id, iteration).
    """
    rows = []

    for local_idx, result in enumerate(results):
        original_sample_id = sample_id_mapping[local_idx]
        gridworld = dataloader.data[local_idx][0]
        expected_steps = dataloader.data[local_idx][1]
        num_iterations = result.get("Iterations_Used", 1)

        for iter_idx in range(num_iterations):
            iteration_prism_probs = result.get("Iteration_Prism_Probs", [{}])
            prism_probs = iteration_prism_probs[iter_idx] if iter_idx < len(iteration_prism_probs) else {}

            if not prism_probs and iter_idx == 0:
                prism_probs = result.get("Prism_Probabilities", {})

            iteration_times = result.get("Iteration_Times", [0.0])
            iteration_time = iteration_times[iter_idx] if iter_idx < len(iteration_times) else 0.0

            iteration_prism_times = result.get("Iteration_PRISM_Times", [0.0])
            prism_time = iteration_prism_times[iter_idx] if iter_idx < len(iteration_prism_times) else 0.0

            iteration_llm_times = result.get("Iteration_LLM_Times", [0.0])
            llm_time = iteration_llm_times[iter_idx] if iter_idx < len(iteration_llm_times) else 0.0

            iteration_mistakes = result.get("Iteration_Mistakes", [0])
            mistakes = iteration_mistakes[iter_idx] if iter_idx < len(iteration_mistakes) else 0

            iteration_costs = result.get("Iteration_Costs", [0.0])
            cost = iteration_costs[iter_idx] if iter_idx < len(iteration_costs) else 0.0

            row = {
                "sample_id": original_sample_id,
                "iteration": iter_idx + 1,
                "size": gridworld.size,
                "goals": len(gridworld.goals),
                "obstacles": len(gridworld.static_obstacles),
                "complexity": expected_steps,
                "iteration_time": iteration_time,
                "prism_time": prism_time,
                "llm_time": llm_time,
                "mistakes": mistakes,
                "cost": cost,
                **{f"prob_{k}": v for k, v in prism_probs.items()},
                "is_final": (iter_idx + 1 == num_iterations),
                "final_ltl_score": result.get("LTL_Score", 0.0),
                "success": result.get("Success", False),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index(["sample_id", "iteration"])
    return df


def make_subset_dataloader(
    full_dataloader: DataLoader,
    sample_ids: List[int],
) -> Tuple[DataLoader, Dict[int, int]]:
    """
    Create a DataLoader subset containing only the specified sample_ids.

    Args:
        full_dataloader: The complete DataLoader with all samples.
        sample_ids: Original sample_ids to include.

    Returns:
        (subset_loader, mapping) where mapping is {local_idx: original_sample_id}.
    """
    subset = DataLoader.__new__(DataLoader)
    subset.data = [full_dataloader.data[sid] for sid in sample_ids]
    mapping = {local_idx: sid for local_idx, sid in enumerate(sample_ids)}
    return subset, mapping


def repair(
    results_dir: str,
    dataset_path: str,
    max_workers: int = 10,
    repair_failed: bool = False,
    dry_run: bool = False,
) -> str:
    """
    Detect missing/failed samples in parquet files and re-run them.

    Args:
        results_dir: Path to the original results directory.
        dataset_path: Path to the CSV dataset file.
        max_workers: Number of parallel workers for re-evaluation.
        repair_failed: Also re-run samples that errored (0 iterations, 0 LTL).
        dry_run: Only report what's missing, don't re-run.

    Returns:
        Path to the repaired output directory.
    """
    # Load the full dataset
    full_loader = DataLoader(dataset_path)
    full_loader.load_data()
    expected_count = len(full_loader.data)

    # Determine output directory
    output_dir = results_dir.rstrip("/\\") + "-repaired"

    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        logger = setup_logger("repair", console_output=True, run_dir=output_dir, include_timestamp=False)
    else:
        # Lightweight logger for dry-run
        import logging
        logger = logging.getLogger("repair_dry_run")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)

    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Dataset: {dataset_path} ({expected_count} samples)")
    logger.info(f"Repair failed: {repair_failed}")
    logger.info(f"Dry run: {dry_run}")

    # Scan all parquet files
    parquet_files = [f for f in os.listdir(results_dir) if f.endswith("_results.parquet")]
    if not parquet_files:
        logger.warning("No parquet files found in results directory.")
        return output_dir

    for pq_file in sorted(parquet_files):
        pq_path = os.path.join(results_dir, pq_file)
        logger.info(f"\n--- {pq_file} ---")

        # Detect missing/failed samples
        missing_ids, failed_ids = detect_missing_samples(pq_path, expected_count, repair_failed)

        # Check for extra sample_ids beyond expected range
        df_existing = pd.read_parquet(pq_path)
        if not df_existing.empty:
            present_ids = set(df_existing.index.get_level_values("sample_id").unique())
            extra_ids = present_ids - set(range(expected_count))
            if extra_ids:
                logger.warning(
                    f"  WARNING: Parquet has sample_ids beyond expected range (0..{expected_count-1}): "
                    f"{sorted(extra_ids)[:10]}{'...' if len(extra_ids) > 10 else ''} — wrong dataset?"
                )

        ids_to_repair = sorted(missing_ids | failed_ids)

        logger.info(f"  Present: {expected_count - len(missing_ids)}/{expected_count}")
        if missing_ids:
            logger.info(f"  Missing ({len(missing_ids)}): {sorted(missing_ids)[:20]}{'...' if len(missing_ids) > 20 else ''}")
        if failed_ids:
            logger.info(f"  Failed  ({len(failed_ids)}): {sorted(failed_ids)[:20]}{'...' if len(failed_ids) > 20 else ''}")

        if not ids_to_repair:
            logger.info("  All samples present — no repair needed.")
            if not dry_run:
                shutil.copy2(pq_path, os.path.join(output_dir, pq_file))
            continue

        if dry_run:
            logger.info(f"  Would repair {len(ids_to_repair)} sample(s).")
            continue

        # Parse model type and create subset loader
        try:
            model_type = parse_model_type_from_filename(pq_file)
        except KeyError:
            logger.error(f"  Could not parse model type from '{pq_file}' — skipping.")
            shutil.copy2(pq_path, os.path.join(output_dir, pq_file))
            continue

        logger.info(f"  Re-running {len(ids_to_repair)} sample(s) for {model_type.name}...")
        subset_loader, mapping = make_subset_dataloader(full_loader, ids_to_repair)

        try:
            model = get_model(model_type)
            results = model.evaluate(subset_loader, max_workers=max_workers, run_dir=output_dir)

            new_df = results_to_multiindex_df_with_mapping(results, subset_loader, mapping)
            logger.info(f"  Re-evaluation produced {len(new_df)} rows for {len(ids_to_repair)} sample(s).")

            # Drop failed sample rows from existing data before merge
            if failed_ids:
                drop_mask = df_existing.index.get_level_values("sample_id").isin(failed_ids)
                df_existing = df_existing[~drop_mask]

            # Concatenate and sort
            combined = pd.concat([df_existing, new_df])
            combined = combined.sort_index()

            combined.to_parquet(os.path.join(output_dir, pq_file))
            logger.info(f"  Saved repaired file with {len(combined.index.get_level_values('sample_id').unique())} unique sample_ids.")

        except Exception as e:
            logger.error(f"  Re-evaluation FAILED for {model_type.name}: {type(e).__name__}: {e}")
            logger.exception("  Full traceback:")
            # Copy original file as-is to not lose existing data
            shutil.copy2(pq_path, os.path.join(output_dir, pq_file))

    logger.info(f"\nRepaired results saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repair missing/failed samples in evaluation parquet files."
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Path to the original results directory (e.g., out/results/100_20260211_08-03-04)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the CSV dataset file (e.g., data/grid_100_samples.csv)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Number of parallel workers for re-evaluation (default: 10)",
    )
    parser.add_argument(
        "--repair-failed",
        action="store_true",
        help="Also re-run samples that errored (0 iterations, 0.0 LTL score)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what's missing, don't re-run anything",
    )
    args = parser.parse_args()

    repair(
        results_dir=args.results_dir,
        dataset_path=args.dataset,
        max_workers=args.max_workers,
        repair_failed=args.repair_failed,
        dry_run=args.dry_run,
    )
