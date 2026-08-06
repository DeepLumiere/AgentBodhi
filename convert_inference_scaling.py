#!/usr/bin/env python3
"""
convert_inference_scaling.py

This script recursive-lists and converts evaluation logs from the Hugging Face Bucket:
https://huggingface.co/buckets/ai-safety-institute/2026-inference-scaling-paper
into the unified schema format of `every_eval_ever` (https://github.com/evaleval/every_eval_ever).

Key features:
1. Efficient O(1) matching of trajectory, submission, and turn CSV data.
2. Insertion of all pending tabular data into the appropriate schema metadata/details fields (as string key-values).
3. Checkpoint mechanism (saved at `data/conversion_checkpoint.json`) to keep track of processed files and resume from interruptions.
4. Download-on-the-fly and immediate cleanup of raw .eval files to conserve disk space.
5. Strict schema validation of all converted aggregate (.json) and instance-level (.jsonl) files.
6. Incremental uploads to a Hugging Face PR on the target datastore repository (default: `evaleval/EEE_datastore`).
"""

import os
import sys
import json
import uuid
import time
import hashlib
import argparse
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import pandas as pd
from huggingface_hub import HfApi

# Import every_eval_ever converter and types
from every_eval_ever.converters.inspect.adapter import InspectAIAdapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.instance_level_types import InstanceLevelEvaluationLog


# ── Utility Functions ───────────────────────────────────────────────────────

def get_sha256_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load the progress checkpoint dictionary."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to parse checkpoint: {e}")
    return {"processed_files": {}, "stats": {"success": 0, "failed": 0}}


def save_checkpoint(checkpoint_path: Path, checkpoint: Dict[str, Any]) -> None:
    """Save the progress checkpoint dictionary."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, sort_keys=True)


# ── Loading & Indexing Tabular CSV Data ─────────────────────────────────────

def load_and_index_csv_data(data_dir: Path) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]],
                                                   Dict[Tuple[str, str, int], List[Dict[str, Any]]],
                                                   Dict[Tuple[str, str, int], List[Dict[str, Any]]]]:
    """
    Load trajectory, submission, and turn CSV tables and build fast lookups
    indexed by (log_file, sample_id, original_epoch).
    """
    print("\n[CSV Indexer] Loading CSV tables from disk...")
    start_time = time.time()

    traj_path = data_dir / "trajectory_data.csv"
    sub_path = data_dir / "submission_data.csv"
    turn_path = data_dir / "turn_data.csv"

    # Load tables
    df_traj = pd.read_csv(traj_path, low_memory=False)
    df_sub = pd.read_csv(sub_path, low_memory=False)
    df_turn = pd.read_csv(turn_path, low_memory=False)

    # Normalize log file paths to forward slashes
    df_traj["log_file"] = df_traj["log_file"].str.replace("\\", "/", regex=False)
    df_sub["log_file"] = df_sub["log_file"].str.replace("\\", "/", regex=False)
    df_turn["log_file"] = df_turn["log_file"].str.replace("\\", "/", regex=False)

    print(f"[CSV Indexer] Processing {len(df_traj)} trajectory rows...")
    traj_lookup = {}
    for row in df_traj.itertuples(index=False, name="Row"):
        row_dict = {k: v for k, v in row._asdict().items() if pd.notna(v)}
        log_file = row_dict.get("log_file")
        sample_id = str(row_dict.get("sample_id"))
        orig_epoch = row_dict.get("original_epoch")
        if log_file and sample_id and orig_epoch is not None:
            traj_lookup[(log_file, sample_id, int(orig_epoch))] = row_dict

    print(f"[CSV Indexer] Processing {len(df_sub)} submission rows...")
    sub_lookup = {}
    for row in df_sub.itertuples(index=False, name="Row"):
        row_dict = {k: v for k, v in row._asdict().items() if pd.notna(v)}
        log_file = row_dict.get("log_file")
        sample_id = str(row_dict.get("sample_id"))
        orig_epoch = row_dict.get("original_epoch")
        if log_file and sample_id and orig_epoch is not None:
            key = (log_file, sample_id, int(orig_epoch))
            if key not in sub_lookup:
                sub_lookup[key] = []
            sub_lookup[key].append(row_dict)

    # Sort submissions by submit_number
    for key in sub_lookup:
        sub_lookup[key].sort(key=lambda x: x.get("submit_number", 0))

    print(f"[CSV Indexer] Processing {len(df_turn)} turn rows...")
    turn_lookup = {}
    for row in df_turn.itertuples(index=False, name="Row"):
        row_dict = {k: v for k, v in row._asdict().items() if pd.notna(v)}
        log_file = row_dict.get("log_file")
        sample_id = str(row_dict.get("sample_id"))
        orig_epoch = row_dict.get("original_epoch")
        if log_file and sample_id and orig_epoch is not None:
            key = (log_file, sample_id, int(orig_epoch))
            if key not in turn_lookup:
                turn_lookup[key] = []
            turn_lookup[key].append(row_dict)

    # Sort turns by turn_number
    for key in turn_lookup:
        turn_lookup[key].sort(key=lambda x: x.get("turn_number", 0))

    elapsed = time.time() - start_time
    print(f"[CSV Indexer] Tables successfully loaded and indexed in {elapsed:.1f}s!")
    return traj_lookup, sub_lookup, turn_lookup


# ── PR Creation & Incremental Uploading ─────────────────────────────────────

def find_or_create_pr(api: HfApi, repo_id: str) -> int:
    """Find the most recent open PR by the current authenticated user on the target repo, or create one."""
    try:
        current_user = api.whoami().get("name")
    except Exception as e:
        print(f"Warning: Could not identify current Hugging Face user: {e}")
        current_user = None

    try:
        discussions = api.get_repo_discussions(repo_id=repo_id, repo_type="dataset")
        open_prs = [
            d for d in discussions
            if getattr(d, "is_pull_request", False)
            and d.status in ("open", "draft")
            and (d.author == current_user if current_user else True)
        ]
        if open_prs:
            latest_pr = max(open_prs, key=lambda x: x.num)
            print(f"[HF Hub] Reusing existing open PR #{latest_pr.num} ({latest_pr.url})")
            return latest_pr.num
    except Exception as e:
        print(f"Warning: Could not fetch PRs: {e}")

    # Create new PR
    print(f"[HF Hub] Creating a new Pull Request on {repo_id}...")
    pr = api.create_pull_request(
        repo_id=repo_id,
        title="Inference Scaling Evaluation Results Batch Conversion",
        description="Automated conversion of the 2026 UK AI Security Institute Inference Scaling Paper logs to the unified schema.",
        repo_type="dataset",
    )
    print(f"[HF Hub] Successfully created PR #{pr.num} ({pr.url})")
    return pr.num


def upload_batch_to_pr(api: HfApi, repo_id: str, pr_num: int, output_dir: Path) -> bool:
    """Upload converted schemas folder incrementally to the given PR branch."""
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN is not set in the environment. Skipping upload.")
        return False

    revision = f"refs/pr/{pr_num}"
    try:
        print(f"[HF Hub] Uploading converted outputs incrementally to PR #{pr_num}...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(output_dir),
            path_in_repo="data",
            repo_type="dataset",
            revision=revision,
            commit_message=f"Incremental batch upload of converted schemas",
        )
        print(f"[HF Hub] Successfully completed upload to PR #{pr_num}!")
        return True
    except Exception as e:
        print(f"[HF Hub] Incremental upload failed: {e}")
        traceback.print_exc()
        return False


# ── Conversion, Joining, and Validation Pipeline ────────────────────────────

def process_log_file(
    log_relative_path: str,
    traj_lookup: Dict[Tuple[str, str, int], Dict[str, Any]],
    sub_lookup: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    turn_lookup: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    output_dir: Path
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Download a single log file, convert it, merge trajectory/submission/turn data,
    validate, and write schemas.
    """
    tmp_eval_file = Path("temp_download.eval")
    if tmp_eval_file.exists():
        tmp_eval_file.unlink()

    # Step 1: Download raw .eval file
    print(f"\n[{log_relative_path}] Downloading log file...")
    download_start = time.time()
    res = subprocess.run([
        "hf", "buckets", "cp",
        f"hf://buckets/ai-safety-institute/2026-inference-scaling-paper/{log_relative_path}",
        str(tmp_eval_file)
    ], capture_output=True, text=True)

    if res.returncode != 0:
        return False, f"Download failed: {res.stderr}", {}

    print(f"[{log_relative_path}] Downloaded successfully in {time.time() - download_start:.1f}s.")

    # Step 2: Convert using InspectAIAdapter
    print(f"[{log_relative_path}] Running base Inspect AI adapter...")
    file_uuid = str(uuid.uuid4())
    metadata_args = {
        "parent_eval_output_dir": "output_schemas",
        "file_uuid": file_uuid
    }

    try:
        adapter = InspectAIAdapter()
        evaluation_log = adapter.transform_from_file(str(tmp_eval_file), metadata_args)
    except Exception as e:
        if tmp_eval_file.exists():
            tmp_eval_file.unlink()
        return False, f"Base adapter failed: {e}\n{traceback.format_exc()}", {}

    # Step 3: Find written physical file locations
    dataset_name = evaluation_log.evaluation_results[0].source_data.dataset_name
    model_id = evaluation_log.model_info.id
    if "/" in model_id:
        model_dev, model_name = model_id.split("/", 1)
    else:
        model_dev, model_name = evaluation_log.model_info.developer or "unknown", model_id

    physical_dir = Path("output_schemas") / dataset_name / model_dev / model_name
    physical_jsonl = physical_dir / f"{file_uuid}_samples.jsonl"
    physical_json = physical_dir / f"{file_uuid}.json"

    # Step 4: Post-process instance-level (.jsonl) file to inject submission and turn data
    print(f"[{log_relative_path}] Merging trajectory, submission, and turn CSV data...")
    post_processed_lines = []
    total_samples = 0
    matched_trajectories_count = 0

    log_file_key_path = log_relative_path

    # Collect trajectory rows for summary statistics
    matched_traj_rows = []

    if physical_jsonl.exists():
        with open(physical_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line_data = json.loads(line)
                sample_id = line_data.get("sample_id")
                metadata = line_data.setdefault("metadata", {})
                orig_epoch_str = metadata.get("epoch", "1")
                try:
                    orig_epoch = int(orig_epoch_str)
                except ValueError:
                    orig_epoch = 1

                lookup_key = (log_file_key_path, str(sample_id), orig_epoch)

                # Fetch and merge trajectory metrics
                traj_row = traj_lookup.get(lookup_key)
                if traj_row:
                    matched_trajectories_count += 1
                    matched_traj_rows.append(traj_row)
                    # Add to metadata
                    for k, v in traj_row.items():
                        if k not in ["eval", "model", "condition", "sample_id", "epoch", "original_epoch", "log_file"]:
                            metadata[f"traj_{k}"] = str(v)

                    # Enrich top-level token_usage
                    token_usage = line_data.setdefault("token_usage", {})
                    token_usage["input_tokens"] = int(traj_row.get("total_input_tokens_target_model", token_usage.get("input_tokens", 0)))
                    token_usage["output_tokens"] = int(traj_row.get("total_output_tokens_target_model", token_usage.get("output_tokens", 0)))
                    token_usage["total_tokens"] = int(traj_row.get("total_tokens_target_model", token_usage.get("total_tokens", 0)))
                    token_usage["input_tokens_cache_read"] = int(traj_row.get("total_cache_read_tokens_target_model", token_usage.get("input_tokens_cache_read", 0)))
                    token_usage["input_tokens_cache_write"] = int(traj_row.get("total_cache_write_tokens_target_model", token_usage.get("input_tokens_cache_write", 0)))

                    # Enrich evaluation turns/tool calls count
                    evaluation_sec = line_data.setdefault("evaluation", {})
                    if "turn_count" in traj_row:
                        evaluation_sec["num_turns"] = int(traj_row["turn_count"])

                # Fetch and merge submission details (multiple candidate answers per trajectory)
                sub_rows = sub_lookup.get(lookup_key)
                if sub_rows:
                    metadata["submissions"] = json.dumps(sub_rows)

                # Fetch and merge per-turn metrics
                turn_rows = turn_lookup.get(lookup_key)
                if turn_rows:
                    metadata["turns"] = json.dumps(turn_rows)

                # Strict validation of each sample line
                try:
                    InstanceLevelEvaluationLog.model_validate(line_data)
                except Exception as ve:
                    if tmp_eval_file.exists():
                        tmp_eval_file.unlink()
                    return False, f"Instance-level schema validation failed for sample_id={sample_id}: {ve}", {}

                post_processed_lines.append(line_data)
                total_samples += 1

        # Write post-processed .jsonl back to disk
        with open(physical_jsonl, "w", encoding="utf-8") as f:
            for line_data in post_processed_lines:
                f.write(json.dumps(line_data) + "\n")

    # Step 5: Post-process aggregate (.json) EvaluationLog
    evaluation_log_dict = json.loads(evaluation_log.model_dump_json(exclude_none=True))

    # Re-compute detailed_evaluation_results checksum and total rows
    if evaluation_log_dict.get("detailed_evaluation_results"):
        evaluation_log_dict["detailed_evaluation_results"]["checksum"] = get_sha256_hash(physical_jsonl)
        evaluation_log_dict["detailed_evaluation_results"]["total_rows"] = total_samples

    # Summarize aggregate trajectory-level details inside evaluation_results.score_details.details
    if matched_traj_rows and evaluation_log_dict.get("evaluation_results"):
        for result in evaluation_log_dict["evaluation_results"]:
            score_details = result.setdefault("score_details", {})
            details = score_details.setdefault("details", {})

            # Summarize metrics
            total_traj = len(matched_traj_rows)
            details["total_matched_trajectories"] = str(total_traj)

            # Stopping reason distributions
            stopping_reasons = {}
            for r in matched_traj_rows:
                sr = r.get("stopping_reason", "unknown")
                stopping_reasons[sr] = stopping_reasons.get(sr, 0) + 1
            for sr, count in stopping_reasons.items():
                details[f"stopping_reason_count_{sr}"] = str(count)

            # Average tokens and timing info
            for k in ["total_tokens_target_model", "total_tokens_other_models", "total_tokens_all_models", "turn_count"]:
                values = [r.get(k) for r in matched_traj_rows if r.get(k) is not None]
                if values:
                    details[f"avg_{k}"] = f"{sum(values) / len(values):.2f}"
                    details[f"total_{k}"] = str(sum(values))

    # Strict validation of the aggregate log
    try:
        EvaluationLog.model_validate(evaluation_log_dict)
    except Exception as ve:
        if tmp_eval_file.exists():
            tmp_eval_file.unlink()
        return False, f"Aggregate schema validation failed: {ve}", {}

    # Write aggregate JSON back to disk
    physical_json.parent.mkdir(parents=True, exist_ok=True)
    with open(physical_json, "w", encoding="utf-8") as f:
        json.dump(evaluation_log_dict, f, indent=4)

    # Clean up raw .eval file
    if tmp_eval_file.exists():
        tmp_eval_file.unlink()

    stats_summary = {
        "dataset_name": dataset_name,
        "model_id": model_id,
        "total_samples": total_samples,
        "matched_trajectories": matched_trajectories_count
    }
    return True, "Success", stats_summary


# ── Main Orchestrator Execution ──────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Inference Scaling logs to schema.")
    parser.add_argument("--repo-id", default="evaleval/EEE_datastore", help="Target Hugging Face dataset repository.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of logs to convert (0 = unlimited).")
    parser.add_argument("--batch-size", type=int, default=10, help="Commit/upload to PR every N logs.")
    parser.add_argument("--dry-run", action="store_true", help="Run conversion locally without PR creation or uploading.")
    args = parser.parse_args()

    output_dir = Path("output_schemas")
    output_dir.mkdir(exist_ok=True)

    checkpoint_path = Path("data/conversion_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)

    # Prepare Hugging Face Client
    hf_token = os.environ.get("HF_TOKEN")
    api = None
    pr_num = None
    if not args.dry_run:
        if not hf_token:
            print("Error: HF_TOKEN is not set in the environment. Exiting.")
            return 1
        api = HfApi(token=hf_token)
        pr_num = find_or_create_pr(api, args.repo_id)

    # Load and Index Tabular data
    traj_lookup, sub_lookup, turn_lookup = load_and_index_csv_data(Path("data"))

    # Fetch recursive list of .eval files from bucket using hf CLI
    print("\n[Orchestrator] Fetching recursive list of .eval files from HF Bucket...")
    res = subprocess.run([
        "hf", "buckets", "list",
        "hf://buckets/ai-safety-institute/2026-inference-scaling-paper/logs",
        "-R"
    ], capture_output=True, text=True)

    if res.returncode != 0:
        print(f"Error listing bucket files: {res.stderr}")
        return 1

    eval_files = []
    for line in res.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            path = parts[-1]
            if path.endswith(".eval"):
                # Normalize log file path
                normalized_path = path.replace("\\", "/")
                eval_files.append(normalized_path)

    print(f"[Orchestrator] Found {len(eval_files)} total log files in the bucket.")

    # Filter out files already successfully processed
    files_to_process = [f for f in eval_files if f not in checkpoint["processed_files"]]
    print(f"[Orchestrator] {len(eval_files) - len(files_to_process)} already processed. {len(files_to_process)} remaining.")

    if args.limit > 0:
        files_to_process = files_to_process[:args.limit]
        print(f"[Orchestrator] Limit set to {args.limit} files.")

    if not files_to_process:
        print("[Orchestrator] All files already processed! Exiting.")
        return 0

    success_count = checkpoint["stats"].get("success", 0)
    failed_count = checkpoint["stats"].get("failed", 0)

    # Iterate and process files
    pending_upload_count = 0
    start_time = time.time()

    for idx, log_file in enumerate(files_to_process):
        print(f"\n======================================================================")
        print(f"[{idx+1}/{len(files_to_process)}] Processing: {log_file}")
        print(f"======================================================================")

        success, msg, summary = process_log_file(
            log_file,
            traj_lookup,
            sub_lookup,
            turn_lookup,
            output_dir
        )

        if success:
            print(f"[{log_file}] Successfully converted!")
            print(f"  Summary: {summary}")
            success_count += 1
            checkpoint["processed_files"][log_file] = {
                "status": "success",
                "timestamp": time.time(),
                "summary": summary
            }
            pending_upload_count += 1
        else:
            print(f"[{log_file}] Failed: {msg}")
            failed_count += 1
            checkpoint["processed_files"][log_file] = {
                "status": "failed",
                "timestamp": time.time(),
                "error": msg
            }

        # Update checkpoint file after every run
        checkpoint["stats"]["success"] = success_count
        checkpoint["stats"]["failed"] = failed_count
        save_checkpoint(checkpoint_path, checkpoint)

        # Batch upload incrementally to avoid loss/interruption
        if not args.dry_run and pending_upload_count >= args.batch_size:
            print(f"\n[Incremental Upload] Reached batch size of {args.batch_size} files. Triggering upload...")
            if upload_batch_to_pr(api, args.repo_id, pr_num, output_dir):
                pending_upload_count = 0
            else:
                print("[Incremental Upload] Failed to upload current batch, will retry on next trigger.")

    # Final upload of any remaining pending files
    if not args.dry_run and pending_upload_count > 0:
        print(f"\n[Final Upload] Uploading remaining {pending_upload_count} processed files...")
        upload_batch_to_pr(api, args.repo_id, pr_num, output_dir)

    total_time = time.time() - start_time
    print(f"\n======================================================================")
    print(f"Finished conversion run!")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {failed_count}")
    print(f"  Total Time: {total_time/60:.2f} minutes")
    print(f"======================================================================")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
