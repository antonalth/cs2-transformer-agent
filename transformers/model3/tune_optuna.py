import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna controller that spawns one process per trial.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--project_name", type=str, default="cs2-behavior-cloning-optuna")
    parser.add_argument("--study_name", type=str, default="model3_hpo")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///...db)")
    parser.add_argument("--output_root", type=str, default="./checkpoints_optuna")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Global optimize timeout in seconds")
    parser.add_argument("--trial_timeout", type=int, default=None, help="Per-trial timeout in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=1200, help="Per-trial optimizer steps")
    parser.add_argument("--devices", type=int, default=4, help="Devices used by each spawned trial process")
    parser.add_argument("--val_every_steps", type=int, default=200)
    parser.add_argument("--val_samples_limit", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_accumulation_steps", type=int, default=None)
    parser.add_argument("--monitor", type=str, default="val/loss")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_group", type=str, default=None, help="Defaults to --study_name.")
    parser.add_argument("--wandb_job_type", type=str, default="optuna-trial")
    parser.add_argument("--wandb_tags", type=str, default="optuna,model3", help="Comma-separated base tags.")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--worker_script", type=str, default="transformers/model3/tune_worker.py")
    return parser.parse_args()


def default_storage_url(output_root: Path, study_name: str) -> str:
    db_path = output_root / f"{study_name}.db"
    return f"sqlite:///{db_path.resolve()}"


def sample_params(trial: optuna.Trial, args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "train.lr": trial.suggest_float("train.lr", 1e-4, 6e-4, log=True),
        "train.weight_decay": trial.suggest_float("train.weight_decay", 1e-3, 2e-2, log=True),
        "model.qformer_num_queries": trial.suggest_categorical("model.qformer_num_queries", [32, 64, 96]),
        "model.qformer_num_hidden_layers": trial.suggest_categorical("model.qformer_num_hidden_layers", [2, 4, 6]),
        # Must stay divisible by backbone_splits=4 in current model config.
        "model.llama_layers": trial.suggest_categorical("model.llama_layers", [8, 12]),
    }

    if args.grad_accumulation_steps is not None:
        params["train.grad_accumulation_steps"] = args.grad_accumulation_steps
    else:
        params["train.grad_accumulation_steps"] = trial.suggest_categorical(
            "train.grad_accumulation_steps", [8, 16]
        )

    return params


def format_trial_run_name(study_name: str, trial_number: int, params: Dict[str, Any]) -> str:
    lr = params["train.lr"]
    wd = params["train.weight_decay"]
    q = params["model.qformer_num_queries"]
    ql = params["model.qformer_num_hidden_layers"]
    ll = params["model.llama_layers"]
    ga = params["train.grad_accumulation_steps"]
    return (
        f"{study_name}_t{trial_number:05d}"
        f"_lr{lr:.1e}_wd{wd:.1e}_q{q}_ql{ql}_l{ll}_ga{ga}"
    )


def build_worker_cmd(
    args: argparse.Namespace,
    python_exec: str,
    trial_number: int,
    run_name: str,
    wandb_group: str,
    wandb_tags: str,
    params_path: Path,
    result_path: Path,
    output_dir: Path,
) -> list[str]:
    cmd = [
        python_exec,
        args.worker_script,
        "--data_root",
        args.data_root,
        "--project_name",
        args.project_name,
        "--run_name",
        run_name,
        "--output_dir",
        str(output_dir),
        "--params_json",
        str(params_path),
        "--result_json",
        str(result_path),
        "--max_steps",
        str(args.max_steps),
        "--devices",
        str(args.devices),
        "--num_workers",
        str(args.num_workers),
        "--val_every_steps",
        str(args.val_every_steps),
        "--val_samples_limit",
        str(args.val_samples_limit),
        "--monitor",
        args.monitor,
        "--seed",
        str(args.seed + trial_number),
    ]
    if args.disable_wandb:
        cmd.append("--disable_wandb")
    else:
        cmd.extend(["--wandb_group", wandb_group])
        cmd.extend(["--wandb_job_type", args.wandb_job_type])
        cmd.extend(["--wandb_tags", wandb_tags])
    if args.save_checkpoints:
        cmd.append("--save_checkpoints")
    return cmd


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.max_steps <= 0:
        raise ValueError("--max_steps must be > 0.")
    if args.val_every_steps <= 0:
        raise ValueError("--val_every_steps must be > 0.")

    # val_check_interval counts training batches, while max_steps counts optimizer steps.
    # Ensure each trial can emit at least one validation metric.
    if args.grad_accumulation_steps is not None:
        min_grad_acc = args.grad_accumulation_steps
    else:
        min_grad_acc = 8  # current search space minimum
    min_train_batches = args.max_steps * min_grad_acc
    if min_train_batches < args.val_every_steps:
        raise ValueError(
            "Current settings may produce zero validation checks for some trials. "
            f"Need max_steps * min_grad_acc ({min_train_batches}) >= val_every_steps ({args.val_every_steps})."
        )

    storage = args.storage or default_storage_url(output_root, args.study_name)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    repo_root = Path(__file__).resolve().parents[2]
    python_exec = sys.executable

    def objective(trial: optuna.Trial) -> float:
        trial_dir = output_root / f"trial_{trial.number:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        params = sample_params(trial, args)
        params_path = trial_dir / "params.json"
        result_path = trial_dir / "result.json"
        log_path = trial_dir / "worker.log"

        with params_path.open("w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        run_name = format_trial_run_name(args.study_name, trial.number, params)
        wandb_group = args.wandb_group or args.study_name
        base_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        base_tags.extend([f"study:{args.study_name}", f"trial:{trial.number}"])
        wandb_tags = ",".join(base_tags)

        cmd = build_worker_cmd(
            args=args,
            python_exec=python_exec,
            trial_number=trial.number,
            run_name=run_name,
            wandb_group=wandb_group,
            wandb_tags=wandb_tags,
            params_path=params_path,
            result_path=result_path,
            output_dir=trial_dir,
        )

        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = proc.wait(timeout=args.trial_timeout)
            except subprocess.TimeoutExpired as exc:
                # Kill the full spawned process group (parent + distributed child ranks).
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                trial.set_user_attr("status", "timeout")
                trial.set_user_attr("trial_dir", str(trial_dir))
                raise optuna.TrialPruned(f"Trial timed out after {args.trial_timeout}s.") from exc

        trial.set_user_attr("worker_returncode", returncode)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("log_path", str(log_path))

        if not result_path.exists():
            raise RuntimeError(
                f"Trial {trial.number} finished without result file ({result_path}). "
                f"Inspect log: {log_path}"
            )

        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)

        status = result.get("status", "error")
        trial.set_user_attr("status", status)

        if returncode != 0 and status == "ok":
            raise RuntimeError(
                f"Trial {trial.number} worker exited with return code {returncode} "
                f"but reported status=ok. Inspect log: {log_path}"
            )

        if status != "ok":
            reason = result.get("reason") or result.get("error") or "worker reported failure"
            # OOM/error are treated as pruned to keep long searches moving.
            raise optuna.TrialPruned(f"Trial {trial.number} pruned: {reason}")

        metric = result.get("best_metric")
        if metric is None:
            raise RuntimeError(f"Trial {trial.number} result missing 'best_metric': {result_path}")

        metric_value = float(metric)
        trial.set_user_attr("best_metric", metric_value)
        trial.set_user_attr("last_step", result.get("last_step"))
        return metric_value

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials were found (all trials pruned/failed).")
        return

    best = study.best_trial
    print("Best trial:")
    print(f"  number={best.number}")
    print(f"  value={best.value}")
    print("  params:")
    for k, v in sorted(best.params.items()):
        print(f"    {k}={v}")


if __name__ == "__main__":
    main()
