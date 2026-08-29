"""Run the canonical benchmark on base then head in one host session.

Installs the base commit's package, times it, reinstalls the head commit's
package, times it with the same flags, then writes Δ%. Both legs use this
checkout's ``benchmark/`` harness so the comparison is fair when the runner
landed on the PR first.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.io import parse_results_csv  # noqa: E402
from benchmark.metrics import _as_float  # noqa: E402
from benchmark.relative import (  # noqa: E402
    compare_in_run_rows,
    write_relative_artifacts,
)

RUNNER = ROOT / "benchmark" / "run_benchmark.py"
RELATIVE_ROOT = ROOT / "benchmark" / "results" / "relative"


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd or ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_sha(ref: str) -> str:
    try:
        return _git("rev-parse", "--verify", ref)
    except subprocess.CalledProcessError:
        _git("fetch", "--depth", "1", "origin", ref.removeprefix("origin/"))
        return _git("rev-parse", "--verify", ref)


def _install_src(src_root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(src_root),
            "--no-deps",
            "--quiet",
        ],
        check=True,
    )


def _run_leg(output_dir: Path, forwarded: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--output-dir",
        str(output_dir),
        *forwarded,
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    csv_path = output_dir / "results.csv"
    if not csv_path.is_file():
        raise SystemExit(f"missing results.csv in {output_dir}")
    return csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "In-run relative benchmark: time origin/main then the PR commit "
            "on this machine"
        )
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="git ref installed and timed first (default: origin/main)",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="git ref installed and timed second (default: HEAD)",
    )
    parser.add_argument(
        "--fail-on-slow-pct",
        type=float,
        default=None,
        help="exit 2 if any ok cell's Δ%% exceeds this (optional)",
    )
    args, forwarded = parser.parse_known_args(argv)
    args.forwarded = forwarded
    if "--output-dir" in forwarded:
        raise SystemExit("pass --output-dir only via run_relative.py internals")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not RUNNER.is_file():
        raise SystemExit(f"missing harness {RUNNER}")
    head_sha = _resolve_sha(args.head)
    try:
        base_sha = _resolve_sha(args.base)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"cannot resolve base ref {args.base!r}; fetch main first"
        ) from error

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_root = RELATIVE_ROOT / f"{stamp}_{head_sha[:7]}"
    suffix = 2
    while out_root.exists():
        out_root = RELATIVE_ROOT / f"{stamp}_{head_sha[:7]}_{suffix}"
        suffix += 1
    base_out = out_root / "base"
    head_out = out_root / "head"
    worktree = Path(tempfile.gettempdir()) / f"skfolio-base-{base_sha[:12]}"
    if worktree.exists():
        try:
            _git("worktree", "remove", "--force", str(worktree))
        except subprocess.CalledProcessError:
            shutil.rmtree(worktree, ignore_errors=True)
    try:
        _git("worktree", "add", "--detach", str(worktree), base_sha)
        print(f"== base {args.base} {base_sha} ==", flush=True)
        _install_src(worktree)
        _run_leg(base_out, args.forwarded)
        print(f"== head {args.head} {head_sha} ==", flush=True)
        _install_src(ROOT)
        _run_leg(head_out, args.forwarded)
    finally:
        shutil.rmtree(worktree / ".git", ignore_errors=True)
        try:
            _git("worktree", "remove", "--force", str(worktree))
        except subprocess.CalledProcessError:
            shutil.rmtree(worktree, ignore_errors=True)
        _install_src(ROOT)

    base_rows = parse_results_csv(base_out / "results.csv")
    head_rows = parse_results_csv(head_out / "results.csv")
    delta_rows = compare_in_run_rows(base_rows, head_rows)
    payload = {
        "kind": "in-run-relative",
        "base_ref": args.base,
        "head_ref": args.head,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "delta_pct_definition": "100 * (head_time - base_time) / base_time",
        "forwarded_flags": args.forwarded,
        "rows": delta_rows,
    }
    write_relative_artifacts(
        out_root,
        rows=delta_rows,
        payload=payload,
        base_ref=args.base,
        head_ref=args.head,
        base_sha=base_sha,
        head_sha=head_sha,
    )
    print(f"Wrote {out_root}", flush=True)
    if args.fail_on_slow_pct is not None:
        slow = [
            row
            for row in delta_rows
            if row.get("base_status") == "ok"
            and row.get("head_status") == "ok"
            and _as_float(row.get("delta_pct")) > args.fail_on_slow_pct
        ]
        if slow:
            print(
                f"{len(slow)} cells slower than {args.fail_on_slow_pct}% "
                "vs in-run base",
                flush=True,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
