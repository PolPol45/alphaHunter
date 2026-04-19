import sys
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator
from agents.base_agent import BASE_DIR, DATA_DIR, LOGS_DIR, REPORTS_DIR
from backtesting.auto_ml_pipeline import AutoMLBacktestPipeline

# ── Disk hygiene ──────────────────────────────────────────────────────────────

_LOG_MAX_MB   = 20    # truncate log files above this size
_REPORT_KEEP  = 48    # keep only last N reports
_JSONL_MAX_MB = 10    # truncate append-only JSONL files above this size

def cleanup_disk(verbose: bool = False) -> None:
    """Remove orphaned .tmp/.bak files, rotate logs, prune old reports."""
    freed = 0

    # 1. Orphaned .tmp files in data/ — .bak files are recovery snapshots, never delete them
    for pattern in ("*.tmp",):
        for p in DATA_DIR.glob(pattern):
            size = p.stat().st_size
            p.unlink()
            freed += size
            if verbose:
                print(f"  [cleanup] removed {p.name} ({size//1024}KB)")

    # 2. Truncate oversized log files (keep last _LOG_MAX_MB)
    for p in LOGS_DIR.glob("*.log"):
        size = p.stat().st_size
        limit = _LOG_MAX_MB * 1024 * 1024
        if size > limit:
            # Keep last limit bytes
            with open(p, "rb") as f:
                f.seek(-limit, 2)
                tail = f.read()
            with open(p, "wb") as f:
                f.write(tail)
            freed += size - limit
            if verbose:
                print(f"  [cleanup] truncated {p.name} {size//1024//1024}MB → {_LOG_MAX_MB}MB")

    # 3. Prune old reports — keep only last _REPORT_KEEP
    reports = sorted(REPORTS_DIR.glob("report_*.json"), key=lambda p: p.stat().st_mtime)
    for old in reports[:-_REPORT_KEEP]:
        size = old.stat().st_size
        old.unlink()
        freed += size
        if verbose:
            print(f"  [cleanup] pruned report {old.name}")

    # 4. Truncate oversized JSONL append files in data/
    for p in DATA_DIR.glob("*.jsonl"):
        size = p.stat().st_size
        limit = _JSONL_MAX_MB * 1024 * 1024
        if size > limit:
            with open(p, "rb") as f:
                f.seek(-limit, 2)
                # Skip partial first line
                f.readline()
                tail = f.read()
            with open(p, "wb") as f:
                f.write(tail)
            freed += size - limit
            if verbose:
                print(f"  [cleanup] truncated {p.name} {size//1024//1024}MB → {_JSONL_MAX_MB}MB")

    if freed > 0:
        print(f"  [cleanup] freed {freed // 1024 // 1024}MB total")


def main():
    print("==========================================================")
    print("  WALK-FORWARD OPTIMIZATION BACKTEST RUNNER               ")
    print("  Usa l'intero pipeline dell'orchestrator per ogni giorno  ")
    print("  ML retraining ogni 30 giorni simulati                   ")
    print("==========================================================")

    orch = Orchestrator()

    # Force ML to retrain immediately (no date-based skip)
    orch.ml_strategy_agent._refresh_days = 0

    # Disable background agent threads — walk-forward is deterministic/sequential,
    # async threads writing to DATA_DIR between simulated days causes file corruption
    # and log interleaving. Sector/Stock/Feature agents run synchronously instead.
    def _no_background_agent(name, agent, *args, **kwargs):
        orch.logger.debug(f"[walk-forward] {name} running synchronously (bg threads disabled)")
        try:
            agent.run()
        except Exception as e:
            orch.logger.error(f"[walk-forward] {name} sync run failed: {e}")

    orch._launch_background_agent = _no_background_agent
    # Also disable auto-backtest rolling thread (would spawn nested BacktestingAgent)
    orch._maybe_trigger_auto_backtest = lambda: None

    # Monkey-patch BacktestingAgent's execution step to inject full orchestrator cycle
    original_exec_run = orch.backtesting_agent._execution.run
    orch.backtesting_agent.steps_counter = 0

    # Walk-forward flag: tells _run_cycle to skip Risk+Execution (already run by BacktestingAgent)
    orch._wf_skip_risk_exec = True

    original_run_cycle = orch._run_cycle

    def wf_run_cycle(step: int) -> None:
        """Run orchestrator cycle but skip RiskAgent+ExecutionAgent — already ran inside BacktestingAgent."""
        original_exec_agent_run  = orch.execution_agent.run
        original_risk_agent_run  = orch.risk_agent.run
        orch.execution_agent.run = lambda: True
        orch.risk_agent.run      = lambda: True
        try:
            original_run_cycle(step)
        finally:
            orch.execution_agent.run = original_exec_agent_run
            orch.risk_agent.run      = original_risk_agent_run

    def hooked_exec_run():
        res = original_exec_run()
        orch.backtesting_agent.steps_counter += 1
        step = orch.backtesting_agent.steps_counter

        # Run full orchestrator cycle (all agents) at every step
        # Risk+Execution are suppressed inside wf_run_cycle — already ran in BacktestingAgent
        orch.logger.info(f"*** Walk-Forward: running full pipeline (giorno simulato {step}) ***")
        wf_run_cycle(step)

        # Periodic disk cleanup every 30 cycles
        if step % 30 == 0:
            cleanup_disk(verbose=False)

        # Extra ML + AutoML retraining every 30 days
        if step % 30 == 0:
            orch.logger.info(f"*** Walk-Forward Optimization (Giorno {step}) — ML Retraining ***")

            orch.logger.info("   1/3 Feature Store...")
            try:
                orch.feature_store_agent.run()
            except Exception as e:
                orch.logger.error(f"Errore FeatureStore: {e}")

            orch.logger.info("   2/3 ML Training K-Fold...")
            try:
                orch.ml_strategy_agent.run()
            except Exception as e:
                orch.logger.error(f"Errore ML: {e}")

            orch.logger.info("   3/3 AutoML Pipeline Pesi...")
            try:
                pipeline = AutoMLBacktestPipeline(repo_dir=BASE_DIR, workspace_dir=BASE_DIR.parent)
                pipeline.run()
            except Exception as e:
                orch.logger.error(f"Errore AutoML: {e}")

            orch.logger.info("*** Fine Ciclo Apprendimento. Ritorno al Mercato! ***")

        return res

    orch.backtesting_agent._execution.run = hooked_exec_run

    print("\nPre-run disk cleanup...")
    cleanup_disk(verbose=True)

    print("\nInizio simulazione... Tutti gli agenti attivi ad ogni step.")
    orch.backtesting_agent.run()


if __name__ == "__main__":
    main()
