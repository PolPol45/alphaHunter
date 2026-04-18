import pathlib
import sys
import unittest


BOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from agents.report_agent import ReportAgent


class ReportAgentMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = ReportAgent()

    def test_portfolio_summary_exposes_accounting_metadata(self) -> None:
        summary = self.agent._portfolio_summary(
            {
                "total_equity": 1000.0,
                "cash": 500.0,
                "initial_capital": 900.0,
                "realized_pnl": 10.0,
                "total_pnl": 100.0,
                "total_pnl_pct": 0.111111,
                "peak_equity": 1100.0,
                "drawdown_pct": 0.05,
                "account_scope": "shared_ibkr_account",
                "equity_basis": "proportional_split_of_shared_ibkr_account",
                "configured_capital_share": 0.25,
                "allocation_note": "Synthetic split",
                "trades": [
                    {"reason": "TP_HIT"},
                    {"reason": "SL_HIT"},
                    {"reason": "TP_HIT"},
                ],
            }
        )

        self.assertEqual(summary["account_scope"], "shared_ibkr_account")
        self.assertEqual(
            summary["equity_basis"],
            "proportional_split_of_shared_ibkr_account",
        )
        self.assertEqual(summary["configured_capital_share"], 0.25)
        self.assertEqual(summary["allocation_note"], "Synthetic split")
        self.assertEqual(summary["exit_breakdown"], {"TP_HIT": 2, "SL_HIT": 1})

    def test_data_sources_exposes_portfolio_accounting_section(self) -> None:
        sources = self.agent._data_sources(
            {
                "execution_mode": "ibkr",
                "account_scope": "shared_ibkr_account",
                "equity_basis": "proportional_split_of_shared_ibkr_account",
            },
            {
                "execution_mode": "ibkr",
                "account_scope": "shared_ibkr_account",
                "equity_basis": "proportional_split_of_shared_ibkr_account",
            },
            {"data_source": "openbb", "world_events": []},
            {},
        )

        self.assertEqual(sources["execution_mode"], "ibkr")
        self.assertEqual(
            sources["portfolio_accounting"]["retail_scope"],
            "shared_ibkr_account",
        )
        self.assertEqual(
            sources["portfolio_accounting"]["institutional_equity_basis"],
            "proportional_split_of_shared_ibkr_account",
        )


if __name__ == "__main__":
    unittest.main()
