"""Point-in-time evidence store. Labels are discarded at the boundary."""
import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool

from .schemas import CaseQuery, VelocityQuery

FEATURES = ["amount", "hour", "history_count", "history_mean", "amount_ratio",
            "count_1h", "amount_1h", "count_24h", "amount_24h"]


class EvidenceStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame[["transaction_id", "account_id", "timestamp", "amount"]].copy()
        self.by_id = self.frame.set_index("transaction_id")
        self.accounts = {}
        for account, group in self.frame.groupby("account_id", sort=False):
            group = group.sort_values("timestamp")
            times = group.timestamp.astype("int64").to_numpy()
            amounts = group.amount.to_numpy(dtype=float)
            self.accounts[account] = (times, np.r_[0.0, amounts.cumsum()])

    def context(self, transaction_id: str):
        row = self.by_id.loc[transaction_id]
        times, sums = self.accounts[row.account_id]
        now = row.timestamp.value
        # Strictly earlier timestamps: simultaneous transactions cannot leak into each other.
        end = int(np.searchsorted(times, now, side="left"))
        return row, times, sums, now, end

    def history(self, transaction_id: str) -> dict:
        row, _, sums, _, end = self.context(transaction_id)
        mean = float(sums[end] / end) if end else 0.0
        return {"prior_count": end, "prior_mean_amount": mean,
                "current_to_mean_ratio": float(row.amount / mean) if mean > 0 else None}

    def velocity(self, transaction_id: str, window_hours: int = 24) -> dict:
        if window_hours not in (1, 24):
            raise ValueError("window_hours must be 1 or 24")
        _, times, sums, now, end = self.context(transaction_id)
        start = int(np.searchsorted(times, now - window_hours * 3600 * 10**9, side="left"))
        return {"window_hours": window_hours, "prior_count": end - start,
                "prior_total_amount": float(sums[end] - sums[start])}

    def features(self) -> pd.DataFrame:
        records = []
        for row in self.frame.itertuples(index=False):
            h = self.history(row.transaction_id)
            v1 = self.velocity(row.transaction_id, 1)
            v24 = self.velocity(row.transaction_id, 24)
            records.append([row.amount, row.timestamp.hour, h["prior_count"],
                            h["prior_mean_amount"], h["current_to_mean_ratio"] or 0.0,
                            v1["prior_count"], v1["prior_total_amount"],
                            v24["prior_count"], v24["prior_total_amount"]])
        return pd.DataFrame(records, columns=FEATURES, index=self.frame.index)

    def tools(self, case_id: str) -> list[StructuredTool]:
        def authorize(transaction_id):
            if transaction_id != case_id:
                raise ValueError("Tools are scoped to the active transaction")

        def account_history(transaction_id: str) -> dict:
            authorize(transaction_id)
            return self.history(transaction_id)

        def velocity(transaction_id: str, window_hours: int = 24) -> dict:
            authorize(transaction_id)
            return self.velocity(transaction_id, window_hours)

        return [
            StructuredTool.from_function(account_history, args_schema=CaseQuery,
                                         description="Prior account count, mean and amount ratio."),
            StructuredTool.from_function(velocity, args_schema=VelocityQuery,
                                         description="Prior account count and spend over 1 or 24 hours."),
        ]
