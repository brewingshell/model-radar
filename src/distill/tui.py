from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from distill.models import Snapshot


def top_rows(snapshot: Snapshot, view_id: str = "performance-top5") -> list[dict[str, object]]:
    model_by_id = {model.model_id: model for model in snapshot.models}
    view = next((item for item in snapshot.views if item.view_id == view_id), None)
    if view is None:
        return []
    rows: list[dict[str, object]] = []
    for model_id in view.model_ids:
        model = model_by_id.get(model_id)
        if model is None:
            continue
        score_name = (
            "aa_token_dollar_efficiency"
            if view_id in {"performance-per-token-top5", "org-copilot-per-token-top10"}
            else "merged_benchmark_index"
            if view_id == "benchmark-synthesis-top10"
            else "intelligence_index"
        )
        score = model.scores.get(score_name)
        rows.append(
            {
                "model_id": model_id,
                "name": model.name,
                "score": score.value if score and score.value is not None else None,
            }
        )
    return rows


def plain_top(snapshot: Snapshot, view_id: str = "performance-top5") -> str:
    return "\n".join(
        f"{index}. {row['name']} ({row['score']})"
        for index, row in enumerate(top_rows(snapshot, view_id), start=1)
    )


class SnapshotApp(App[None]):
    def __init__(self, snapshot: Snapshot) -> None:
        super().__init__()
        self.snapshot = snapshot

    def compose(self) -> ComposeResult:
        yield Static(plain_top(self.snapshot))
