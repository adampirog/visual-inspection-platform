import json
from collections import UserDict
from pathlib import Path

from prettytable import PrettyTable


class EvaluationResult(UserDict[str, dict[str, float]]):
    """Segmentation evaluation metrics grouped by prediction type."""

    def __str__(self) -> str:
        if not self.data:
            return super().__str__()

        table = PrettyTable()
        table.field_names = ["Group", "Metric", "Value"]
        table.align = "l"
        table.align["Value"] = "r"

        for group, metrics in self.items():
            for i, (name, value) in enumerate(metrics.items()):
                table.add_row(
                    [
                        group.title() if i == 0 else "",
                        name.replace("_", " ").title(),
                        f"{value:.4f}",
                    ]
                )

        return table.get_string()

    def save(self, path: str | Path) -> None:
        """Save as .txt or .json, inferred from the file extension."""
        path = Path(path)

        if path.suffix == ".txt":
            path.write_text(str(self), encoding="utf-8")
        elif path.suffix == ".json":
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=2)
        else:
            raise ValueError(f"Unsupported evaluation format: '{path.suffix}'")
