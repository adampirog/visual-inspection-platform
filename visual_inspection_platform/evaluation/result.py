import json
import math
from collections import UserDict
from pathlib import Path

from prettytable import PrettyTable


METRIC_NAMES = {
    "pixel_auroc": "Pixel AUROC",
    "pixel_ap": "Pixel AP",
}


class EvaluationResult(UserDict):
    """Segmentation metrics grouped by evaluation type and category."""

    def __str__(self) -> str:
        if not self.data:
            return super().__str__()

        reports = []

        for section, rows in self.items():
            if not rows:
                continue

            metric_names = list(next(iter(rows.values())))
            table = PrettyTable()

            table.field_names = [
                "Category",
                *[METRIC_NAMES.get(name, name.replace("_", " ").title()) for name in metric_names],
            ]

            table.align["Category"] = "l"
            for column in table.field_names[1:]:
                table.align[column] = "r"

            for category, metrics in rows.items():
                label = "Macro avg" if category == "macro_avg" else category
                values = [
                    f"{metrics[name]:.4f}" if math.isfinite(metrics[name]) else "N/A"
                    for name in metric_names
                ]
                table.add_row([label, *values])

            reports.append(f"{section.title()} segmentation\n{table}")

        return "\n\n".join(reports)

    def save(self, path: str | Path) -> None:
        """Save evaluation results as .txt or .json."""
        path = Path(path)

        if path.suffix == ".txt":
            path.write_text(str(self), encoding="utf-8")
        elif path.suffix == ".json":
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=2, allow_nan=True)
        else:
            raise ValueError(f"Format '{path.suffix}' not supported")
