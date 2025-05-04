from typing import Dict, List
import plotext as plt

from rlms_mlr.loggers.stdout_logger import StdOutLogger


class StdOutGrapher(StdOutLogger):
    """
    Logger that renders real-time scalar graphs in the terminal using plotext.
    """
    def __init__(self, include: list[str]=None) -> None:
        super().__init__()
        self.metrics: Dict[str, List[float]] = {}
        self.steps: Dict[str, List[int]] = {}
        self.last_step = -1
        self.include = include or []

    def log_float(self, name: str, value: float, step: int) -> None:
        # Initialize history
        if len(self.include) and name not in self.include:
            return

        if name not in self.metrics:
            self.metrics[name] = []
            self.steps[name] = []

        # Append new data point
        self.metrics[name].append(value)
        self.steps[name].append(step)

        # Redraw once per new step
        if step != self.last_step:
            self.last_step = step
            self._render_graphs()

    def _render_graphs(self) -> None:
        # 1. Clear everything (subplots, data, styles)
        print("\033c", end="")
        print('\x1b[?25l', end='')
        plt.clear_figure()  # resets the entire canvas
        plt.canvas_color('none')  # removes all colors from the figure :contentReference[oaicite:3]{index=3}

        # 2. Determine a near-square grid
        n = len(self.metrics)
        rows = int(n ** 0.5) or 1
        cols = (n + rows - 1) // rows

        # 3. Define subplot layout
        plt.subplots(rows, cols)  # create an rows×cols matrix

        # 4. Plot each metric with no background fill
        for idx, (name, values) in enumerate(self.metrics.items()):
            row_idx = (idx // cols) + 1
            col_idx = (idx % cols) + 1

            plt.subplot(row_idx, col_idx)
            plt.clear_color()
            plt.plot(self.steps[name], values)
            plt.title(name)

        for r in range(row_idx, rows + 1):
            for c in range(1, cols + 1):
                plt.subplot(r, c)
                plt.clear_color()

        # 5. Render to terminal and throttle updates
        plt.show()
