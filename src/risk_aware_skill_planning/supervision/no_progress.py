from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class NoProgressWindow:
    patience: int = 8
    threshold: float = 0.8
    values: deque[float] = field(default_factory=deque)

    def update(self, score: float) -> bool:
        if not 0.0 <= score <= 1.0:
            raise ValueError("no_progress score must lie in [0, 1]")
        self.values.append(score)
        while len(self.values) > self.patience:
            self.values.popleft()
        return len(self.values) == self.patience and all(value >= self.threshold for value in self.values)
