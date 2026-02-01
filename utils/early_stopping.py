
from typing import Optional


class EarlyStopping:

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max", verbose: bool = True):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.verbose = bool(verbose)

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_epoch = 0

        if self.mode == "max":
            self.best_score = float("-inf")
        else:
            self.best_score = float("inf")

    def __call__(self, current_score: float, epoch: int = 0) -> bool:
        is_best = False

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            if self.verbose:
                print(f"  EarlyStopping: improved {self.best_score:.4f} -> {current_score:.4f}")
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            is_best = True
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: no improve {self.counter}/{self.patience} (best={self.best_score:.4f})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("  EarlyStopping: STOP")

        return is_best

    def state_dict(self) -> dict:
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, d: dict) -> None:
        self.counter = d.get("counter", 0)
        self.best_score = d.get("best_score", self.best_score)
        self.early_stop = d.get("early_stop", False)
        self.best_epoch = d.get("best_epoch", 0)
