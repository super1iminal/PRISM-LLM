"""When the refinement loop throws its feedback away and starts again from the initial prompt."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Parsed from `planner.retry`:

        stall:k   after k consecutive rounds that did not improve the kept policy (default stall:2)
        never     always continue with feedback
        every:k   at rounds 1 + k, 1 + 2k, ... regardless of progress (every:3 -> rounds 4, 7, ...)
        gain:eps  when the last round reduced the kept policy's total worst-case shortfall by less than eps
        always    every round is a fresh initial prompt (pure resampling with keep-best)
    """
    kind: str
    value: float = 0.0

    @classmethod
    def parse(cls, text: str) -> "RetryPolicy":
        kind, _, arg = str(text).partition(":")
        if kind in ("never", "always") and not arg:
            return cls(kind)
        if kind in ("stall", "every") and arg.isdigit() and int(arg) >= 1:
            return cls(kind, int(arg))
        if kind == "gain" and arg:
            try:
                return cls(kind, float(arg))
            except ValueError:
                pass
        raise ValueError(f"invalid retry policy {text!r} (stall:k | never | every:k | gain:eps | always)")

    def restart(self, next_round: int, stall: int, gain: float) -> bool:
        """Should round `next_round` (>= 2) start from the initial prompt?

        `stall` is the number of consecutive rounds without improvement so far; `gain` is how much the
        last round reduced the kept policy's total worst-case shortfall (0 if it did not improve).
        """
        if self.kind == "always":
            return True
        if self.kind == "never":
            return False
        if self.kind == "stall":
            return stall >= self.value
        if self.kind == "every":
            return (next_round - 1) % int(self.value) == 0
        return gain < self.value   # gain:eps
