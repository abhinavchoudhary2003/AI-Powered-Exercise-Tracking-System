class PlankTracker:
    """
    Plank is a held position, not a rep count, so this tracks accumulated
    hold time instead of counting reps. Time only accrues while the body
    is in a valid straight-line plank position; breaking position pauses
    the clock rather than resetting it, so briefly losing form doesn't
    wipe out progress.
    """

    def __init__(self):
        self.hold_seconds = 0.0
        self.in_position = False
        self._last_update = None

    def update(self, is_plank_position, now):
        """
        is_plank_position: bool, computed by the caller from pose angles.
        now: time.time() from the caller, so this stays testable and in
        sync with the rest of session_manager's timing.

        Returns (hold_seconds, in_position).
        """
        if is_plank_position:
            if self.in_position and self._last_update is not None:
                self.hold_seconds += now - self._last_update
            self.in_position = True
            self._last_update = now
        else:
            self.in_position = False
            self._last_update = None

        return self.hold_seconds, self.in_position