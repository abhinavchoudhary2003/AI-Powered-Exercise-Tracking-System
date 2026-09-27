# class for counting the squats
class SquatCounter:

    def __init__(self):
        self.count = 0
        self.state = "UP"

    def update(self, knee_angle, is_standing_position):
        """
        Update squat state using the knee angle, but only when the
        body is actually in a standing/squatting posture (vertical,
        not lying down or bent over like a pushup). If not in that
        posture, knee angle changes are ignored entirely.
        """

        # Ignore knee angle changes if body isn't in a standing posture
        if not is_standing_position:
            return self.count, self.state

        # Knee bends deeply → DOWN position (squatting)
        if knee_angle < 100:
            self.state = "DOWN"

        # Knee straightens again → completed squat
        elif knee_angle > 160 and self.state == "DOWN":
            self.count += 1
            self.state = "UP"

        return self.count, self.state


#     What this does
#
# Only when is_standing_position is True does knee angle matter.
#
# If the angle goes (while standing):
# 170°  → UP
# 130°  → UP
# 95°   → DOWN   (deep squat)
# 90°   → DOWN
# 110°  → DOWN
# 165°  → UP + COUNT
#
# The counter becomes:
# 1 squat