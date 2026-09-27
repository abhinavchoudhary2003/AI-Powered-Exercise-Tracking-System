# this is for pushup counts 
class PushUpCounter:

    def __init__(self):
        self.count = 0
        self.state = "UP"

    def update(self, elbow_angle, is_pushup_position):
        """
        Update push-up state using elbow angle,
        but only if the body is actually in a horizontal (plank) position.
        """

        # Ignore arm movement entirely if not in pushup posture
        if not is_pushup_position:
            return self.count, self.state

        if elbow_angle < 90:
            self.state = "DOWN"

        elif elbow_angle > 160 and self.state == "DOWN":
            self.count += 1
            self.state = "UP"

        return self.count, self.state