import numpy as np

#his method is just for angle calculation between body parts means we write the angle calculation once and reuse it for all exercises.
def calculate_angle(a, b, c):
    """
    Calculate the angle ABC in degrees.

    a, b, c should be points in the form:
    [x, y]
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vectors from the middle point
    ba = a - b
    bc = c - b

    # Calculate cosine of the angle
    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    # Prevent small floating-point errors
    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    # Convert radians to degrees
    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return angle
