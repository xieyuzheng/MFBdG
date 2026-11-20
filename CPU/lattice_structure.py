import numpy as np

SQUARE_BONDS = (
    ((1, 0), (0, 1)),
    ((1, 1), (1, -1)),
    ((2, 0), (0, 2)),
)
TRIANGLE_BASIS = np.array([[1., 0.], [0.5, np.sqrt(3)/2]])
TRIANGLE_BONDS = (
    ((1, 0), (-1, 1), (0, -1)),
    ((1, 1), (-2, 1), (1, -2)),
    ((2, 0), (-2, 2), (0, -2)),
)


