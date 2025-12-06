import numpy as np

class WorldCamera:
    def __init__(self):
        self.position = np.array((-54.88611607,  -36.96915122, -185.77141157))
        self.orientation = np.array((0.29889257, 0.21649844, 0.92755202))
        self.up = np.array((-0.74086449 , 0.5812467 ,  0.33646813))
        self.left = np.cross(self.up, self.orientation)