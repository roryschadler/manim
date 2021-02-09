from manimlib.imports import *

class Scratch(Scene):
    def construct(self):
        line=Line(np.array([3,0,1]),np.array([5,2,0]))
        self.play(ApplyPointwiseFunction(
            lambda points: np.array([(1, 0, 0), (0, 2, 0), (0, 0, 1)]).dot(points),
            line))
