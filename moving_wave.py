from manimlib.imports import *

class MovingWave(GraphScene):
    CONFIG = {
    "graph_origin" : ORIGIN + 4*LEFT,
    "x_min" : -1,
    "x_max" : 8,
    "y_min" : -1.5,
    "y_max" : 1.5,
    "amp" : 1,
    "rate" : PI/128,
    "t_offset" : 0,
    "graph_args" : {"x_min":-5, "x_max":4, "color":RED}
    }
    def construct(self):
        self.setup_axes()
        f = self.func_to_graph(0)
        self.play(ShowCreation(f))

        f.add_updater(self.update_func)
        self.add(f)

        self.wait(5)



    def update_func(self, f, dt):
        rate = self.rate + dt
        f.become(self.func_to_graph(self.t_offset + rate))
        self.t_offset += rate

    def func_to_graph(self, dx):
        N = 25
        return FunctionGraph(lambda x: self.amp * (1/N) * sum([np.cos(n * (x - dx)) for n in range(1,N)]),
                             **self.graph_args)

# def sum_range(n):
