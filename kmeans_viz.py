from manimlib.imports import *
X_SCALE = 14.2 / 22
Y_SCALE = 8 / 21

class MLVisuals(GraphScene):
    CONFIG = {
        "x_min" : -7,
        "x_max" : 7,
        "y_min" : -2,
        "y_max" : 14,
        "graph_origin" : ORIGIN + 2*DOWN,
        "function_color" : RED ,
        "axes_color" : GREY,
        "x_labeled_nums" :range(-7,9,2),
        "y_labeled_nums" : range(2,16,2),
    }

    def construct(self):
        self.setup_axes()
        data = self.load_data()
        x = np.log(data[:-1,:])
        y = np.log(data[-1:,:])
        # stack with zeros for 3D convention
        data = np.vstack((x, y, np.zeros(x.shape)))
        points = []
        for i in range(data.shape[1]):
            points.append(self.convert_data_to_grid(data[:,i]))
        point_group = VGroup(*[SmallDot(point,
                                        stroke_width=1,
                                        stroke_color=WHITE,
                                        fill_color=DARK_BLUE)
                                        for point in points])


        w = np.random.rand(2) / 5 - 0.1
        weights, costs = gradient_descent(least_squares, 0.01, 1000, w, x, y)

        iter_num = []
        for i in iteration_range():
            obj = TexMobject("i = " + str(i))
            obj.shift(3*UP + 5*LEFT)
            iter_num.append(obj)

        iter_term = VectorizedPoint(3*UP + 5*LEFT)
        approx_graph = VectorizedPoint((w[0] * Y_SCALE - 2)* UP)

        graphs = []
        for i in iteration_range():
            graphs.append(Line(start=np.array([np.min(x) * X_SCALE, model(np.min(x), weights[i]) * Y_SCALE - 2, 0]),
                               end=np.array([np.max(x) * X_SCALE, model(np.max(x), weights[i]) * Y_SCALE - 2, 0]),
                               stroke_color=RED))

        self.play(FadeIn(point_group))

        for n, graph in enumerate(graphs):
            self.play(
                Transform(approx_graph, graph, run_time=0.3),
                Transform(iter_term, iter_num[n], run_time=0.3)
            )
