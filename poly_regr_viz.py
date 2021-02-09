from manimlib.imports import *
import autograd.numpy as np
from autograd import grad
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_SCALE = 14.2 / (2.9 + 5)
Y_SCALE = 8 / (1.3 + 4)

COUNT = 0

class MLVisuals(GraphScene):
    CONFIG = {
        "x_min" : 0,
        "x_max" : 5,
        "y_min" : -2,
        "y_max" : 2,
        "graph_origin" : ORIGIN + 4*LEFT,
        "function_color" : RED ,
        "axes_color" : GREY,
        "x_labeled_nums" :range(1,6,1),
        "y_labeled_nums" : [-2, -1, 0, 1, 2],
    }

    def construct(self):
        self.setup_axes()
        self.wait()
        data = load_data()
        regrs = run_regressions(data)
        x = data[:,:1]
        y = data[:,-1:]
        s, s_step = np.linspace(np.min(x), np.max(x), retstep=True)
        # stack with zeros for 3D convention
        point_data = np.hstack((x, y, np.zeros(x.shape)))
        point_group = VGroup(*[SmallDot(self.scale_data(point_data[i,:]),
                                        stroke_width=1,
                                        stroke_color=WHITE,
                                        fill_color=DARK_BLUE)
                                        for i in range(point_data.shape[0])])

        deg_num = []
        for i in range(1,10):
            obj = TexMobject("deg\ " + str(i))
            obj.shift(2*UP + 4*RIGHT)
            deg_num.append(obj)

        deg_term = VectorizedPoint(2*UP + 4*RIGHT)
        approx_graph = VectorizedPoint(self.scale_data(np.array([3.2,0,0])))

        graphs = []
        s_stack = np.vstack((s, s**2, s**3, s**4, s**5, s**6, s**7, s**8, s**9)).T
        s_list = s.tolist()
        for i in range(1,10):
            model_output = regrs[i - 1].predict(s_stack[:, :i])
            f = lambda t: self.anon_parametric(t, model_output, s_list)
            graphs.append(ParametricFunction(function=f, discontinuities=s_list[1:-1],
                               t_min=s_list[0], t_max=s_list[-1], step_size=s_step,
                               stroke_color=RED))

        self.play(FadeIn(point_group))

        for n, graph in enumerate(graphs):
            self.play(
                Transform(approx_graph, graph, run_time=0.6),
                Transform(deg_term, deg_num[n], run_time=0.6)
            )

    def scale_data(self, point):
        final = np.zeros((3))
        # x needs window scale factor, and a shift because of the
        # origin shift
        final[0] = point[0] * X_SCALE - 4
        # y just needs window scale factor
        final[1] = point[1] * Y_SCALE
        # z is zero but here for completeness
        final[2] = point[2]
        return final

    def anon_parametric(self, input, output, s):
        return self.scale_data(np.array([input, output[find_nearest(s, input)], 0.0]))

def load_data():
    # pull in data
    csvname = 'poly_regr.csv'
    data = np.loadtxt(csvname,delimiter=',')
    x = data[:,:-1]
    y = data[:,-1:]
    transformed_data = np.hstack((x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, y))
    return transformed_data

def run_regressions(transformed_data):
    train, test = train_test_split(transformed_data, train_size=0.8)
    ridge_runs = list()
    train_y = train[:,-1:]
    test_y = test[:,-1:]
    for i in range(9):
        ridge_runs.append(Ridge())
        train_x = train[:,:i + 1]
        test_x = test[:,:i + 1]
        ridge_runs[i].fit(train_x, train_y)
    return ridge_runs

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
