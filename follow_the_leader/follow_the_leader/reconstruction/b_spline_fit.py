#!/usr/bin/env python3
from typing import Tuple, Union
import numpy as np
from matplotlib import pyplot as plt

from b_spline import BSplineCurve
np.set_printoptions(precision=3, suppress=True)
"""
Resources:
https://mathworld.wolfram.com/B-Spline.html
https://xiaoxingchen.github.io/2020/03/02/bspline_in_so3/general_matrix_representation_for_bsplines.pdf
"""
class BSplineFit(BSplineCurve):
    # ROADMAP
    # cord length parameterization for t, find initial point
    # interpolate:
    #   1. repeat max thrice:
    #       - parameterize and make a simple "bezier curve", project pts to hull, convert to real t values
    # verify that t in order and should be half way in between
    #       - set up equations with x as control pts and solve
    #   2. determine if throw as outlier if too far off
    #   3. determine if split: resample
    #   4. determine if extend: find out if t > 1 global, try residual
    #      to fix local curve, solve the pts with the interpolate method for the excess pts:
    #           - use one or more values value from the past in the solver: use identity basis matrix to fix local cirve
    #           - use first derivatives
    #  5. stretch : resample
    #  local refits: split matrix

    def __init__(self, degree: str = "quadratic", dim: int = 2, ctrl_pts: list[np.ndarray] = [], data_pts: list[np.ndarray] = [], figax=None) -> None:
        """BSplineFit initialization

        :param degree: degree of spline, defaults to "quadratic"
        :param dim: dimension of spline, defaults to 2
        :param ctrl_pts: control points, defaults to []
        :param data_pts: points to fit, defaults to []
        :param figax: fig, ax tuple for interactive plotting, defaults to None
        """
        super().__init__(degree, dim, ctrl_pts, data_pts, figax)
        self.clicked: list[np.ndarray] = []
        self.residuals: np.ndarray = np.array([])
        self.parameter_normalization = 1.0
        self.ts = np.empty(0, dtype=float)

    def parameterize_chord(self, points: Union[list[np.ndarray], np.ndarray], renorm: bool = True) -> np.ndarray:
        """Get chord length parameterization of euclidean points

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        """
        distances = [np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points))]
        distances.insert(0, 0.)
        parameterized = np.cumsum(distances)
        if renorm:
            self.parameter_normalization = parameterized[-1] + 1e-3
        parameterized = parameterized / self.parameter_normalization
        return parameterized
 
    def setup_basic_lsq(self, ts: np.ndarray, num_ctrl_pts: Union[None, int] = None) -> np.ndarray:
        """Set up least squares problem for fitting a bspline curve to the parameterized points for 

        :param ts: points in t
        :type ts: np.ndarray
        :return: A matrix for the least squares problem
        :rtype: np.ndarray
        """
        if num_ctrl_pts is None:
            num_ctrl_pts = int(np.floor(ts[-1])) + self.degree + 1
        a_constraints = np.zeros((len(ts), num_ctrl_pts), dtype=float)
        for i in range(len(ts)):
            idx = int(np.floor(ts[i]))
            if idx + self.degree >= num_ctrl_pts:
                continue
            # print(f"i {i} idx {idx} for {ts[i]} a rn {a_constraints[i, idx: (idx + self.degree + 1)]} eval basis {self.eval_basis(ts[i])}")
            a_constraints[i, idx: (idx + self.degree + 1)] = self.eval_basis(ts[i])
        return a_constraints
    
    def simple_fit(self, points: list[np.ndarray]):
        """Fit a simple bezier curve, ie t = [0, 1) to the points

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        :return: control points, residuals, rank
        :rtype: Tuple[np.ndarray, np.ndarray, int]
        """
        points_in_t = self.parameterize_chord(points, renorm=True)
        print(f"{points} \n t: \n {points_in_t} \n a constr {self.setup_basic_lsq(points_in_t)}")
        ctrl_pts, residuals, rank, _ = np.linalg.lstsq(a=self.setup_basic_lsq(points_in_t), b=points, rcond=None)
        print(f"Residuals {residuals}, rank {rank}")
        return ctrl_pts, points_in_t, residuals, rank

    def extend_curve(self, new_data_pts: list[np.ndarray]):
        """Extend the curve to fit the new data points

        :param new_data_pts: new data points
        :type new_data_pts: list[np.ndarray]
        """
        print(f"extending curve of {len(self.ts)} by {len(new_data_pts)}")
        new_points_in_t = np.zeros(len(new_data_pts), dtype=float)
        points_in_t = np.zeros((len(self.ts) + len(new_data_pts)), dtype=float)
        old_points = np.reshape(self.data_pts, (-1, self.dim))
        b_constraints = np.zeros((len(points_in_t), self.dim), dtype=float)
        if len(new_data_pts) == 1:
            new_points_in_t = (np.linalg.norm(new_data_pts[0] - old_points[-1]) / self.parameter_normalization) + self.ts[-1]
        else:
            points = new_data_pts
            new_points_in_t = self.parameterize_chord(points) + self.ts[-1]
        points_in_t = np.hstack((self.ts, new_points_in_t))
        for i in range(len(self.ts)):
            b_constraints[i] = self.eval_crv(self.ts[i])
        b_constraints[len(self.ts):] = new_data_pts
        a_constraints = self.setup_basic_lsq(points_in_t)
        print(f"{b_constraints} \n t: \n {points_in_t} \n a constr {self.setup_basic_lsq(points_in_t)}")
        ctrl_pts, residuals, rank, _ = np.linalg.lstsq(a=self.setup_basic_lsq(points_in_t), b=b_constraints, rcond=None)
        print(b_constraints - np.dot(a_constraints, ctrl_pts))
        print(f"Extended residuals {residuals}, rank {rank}")
        return ctrl_pts, points_in_t, residuals, rank

    def plot_points(self, fig = None, ax = None):
        """plot clicked points with ctrl and data points. do not pass fig or ax for using existing canvas

        :param fig: mpl figure to draw on, defaults to None
        :param ax: mpl axes to use, defaults to None
        :return: (ctrl_point_line, spline_line)
        """
        if fig == None and ax == None and self.fig == None and self.ax == None:
            print("Atleast pass figure and ax!")
        elif fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
        ctrl_array = np.reshape(self.ctrl_pts, (-1, self.dim))
        data_array = np.reshape(self.data_pts, (-1, self.dim))
        clicked_array = np.reshape(self.clicked, (-1, self.dim))
        self.ax.plot(clicked_array[:, 0], clicked_array[:, 1], "bo", label="clicked points")
        self.ax.plot(ctrl_array[:, 0], ctrl_array[:, 1], "ro", label="control points")
        self.ax.plot(data_array[:, 0], data_array[:, 1], "go", label="data points")
        self.ax.plot([min(-2, min(clicked_array[:, 0]) - 5), max(10, max(clicked_array[:, 0]) + 5)], [0, 0], "-k")  # x axis
        self.ax.plot([0, 0], [min(-10, min(clicked_array[:, 1]) - 5), max(10, max(clicked_array[:, 1]) + 5)], "-k")
        self.ax.axis('equal')
        self.ax.grid()
        plt.draw()
               
    def plot_curve(self, fig = None, ax = None):
        """plot spline curve. do not pass fig or ax for using existing canvas

        :param fig: mpl figure to draw on, defaults to None
        :param ax: mpl axes to use, defaults to None
        """

        if fig == None and ax == None and self.fig == None and self.ax == None:
            print("Atleast pass figure and ax!")
        elif fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
        tr = np.linspace(0, len(self.ctrl_pts) - self.degree, 1000)
        print(f"now at {tr[-1]}")
        tr = tr[:-1]  # [0, 1) range
        spline = []
        for t in tr:
            spline.append(self.eval_crv(t=t))
        spline = np.array(spline)

        # print(f"{min(spline[:, 0])} to {max(spline[:, 0])} with {len(ctrl_array)} points")
        self.ax.plot(spline[:, 0], spline[:, 1],  label="spline")
        self.ax.axis('equal')
        self.ax.grid()
        plt.draw()

    def onclick(self, event):
        """manages matplotlib interactive plotting

        :param event: _description_
        """

        # print(type(event))
        if event.button==1: # projection on convex hull LEFT
            ix, iy = np.round(event.xdata, 3), np.round(event.ydata, 3)
            if ix == None or iy == None:
                print("You didn't actually select a point!")
                return
            self.clicked.append(np.array((ix, iy)))
            print(f"x {ix} y {iy} added")
            self.ax.clear()
            self.plot_points()
            if len(self.clicked) > self.degree:
                new_control_points, points_in_t, residuals, rank = self.simple_fit(self.clicked)
                print(f"Residuals {residuals} {(residuals < 1e-3)}, size {residuals.size}")
                if residuals.size == 0 or (residuals < 1e-3).all():
                    self.ctrl_pts = [new_control_points[i] for i in range(new_control_points.shape[0])]
                    self.ts = points_in_t
                    print(f"Control points {new_control_points}")
                    self.add_data_points(self.clicked)
                    self.residuals = residuals
                    if residuals.size == 0:
                        self.residuals = np.zeros(len(self.data_pts))
                else:
                    print("Residuals too high, extending curve")
                    # print(f"clicked {self.clicked} data {self.data_pts} sliced {self.clicked[-1]}")
                    self.parameterize_chord(self.data_pts, True) # TODO: i don't think this should work
                    new_control_points, points_in_t, residuals, rank = self.extend_curve([self.clicked[-1]])
                    if residuals.size == 0 or (residuals < 2.0).all():
                        self.ctrl_pts = [new_control_points[i] for i in range(new_control_points.shape[0])]
                        self.ts = points_in_t
                        print(f"Control points {new_control_points}")
                        self.add_data_point(self.clicked[-1])
                        self.residuals = residuals
                        if residuals.size == 0:
                            self.residuals = np.zeros(len(self.data_pts))
                        for i in range(len(self.ts)):
                            print(f"point: {self.ts[i]} {self.eval_crv(self.ts[i])}")
                        print(f"{self.clicked}")
                    else:
                        print("Residuals too high, not adding points")
                    self.ax.clear()
                self.plot_points() 
                self.plot_curve()

            
            self.plot_points()
        print("plotted")
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.draw()

    def enable_onclick(self):
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)

def main():
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    bs = BSplineFit(dim=2, ctrl_pts=[], degree="cubic", figax=(fig,ax))
    bs.enable_onclick()
    # bs.plot_basis(plt)
    # bs.plot_curve()
    ax.plot([min(-2, -5), max(10, 5)], [0, 0], "-k")  # x axis
    ax.plot([0, 0], [min(-10, -5), max(10, 5)], "-k")
    plt.show()
    fig.canvas.mpl_disconnect(bs.cid)
    return


if __name__ == "__main__":
    main()
