#!/usr/bin/env python3
from copy import deepcopy
import numpy as np
from numpy.polynomial import Polynomial as P

from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# from bisect import bisect_left
# import scipy.interpolate as si
from scipy.optimize import fmin, least_squares
from geom_utils import LineSeg2D, ControlHull
import pprint

"""
Resources:
https://mathworld.wolfram.com/B-Spline.html
https://xiaoxingchen.github.io/2020/03/02/bspline_in_so3/general_matrix_representation_for_bsplines.pdf
"""
class BSplineCurve(object):
    degree_dict = dict(
            linear=1,
            quadratic=2,
            cubic=3,)

    # uniform knot vector only
    # General Matrix Representations for B-Splines, Kaihuai Qin
    # each matrix column represents a series of coeffs of increasing degree for a basis function 
    basis_matrix_dict = {
    0: np.array([1]),
    1: np.array([[1, 0], [-1, 0]]),
    2: 1/2 * np.array([[1, 1, 0], 
                       [-2, 2, 0],
                       [1, -2, 1]]),
    3: 1/6 * np.array([[1, 4, 1, 0],
                       [-3, 0, 3, 0],
                       [3, -6, 3, 0], 
                       [-1, 3, -3, 1]]),
    }
    derivative_dict = {
    0: np.array([0]),
    1: np.array([[0, 0], [-1, 0]]),
    2: 1/2 * np.array([[0, 1, 0],
                       [-2, 2, 0],
                       [2, -4, 2]]),
    3: 1/6 * np.array([[1, 4, 1, 0],
                       [-3, 0, 3, 0],
                       [6, -12, 6, 0], 
                       [-3, 9, -9, 3]]),
    }

    # ROADMAP
    # curve binomial for frenet frames
    # projectToCtrlHull -> projectToPolyLine -> project to segment: tells us we need extension
    # projectToCurve -> use t from ctr hull, convert to polynomial eval, needs fmin
    # apply a filter -> move to middle
    # local refit: split matrix

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
    #  integrate radius of curvature over the curve as metric for curviness

    def __init__(self, degree: str = "quadratic", dim: int = 2, ctrl_pts: list[np.ndarray] = [np.array([0, 0])], figax=None) -> None:
        """BSpline initialization

        :param degree: degree of spline, defaults to "quadratic"
        :param dim: dimension of spline, defaults to 2
        :param ctrl_pts: control points, defaults to [0, 0]
        :param figax: fig, ax tuple for interactive plotting, defaults to None
        """
        self.data_pts: list[np.ndarray] = None
        self.ctrl_pts: list[np.ndarray] = deepcopy(ctrl_pts)  # make per dim list
        if ctrl_pts is not None:
            if len(ctrl_pts[0]) != dim:
                raise ValueError("Mismatch in control point dimension and initialized dim!")
        self.dim = dim
        self.degree: int = self.degree_dict[degree]
        self.basis_matrix: np.ndarray = BSplineCurve.basis_matrix_dict[
            self.degree
        ]  # B-spline coefficients

        # get basis polynomials
        self.basis_poly = [P(self.basis_matrix[:, j]) for j in range(self.degree + 1)]

        self.deriv_matrix: np.ndarray = BSplineCurve.derivative_dict[self.degree]
        self.curv_eqn: np.ndarray = None

        # for interactive plotting
        self.fig = None
        self.ax = None
        if figax is not None:
            self.fig, self.ax = figax
            self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        return

    def _generate_power_series(self, t) -> np.ndarray:
        # print(f"trying to generate power series with t {t}")
        if type(t) == np.ndarray:
            t = t[0]
        power_series = [1]
        for i in range(1, self.degree + 1):
            power_series.append(t**i)
        # print(f'power series {power_series}')
        return np.array(power_series)

    def add_data_point(self, point: Union[Tuple[float], np.ndarray]) -> None:
        """Add a point to the sequence"""
        if len(point) != self.dim:
            raise ValueError(f"Bad point dimension! Existing is {self.dim}, we got {len(point)}")
        self.data_pts.append(point)
        return

    def add_data_points(self, points: np.ndarray) -> None:
        """Add a set of data points"""
        if len(points[0]) != self.dim:
            raise ValueError(f"Bad point dimension! Existing is {self.dim}, we got {points.shape[1]}")
        self.data_pts.extend(points)
        return

    def add_ctrl_point(self, point: Union[Tuple[float], np.ndarray]) -> None:
        """Add a control point to the sequence"""
        if len(point) != self.dim:
            raise ValueError(f"Bad point dimension! Existing is {self.dim}, we got {len(point)}")
        self.ctrl_pts.append(point)
        return

    def eval_basis(self, t) -> np.ndarray:
        idx = int(np.floor(t))
        t_prime = t - float(idx)
        # print("new", [self.basis_poly[i](t_prime) for i in range(self.degree + 1)])
        return [self.basis_poly[i](t_prime) for i in range(self.degree + 1)]

    def plot_basis(self, plt):
        """Plots the basis function in [0, 1)]"""
        tr = np.linspace(0, 1.0, 100)
        tr = tr[0:-2]  # [0, 1) range
        basis = []
        for t in tr:
            basis.append(self.eval_basis(t=t))
        basis = np.array(basis)
        for i in range(0, self.degree + 1):
            plt.scatter(tr + (self.degree - i), basis[:, i])
        plt.xlabel("t values")
        plt.show()

    def _eval_crv_at_zero(self, t: float) -> np.ndarray:
        """Helper function to evaluate the curve at parameter t set from [0,1)
        @param t - parameter
        @return 3d point
        """
        idx = int(np.floor(t))
        t_prime = t - float(idx)
        # print(f"t {t}, idx {idx}")

        if self.ctrl_pts is None or len(self.ctrl_pts) <= self.degree:
            raise ValueError(
                f"Need atleast degree + 1: {self.degree + 1} control points for creating a bezier curve"
            )
        if len(self.ctrl_pts) + self.degree <= idx:
            raise ValueError(f"Not enough ctrl pts to get segment at index {idx}")
        try:
            return np.matmul(
                self.eval_basis(t_prime),
                np.reshape(self.ctrl_pts, (-1, self.dim))[idx - self.degree : idx + 1, :],
            )
        except ValueError as v:
            print(self.basis_matrix,
                np.reshape(self.ctrl_pts, (-1, self.dim))[idx - self.degree : idx + 1, :])
            raise ValueError(f"Something went really wrong! {v}")

    def eval_crv(self, t: float) -> np.ndarray:
        """Evaluate the curve at parameter t
        @param t - parameter
        @return 3d point
        """
        res = self._eval_crv_at_zero(t=t)
        return res
    
    def min_crv(self, t: np.ndarray, pt) -> np.ndarray:
        """Evaluate the curve at parameter t
        @param t - parameter
        @return 3d point
        """
        res = self._eval_crv_at_zero(t=t[0])
        return np.sum((res - pt) ** 2)

    def project_ctrl_hull(self, pt) -> float:
        """ Get t value for projecting point on hull

        :param pt: point to project
        :return: t value
        """
        self.hull = ControlHull(self.ctrl_pts)
        
        # just drawing the hull
        x_ = []
        y_ = []
        for pairs in self.hull.polylines:
            # print(simplex[0])
            # print(self.ctrl_pts, self.ctrl_pts[simplex[0]])
            x_seg = [self.ctrl_pts[pairs[0]][0], self.ctrl_pts[pairs[1]][0]]
            y_seg = [self.ctrl_pts[pairs[0]][1], self.ctrl_pts[pairs[1]][1]]
            x_.extend(x_seg)
            y_.extend(y_seg)
        self.ax.plot(x_, y_, "-g", label="control hull")
    
        t, pt_proj, min_seg = self.hull.parameteric_project(pt)
        t_reindex = t + min_seg[0] + self.degree
        
        # just drawing the line segments and prjections
        self.ax.plot([pt_proj[0], pt[0]], [pt_proj[1], pt[1]], marker="D",label=f"t_val: {t_reindex}", color="orange")
        x_seg = [self.ctrl_pts[min_seg[0]][0], self.ctrl_pts[min_seg[1]][0]]
        y_seg = [self.ctrl_pts[min_seg[0]][1], self.ctrl_pts[min_seg[1]][1]]
        self.ax.plot(x_seg, y_seg, "-r", label="current")
        
        print(f"original {t}")
        return t_reindex
    
    def project_to_curve(self, pt):
        """Project a point on the current spline

        :param pt: point to project
        """
        t = self.project_ctrl_hull(pt)
        result = fmin(self.min_crv, np.array(t), args=(pt, )) # limit to 10 TODO
        proj_pt = self.eval_crv(result.x)
        print(f"proj pt at real value {proj_pt}")
        return pt

    def _pts_vec(self):
        """Find point residuals from line between t=0 and t=1"""
        slope_vec = self.data_pts[-1] - self.data_pts[0]
        z_intercept_vec = self.data_pts[0]
        return

    def fit_curve(self):
        """Fit a b-spline to the points"""
        return

    def derivative(self, t: float) -> np.ndarray:
        """Get the value of the derivative of the spline at parameter t"""
        return

    def plot_curve(self, fig = None, ax = None):
        """plot spline curve. do not pass fig or ax for using existing canvas

        :param fig: mpl figure to draw on, defaults to None
        :param ax: mpl axes to use, defaults to None
        :return: (ctrl_point_line, spline_line)
        """

        if fig == None and ax == None and self.fig == None and self.ax == None:
            print("Atleast pass figure and ax!")
        elif fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
        self.ax.clear()
        tr = np.linspace(self.degree, len(self.ctrl_pts), 1000)
        tr = tr[0:-2]  # [0, 1) range
        spline = []
        for t in tr:
            spline.append(self.eval_crv(t=t))
        spline = np.array(spline)
        ctrl_array = np.reshape(self.ctrl_pts, (-1, self.dim))
        # print(f"{min(spline[:, 0])} to {max(spline[:, 0])} with {len(ctrl_array)} points")
        (ln,) = self.ax.plot(ctrl_array[:, 0], ctrl_array[:, 1], "bo", label="control points")
        (ln2,) = self.ax.plot(spline[:, 0], spline[:, 1],  label="spline")
        self.ax.plot([min(-2, min(ctrl_array[:, 0] - 5)), max(10, max(ctrl_array[:, 0] + 5))], [0, 0], "-k")  # x axis
        self.ax.plot([0, 0], [min(-10, min(ctrl_array[:, 1] - 5)), max(10, max(ctrl_array[:, 1] + 5))], "-k")
        self.ax.grid()
        plt.draw()
        return ln, ln2

    def onclick(self, event):
        """manages matplotlib interactive plotting

        :param event: _description_
        """
        # print(type(event))
        if event.button==1: # projection on convex hull LEFT
            ix, iy = event.xdata, event.ydata
            if ix == None or iy == None:
                print("You didn't actually select a point!")
                return
            print(f"projecting x {ix} y {iy}")
            self.project_ctrl_hull((ix, iy))
            # self.project_to_curve((ix, iy))
        elif event.button==3: # add control point RIGHT
            ix, iy = event.xdata, event.ydata
            self.add_ctrl_point((ix, iy))
            if ix == None or iy == None:
                print("You didn't actually select a point!")
                return
            print(f"x {ix} y {iy} added")
            self.plot_curve()
        print("plotted")
        plt.grid()
        plt.legend()
        plt.draw()


    # def quadratic_bspline_control_points(self) -> np.ndarray:
    #     """Return the control points of a quadratic b-spline from the basis matrix
    #     @return ctrl_pts: np.ndarray - An array of three control points that define the b-spline"""
    #     ctrl_pts, residuals, rank, s = np.linalg.lstsq(a=self.basis_matrix, b=self.pts, rcond=None)
    #     return ctrl_pts


def plot_ctrl_pts(data, fig=None):
    if fig is None:
        fig = go.Figure()

    data = np.array(data)

    fig.add_trace(
        go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], mode="markers")
    )
    return fig


def plot_data_pts(data, fig=None):
    if fig is None:
        fig = go.Figure()

    data = np.array(data)

    fig.add_trace(
        go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], mode="markers")
    )
    return fig


def plot_spline(spline, fig=None):
    if fig is None:
        fig = go.Figure()

    spline = np.array(spline)

    fig.add_trace(go.Scatter3d(x=spline[0], y=spline[1], z=spline[2], mode="lines"))
    return fig


def plot_knots(knots, fig=None):
    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=knots[0],
            y=knots[1],
            z=knots[2],
            marker=dict(
                size=8,
            ),
        )
    )
    return fig


def main():
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    bs = BSplineCurve(ctrl_pts=[np.array([0, 0]), np.array([3, 5]), np.array([6, -5]), np.array([6.5, -3])], degree="quadratic", figax=(fig,ax))
    # bs.plot_basis(plt)
    bs.plot_curve()
    plt.show()
    fig.canvas.mpl_disconnect(bs.cid)
    return


if __name__ == "__main__":
    main()
