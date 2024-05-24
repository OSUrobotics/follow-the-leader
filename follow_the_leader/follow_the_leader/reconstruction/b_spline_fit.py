#!/usr/bin/env python3
from typing import Tuple, Union, List, Callable
import numpy as np
from matplotlib import pyplot as plt

from b_spline import BSplineCurve
import logging
from copy import deepcopy

logger = logging.getLogger("b_spline_fit")
logger.setLevel(level=logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

np.set_printoptions(precision=3, suppress=True)

"""
Resources:
https://mathworld.wolfram.com/B-Spline.html
https://xiaoxingchen.github.io/2020/03/02/bspline_in_so3/general_matrix_representation_for_bsplines.pdf
"""


class BSplineFit(BSplineCurve):
    # ROADMAP
    # todo: make static, not inherited
    # chord length parameterization for t, find initial point
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

    def __init__(
        self,
        degree: str = "quadratic",
        dim: int = 2,
        ctrl_pts: list[np.ndarray] = [],
        data_pts: list[np.ndarray] = [],
        figax=None,
        residuals: np.ndarray = np.array([]),
        parameter_normalization: float = 1.0,
        ts: np.ndarray = np.empty(0, dtype=float),
    ) -> None:
        """BSplineFit initialization

        :param degree: degree of spline, defaults to "quadratic"
        :param dim: dimension of spline, defaults to 2
        :param ctrl_pts: control points, defaults to []
        :param data_pts: points to fit, defaults to []
        :param figax: fig, ax tuple for interactive plotting, defaults to None
        """
        super().__init__(degree, dim, ctrl_pts, data_pts, figax)
        self.points_to_fit: list[np.ndarray] = []
        self.residuals: np.ndarray = residuals
        self.parameter_normalization = parameter_normalization
        self.ts = ts

    def parameterize_chord(
        self, points: Union[list[np.ndarray], np.ndarray], renorm: bool = False, start: int = 0, stop: int = 1
    ) -> np.ndarray:
        """Get chord length parameterization of euclidean points

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        :param renorm: renormalize the parameterization, defaults to False
        :type renorm: bool, optional
        :param start: start parameterization at t value, defaults to 0
        :type start: int, optional
        :param stop: stop parameterization at t value, defaults to 1
        :type stop: int, optional
        """
        distances = [
            np.linalg.norm(points[i] - points[i - 1]) for i in range(1, len(points))
        ]
        distances.insert(0, 0.0)
        parameterized = np.cumsum(distances)
        if renorm:
            self.parameter_normalization = (stop - start) / (parameterized[-1])
        parameterized = start + self.parameter_normalization * parameterized
        return parameterized

    def setup_basic_lsq(
        self, ts: np.ndarray, num_ctrl_pts: Union[None, int] = None
    ) -> np.ndarray:
        """Set up least squares problem for fitting a bspline curve to the parameterized points for
        arbitrary or minimum number of control points

        :param ts: points in t
        :type ts: np.ndarray
        :return: banded diagonal A matrix for the least squares problem
        :rtype: np.ndarray
        """
        if num_ctrl_pts is None:  # get minimum number of control points
            num_ctrl_pts = int(np.ceil(ts[-1])) + self.degree
        elif int(np.ceil(ts[-1])) + self.degree > num_ctrl_pts:
            raise ValueError("Number of control points too few for the parametrization")
        a_constraints = self.get_banded_matrix(ts, num_ctrl_pts)
        return a_constraints

    def renorm_fit(self, points: list[np.ndarray], start: int = 0, stop: int = 1):
        """Fit the curve to the points with renormalized parameterization

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        :param start: start parameterization at t value, defaults to 0
        :type start: int, optional
        :param stop: stop parameterization at t value, defaults to 1
        :type stop: int, optional
        """
        points_in_t = self.parameterize_chord(points, renorm=True, start=start, stop=stop)
        a_constraints = self.setup_basic_lsq(points_in_t, num_ctrl_pts=stop + self.degree)
        ctrl_pts, residuals, rank, _ = np.linalg.lstsq(
            a=a_constraints, b=points, rcond=None
        )
        residuals = np.linalg.norm(
            points - np.dot(a_constraints, ctrl_pts), axis=1
        )
        return ctrl_pts, points_in_t, residuals

    def bezier_fit(self, points: list[np.ndarray]):
        """Fit a simple bezier curve, ie t = [0, 1) to the points

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        :return: control points, residuals, rank
        :rtype: Tuple[np.ndarray, np.ndarray, int]
        """
        logger.info("attempt bezier fit")
        return self.renorm_fit(points, 0, 1)
    
    def simple_fit(self, points: list[np.ndarray]):
        """Fit the curve to the points with current parameterization

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        """
        points_in_t = self.parameterize_chord(points)
        a_constraints = self.setup_basic_lsq(points_in_t)
        ctrl_pts, residuals, rank, _ = np.linalg.lstsq(
            a=a_constraints, b=points, rcond=None
        )
        residuals = np.linalg.norm(
            points - np.dot(a_constraints, ctrl_pts), axis=1
        )
        return ctrl_pts, points_in_t, residuals

    def iteratively_fit(self, points: list[np.ndarray], outlier_ratio: float = 0.1, max_iter: int = 20, max_ctrl_ratio = 10):
        """Iteratively fit the curve to the points using RANSAC type approach

        :param points: x,y or xyz
        :type points: list[np.ndarray]
        """
        results = []
        # sample points
        for i in range(max_iter):
            sampled_points = np.random.sample(points, int(len(points) * outlier_ratio))
            ctrl_pts, _, __ = self.simple_fit(sampled_points)
            all_points_in_t = self.parameterize_chord(points)
            a_constraints = self.setup_basic_lsq(all_points_in_t)
            residuals = np.linalg.norm(
                points - np.dot(a_constraints, ctrl_pts), axis=1
            )
            results.append((ctrl_pts, all_points_in_t, residuals))
        # get best curve from results
        min_result = min(results, key=lambda x: np.sum(x[2]))
        return min_result

    def extend_curve(self, new_data_pts: list[np.ndarray]):
        """Extend the curve to fit the new data points
        :param new_data_pts: new data points
        :type new_data_pts: list[np.ndarray]
        """
        logger.debug(f"extending curve of {len(self.ts)} by {len(new_data_pts)}")
        new_points_in_t = np.zeros(len(new_data_pts), dtype=float)
        points_in_t = np.zeros((len(self.ts) + len(new_data_pts)), dtype=float)
        old_points = np.reshape(self.data_pts, (-1, self.dim))
        b_constraints = np.zeros((len(points_in_t), self.dim), dtype=float)
        if len(new_data_pts) == 1:
            new_points_in_t = (
                np.linalg.norm(new_data_pts[0] - old_points[-1]) * self.parameter_normalization
            ) + self.ts[-1]
            print(f"last {self.ts[-1], new_data_pts[0], self.data_pts}")
        else:
            points = new_data_pts
            new_points_in_t = self.parameterize_chord(points) + self.ts[-1]
        points_in_t = np.hstack((self.ts, new_points_in_t))
        b_constraints[: len(self.ts)] = self.eval_crv(self.ts)
        b_constraints[len(self.ts) :] = new_data_pts
        a_constraints = self.setup_basic_lsq(points_in_t)
        logger.debug(
            f" A = \n{a_constraints}\n B = \n{b_constraints} \n calculated using t: \n {points_in_t} \n"
        )
        ctrl_pts, residuals, rank, _ = np.linalg.lstsq(
            a=self.setup_basic_lsq(points_in_t), b=b_constraints, rcond=None
        )
        residuals = np.linalg.norm(
            b_constraints - np.dot(a_constraints, ctrl_pts), axis=1
        )
        diff_of_curves = (
            1 - abs(self.ctrl_hull_length - self.curve_length) / self.ctrl_hull_length
        )
        # if diff_curves is > 0.1 and residuals are low: overfitting
        logger.debug(
            f"Extended residuals {residuals}, rank {rank}, curve length diff {diff_of_curves}"
        )
        return ctrl_pts, points_in_t, residuals

    def plot_points(self, fig=None, ax=None):
        """plot clicked points with ctrl and data points. do not pass fig or ax for using existing canvas

        :param fig: mpl figure to draw on, defaults to None
        :param ax: mpl axes to use, defaults to None
        :return: (ctrl_point_line, spline_line)
        """
        logger.debug("plot_points")
        if fig == None and ax == None and self.fig == None and self.ax == None:
            logger.debug("Atleast pass figure and ax!")
        elif fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
        ctrl_array = np.reshape(self.ctrl_pts, (-1, self.dim))
        data_array = np.reshape(self.data_pts, (-1, self.dim))
        clicked_array = np.reshape(self.points_to_fit, (-1, self.dim))
        self.ax.plot(
            clicked_array[:, 0], clicked_array[:, 1], "bo", label="clicked points"
        )
        # self.ax.plot(ctrl_array[:, 0], ctrl_array[:, 1], "ro", label="control points")
        self.ax.plot(data_array[:, 0], data_array[:, 1], "go", label="data points")
        self.ax.plot(
            [
                min(-2, min(clicked_array[:, 0]) - 5),
                max(10, max(clicked_array[:, 0]) + 5),
            ],
            [0, 0],
            "-k",
        )  # x axis
        self.ax.plot(
            [0, 0],
            [
                min(-10, min(clicked_array[:, 1]) - 5),
                max(10, max(clicked_array[:, 1]) + 5),
            ],
            "-k",
        )
        self.ax.axis("equal")
        self.ax.grid()
        plt.draw()

    def plot_curve(self, fig=None, ax=None):
        """plot spline curve. do not pass fig or ax for using existing canvas

        :param fig: mpl figure to draw on, defaults to None
        :param ax: mpl axes to use, defaults to None
        """
        logger.debug("plot_curve")
        if fig == None and ax == None and self.fig == None and self.ax == None:
            logger.error("Atleast pass figure and ax!")
        elif fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
        tr = np.linspace(0, self.ts[-1], 1000)
        spline = self.eval_crv(tr)
        logger.debug(f"{spline}")
        logger.debug(f"{min(spline[:, 0])} to {max(spline[:, 0])} with {len(self.ctrl_pts)} points")
        self.ax.plot(spline[:, 0], spline[:, 1], label="spline")
        self.ax.axis("equal")
        self.ax.grid()
        plt.draw()

    def evaluate(
        self, result, residual_min=1e-6, residual_max=1e-2, max_control_ratio=10
    ) -> tuple[bool, Union[BSplineCurve, Tuple[Callable, List]]]:
        """Evaluate the result of the fit and recommend next steps if not satisfactory

        :param result: tuple of control points, points in t and residuals
        :param residual_threshold: residual threshold per segment, defaults to 0.1
        :param max_control_points: _description_, defaults to 30
        :return: whether to
        """
        ctrl_pts, points_in_t, residuals = result
        min_t = int(np.floor(min(points_in_t)))
        max_t = int(np.ceil(max(points_in_t)))
        logger.debug(f"minmax: {min_t, max_t}")

        # overfitting
        if len(self.points_to_fit) / max_control_ratio > len(ctrl_pts):
            logger.warning(f"Too many control points {len(ctrl_pts)} for data points {len(self.data_pts)}")
            return False, (self.renorm_fit, [self.data_pts, min_t, max_t - 1])
    
        # bucket residuals
        t_buckets = np.array(range(min_t, max_t + 1))

        segmentwise_residual = []
        for i in range(len(t_buckets) - 1):
            cond = np.asarray((points_in_t >= t_buckets[i]) & (points_in_t < t_buckets[i + 1]))
            segmentwise_residual.append(
                np.sum(residuals[cond.nonzero()])
            )
        logger.debug(f"{segmentwise_residual, t_buckets}")
        segmentwise_residual = np.array(segmentwise_residual)
        max_residual = segmentwise_residual < residual_max
        min_residual = segmentwise_residual > residual_min
        check_residual = max_residual & min_residual
        if check_residual.all() or (max_t == 1 and max_residual.all()):
            new_spline = deepcopy(self)
            new_spline.ctrl_pts = [ctrl_pts[i] for i in range(ctrl_pts.shape[0])]
            new_spline.ts = points_in_t
            new_spline.residuals = residuals
            logger.debug(f"Valid spline with residuals {segmentwise_residual}")
            return True, new_spline
        elif not max_residual.all():
            indices = np.asarray(max_residual == False).nonzero()
            logger.info(f"Residuals too high in segments {t_buckets[indices]}")
            # if residuals high towards the end, extend curve
            if np.mean(t_buckets[indices]) > 0.8 * (max_t - 1):
                return False, (self.extend_curve, t_buckets[indices])
            else: # add another control point
                return False, (self.renorm_fit, [self.data_pts, min_t, max_t + 1])
        # elif not min_residual.all(): # overfitting
        #     return False, (self.renorm_fit, [self.data_pts, min_t, max_t - 1])
        

    def onclick(self, event):
        """manages matplotlib interactive plotting

        :param event: _description_
        """

        # print(type(event))
        if event.button == 1:  # projection on convex hull LEFT
            ix, iy = np.round(event.xdata, 3), np.round(event.ydata, 3)
            if ix == None or iy == None:
                logger.warning("You didn't actually select a point!")
                return
            self.points_to_fit.append(np.array((ix, iy)))
            logger.info(f"x {ix} y {iy} added")
            self.ax.clear()
            self.plot_points()
            if len(self.points_to_fit) > self.degree:
                if not self.is_initialized:
                    new_control_points, points_in_t, residuals = self.bezier_fit(
                        self.points_to_fit
                    )
                else:
                    new_control_points, points_in_t, residuals = self.renorm_fit(
                        self.points_to_fit, 0, self.max_t
                    )
                good, func = self.evaluate((new_control_points, points_in_t, residuals))
                if good:
                    self.__dict__.update(func.__dict__)
                    logger.debug(f"Control points {self.ctrl_pts}")
                    self.data_pts = self.points_to_fit

                else:
                    logger.info("Residuals too high, extending curve")
                    new_control_points, points_in_t, residuals = self.extend_curve([self.points_to_fit[-1]])
                    
                    if residuals.size == 0 or (residuals < 2.0).all():
                        self.ctrl_pts = [
                            new_control_points[i]
                            for i in range(new_control_points.shape[0])
                        ]
                        self.ts = points_in_t
                        logger.debug(f"Control points {new_control_points}")
                        self.add_data_point(self.points_to_fit[-1])
                        self.residuals = residuals
                        self.residuals = np.zeros(len(self.data_pts))
                        logger.debug(
                            f"ts: {self.ts} now eval to \n{self.eval_crv(self.ts)}\n"
                            f"for original \n{self.unflatten_dim(self.points_to_fit, self.dim)}"
                        )
                    else:
                        logger.warning("Residuals too high, not adding points")
                    self.ax.clear()
                self.plot_points()
                self.plot_curve()

            self.plot_points()
        logger.debug("plotted")
        plt.axis("equal")
        plt.grid()
        plt.legend()
        plt.draw()

    def enable_onclick(self):
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)


def plot_test():
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    bs = BSplineFit(dim=2, ctrl_pts=[], degree="cubic", figax=(fig, ax))
    bs.enable_onclick()
    # bs.plot_basis(plt)
    # bs.plot_curve()
    ax.plot([min(-2, -5), max(10, 5)], [0, 0], "-k")  # x axis
    ax.plot([0, 0], [min(-10, -5), max(10, 5)], "-k")
    plt.show()
    fig.canvas.mpl_disconnect(bs.cid)
    return


if __name__ == "__main__":
    plot_test()
