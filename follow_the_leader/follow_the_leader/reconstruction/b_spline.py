#!/usr/bin/env python3
import numpy as np

from typing import List, Tuple, Union
import plotly.graph_objects as go
from bisect import bisect_left
import scipy.interpolate as si

import pprint


"""
Resources:
https://mathworld.wolfram.com/B-Spline.html
https://core.ac.uk/download/pdf/82327690.pdf
https://stats.stackexchange.com/questions/517375/splines-relationship-of-knots-degree-and-degrees-of-freedom
"""

class BSplineCurve():
    def __init__(self, degree: str = "quadratic") -> None:
        self.data_pts: list = []
        self.ctrl_pts: list = []
    
        self.degree_dict = dict(
            linear=1,
            quadratic=2,
            cubic=3,
        )
        
        self.degree: int = self.degree_dict[degree]
        self.t: np.ndarray # Vector of knots
        self.c: np.ndarray # B-spline coefficients 
        self.k: int # degree of the spline

        # self.basis_matrix = np.zeros(shape=(len(self.pts)+1, self.degree))
        return
    
    def add_data_point(self, point: Union[Tuple[float], np.ndarray]) -> None:
        """Add a point to the sequence"""
        self.data_pts.append(point)
        return
    
    def add_data_points(self, points: np.ndarray) -> None:
        """Add a set of control points"""
        self.data_pts += points
        return
    
    def _pts_vec(self):
        """Find point residuals from line between t=0 and t=1"""
        slope_vec = self.data_pts[-1] - self.data_pts[0]
        z_intercept_vec = self.data_pts[0]
        return

    def fit_curve(self):
        """Fit a b-spline to the points"""
        return
    
    def eval_basis(self, i: int, t: float) -> float:
        """Return the basis function value for the ith control point at parameter t
        @param i - index of the control point
        @param t - parameter
        https://core.ac.uk/download/pdf/82327690.pdf
        NOTE: This can be cleaned up if needed. Written explicitly from source above.
        """
        if i <= t < i + 1:
            return ((t - i) / (i + 2 - i)) * ((t - i) / (i + 1 - i))
        elif i + 1 <= t < i + 2:
            return ((((i + 2) - t) / ((i + 2) - (i + 1))) * ((t - i) / ((i + 2) - i))) + ((((i + 3) - t) / ((i + 3) - (i + 1))) * ((t - (i + 1))) / ((i + 2) - (i + 1)))
        elif i + 2 <= t < i + 3:
            return ((i + 3 - t) / (i + 3 - (i + 1))) * ((i + 3 - t) / (i + 3 - (i + 2)))
        return 0.0

    def eval_crv(self, t: float) -> np.ndarray:
        """Evaluate the curve at parameter t
        @param t - parameter
        @return 3d point
        """
        res = self._eval_crv_at_zero(t=t)
        return res
    
    def _eval_crv_at_zero(self, t: float) -> np.ndarray:
        """Helper function to evaluate the curve at parameter t set from [0,1]
        @param t - parameter
        @return 3d point
        """
        idx = int(np.floor(t))
        t_prime = t - idx
        coefs = np.array(self.c).transpose()[idx]
        val = coefs[0] + coefs[1] * t_prime + coefs[2] * t_prime**2
        return val
    
    def derivative(self, t: float) -> np.ndarray:
        """Get the value of the derivative of the spline at parameter t"""
        return

    def basis_mat(self) -> None:
        """Set the basis matrix for the ith control point at parameter t
        @param i - index of the control point
        @param t - parameter
        """
        # divide into k+1 steps based on number of points
        k = len(self.pts)
        # parameterize t with equal steps TODO: add method for discretization by point distance
        self.t_ks = np.arange(0, 1, 1/k)
        self.basis_matrix = np.zeros(shape=(len(self.t_ks), self.degree+1))
        
        for _k, t_k in enumerate(self.t_ks):
            for i in range(self.degree+1):
                self.basis_matrix[_k, i] = self.basis(i, t_k+2)
        return
# 
    def quadratic_bspline_control_points(self) -> np.ndarray:
        """Return the control points of a quadratic b-spline from the basis matrix
        @return ctrl_pts: np.ndarray - An array of three control points that define the b-spline"""
        ctrl_pts, residuals, rank, s = np.linalg.lstsq(a=self.basis_matrix, b=self.pts, rcond=None)
        return ctrl_pts
    # 
    # def take_closest_t_idx(self, t: float):
    #     """Gets the closest t_k value used to make the basis matrix.
    #     Assumes self.t_k is sorted.
    #     """
    #     pos = bisect_left(self.t_ks, t)
    #     if pos == 0:
    #         return self.t_ks[0]
    #     if pos == len(self.t_ks):
    #         return self.t_ks[-1]
    #     before = self.t_ks[pos - 1]
    #     after = self.t_ks[pos]
    #     if (after - t) < (t - before):
    #         return pos
    #     else:
    #         return pos - 1

    # def curve(self, t: float, pts: np.ndarray) -> np.ndarray:
    #     """Return the point on the curve at parameter t
    #     @param t: float - parameter
    #     @return 3d point"""
    #     print(self.basis_matrix)
    #     print(pts)

    #     t_idx = self.take_closest_t_idx(t)
    #     s_ = np.zeros(3)
    #     for i in range(len(pts)):
    #         s_ += pts[i] * self.basis_matrix[t_idx]
    #     print(s_)
    #     return s_
    
    def get_spline_representation(self):
        """Return a tuple of the vector of knots, the spline coefficients, the spline degree, and the parameterization values"""
        # smoothing factor ~ square(acceptable arror) in m
        (self.t, self.c, self.k), u = si.splprep(np.array(self.data_pts).transpose(), k=self.degree, s=1e-6)
        return ((self.t, self.c, self.k), u)

    
    def b_spline_crv(self, u: Union[float, np.ndarray]):
        """Return a b-spline value"""
        u_fine = np.linspace(0,1,500)
        return si.splev(u_fine, (self.t, self.c, self.k)), si.splev(u, (self.t, self.c, self.k))
         
        
def plot_ctrl_pts(data, fig=None):
    if fig is None:
        fig = go.Figure()

    data = np.array(data)

    fig.add_trace(
        go.Scatter3d(
            x=data[:,0],
            y=data[:,1],
            z=data[:,2],
            mode="markers"
        )
    )
    return fig

def plot_data_pts(data, fig=None):
    if fig is None:
        fig = go.Figure()

    data = np.array(data)

    fig.add_trace(
        go.Scatter3d(
            x=data[:,0],
            y=data[:,1],
            z=data[:,2],
            mode="markers"
        )
    )
    return fig

def plot_spline(spline, fig=None):
    if fig is None:
        fig = go.Figure()

    spline = np.array(spline)

    fig.add_trace(
        go.Scatter3d(
            x=spline[0],
            y=spline[1],
            z=spline[2],
            mode="lines"
        )
    )
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
    bs = BSplineCurve(degree="quadratic")
    bs.add_data_points([
        [0,0,0],
        [0.5,0.5,1],
        [1,1,1],
        (2,2,1.5),
        (3,3, 2),
        (4,4,2.2)
    ])

    tck, u = bs.get_spline_representation()
    # pprint.pprint(np.array(tck[1]).transpose())
    spline, knots = bs.b_spline_crv(u)


    bs.eval_crv(t=0.5)

    # fig = plot_data_pts(bs.data_pts)
    # fig = plot_spline(spline, fig=fig)
    # fig = plot_knots(knots, fig=fig)
    # fig.show()
    return

if __name__ == "__main__":
    main()