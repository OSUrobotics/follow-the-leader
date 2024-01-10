#!/usr/bin/env python3
import numpy as np

from typing import List, Union
import plotly.graph_objects as go
from bisect import bisect_left
import scipy.interpolate as si

import pprint

class BSplineCurve():
    def __init__(self, degree: str = "quadratic") -> None:
        self.pts: list = []
        self.r0: float = 0.0
        self.r1: float = 0.0
        self.r2: float = 0.0

        self.degree_dict = dict(
            linear=1,
            quadratic=2,
            cubic=3,
        )
        self.degree: int = self.degree_dict[degree] 

        self.basis_matrix = np.zeros(shape=(len(self.pts)+1, self.degree))
        return
    
    def add_point(self, point: np.ndarray) -> None:
        """Add a point to the sequence"""
        self.pts.append(point)
        return
    
    def add_points(self, points: np.ndarray) -> None:
        """Add a set of points to the sequence"""
        self.pts += points
        return
    
    def _pts_vec(self):
        """Find point residuals from line between t=0 and t=1"""

        return

    def fit_curve(self):
        """Fit a b-spline to the points"""
        return
    
    def fit_radius(self):
        """Fit a radius to the spline"""
        return
    
    def get_fit_error(self):
        return
    
    def inside_cylinder(self) -> bool:
        """Scoring function to test whether the curve falls inside the cylinder"""
        return
    
    def basis(self, i: int, t: float) -> float:
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

    def quadratic_bspline_control_points(self) -> np.ndarray:
        """Return the control points of a quadratic b-spline from the basis matrix
        @return ctrl_pts: np.ndarray - An array of three control points that define the b-spline"""
        ctrl_pts, residuals, rank, s = np.linalg.lstsq(a=self.basis_matrix, b=self.pts, rcond=None)
        return ctrl_pts
    
    def take_closest_t_idx(self, t: float):
        """Gets the closest t_k value used to make the basis matrix.
        Assumes self.t_k is sorted.
        """
        pos = bisect_left(self.t_ks, t)
        if pos == 0:
            return self.t_ks[0]
        if pos == len(self.t_ks):
            return self.t_ks[-1]
        before = self.t_ks[pos - 1]
        after = self.t_ks[pos]
        if (after - t) < (t - before):
            return pos
        else:
            return pos - 1

    def get_b_spline_representation(self):
        # print(np.array(self.pts).transpose())
        res = si.splprep(
            x = np.array(self.pts).transpose(),
            k=2,
        )
        pprint.pprint(res)

    def curve(self, t: float, pts: np.ndarray) -> np.ndarray:
        """Return the point on the curve at parameter t
        @param t: float - parameter
        @return 3d point"""
        print(self.basis_matrix)
        print(pts)

        t_idx = self.take_closest_t_idx(t)
        s_ = np.zeros(3)
        for i in range(len(pts)):
            s_ += pts[i] * self.basis_matrix[t_idx]
        print(s_)
        return s_
    
    def radius(self, t):
        """Return radius at a point t along the spline"""
        return

    def tangent_axis(self, t):
        """ Return the tangent vec 
        @param t in 0, 1
        @return 3d vec
        """
        vec_axis = [2 * t * (self.pt0[i] - 2.0 * self.pt1[i] + self.pt2[i]) - 2 * self.pt0[i] + 2 * self.pt1[i] for i in range(0, 3)]
        return np.array(vec_axis)

    def binormal_axis(self, t):
        """ Return the bi-normal vec, cross product of first and second derivative
        @param t in 0, 1
        @return 3d vec"""
        vec_tang = self.tangent_axis(t)
        vec_tang = vec_tang / np.linalg.norm(vec_tang)
        vec_second_deriv = np.array([2 * (self.pt0[i] - 2.0 * self.pt1[i] + self.pt2[i]) for i in range(0, 3)])

        vec_binormal = np.cross(vec_tang, vec_second_deriv)
        if np.isclose(np.linalg.norm(vec_second_deriv), 0.0):
            for i in range(0, 2):
                if not np.isclose(vec_tang[i], 0.0):
                    vec_binormal[i] = -vec_tang[(i+1)%3]
                    vec_binormal[(i+1)%3] = vec_tang[i]
                    vec_binormal[(i+2)%3] = 0.0
                    break

        return vec_binormal / np.linalg.norm(vec_binormal)
    
def plot_ctrl_pts(data, fig = None):
    if fig is None:
        fig = go.Figure()

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


def main():
    bs = BSplineCurve()
    bs.add_points([
        (0,0,0),
        (1,1,1),
        (2,2,1.5),
        (3,3, 2),
        (4,4,2.2)
    ])
    bs.get_b_spline_representation()

    bs.basis_mat()
    ctrl_pts = bs.quadratic_bspline_control_points()
    print(ctrl_pts)
    # # fig = plot_ctrl_pts(ctrl_pts)
    # fig = plot_data_pts(bs.pts)

    # # bs.curve(0.57, ctrl_pts)
    # fig.show()
    return

if __name__ == "__main__":
    main()