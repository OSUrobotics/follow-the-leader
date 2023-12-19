#!/usr/bin/env python3
import numpy as np

from typing import List, Union
import plotly.graph_objects as go

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

    def fit_curve(self):
        """Fit a b-spline to the points"""
        return
    
    def fit_radius(self):
        """Fit a radius to the spline"""
    
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
        t_ks = np.arange(0, 1, 1/k)
        self.basis_matrix = np.zeros(shape=(len(t_ks), self.degree+1))
        
        for _k, t_k in enumerate(t_ks):
            for i in range(self.degree+1):
                self.basis_matrix[_k, i] = self.basis(i, t_k+2)
        return

    def quadratic_bspline_control_points(self) -> np.ndarray:
        """Return the control points of a quadratic b-spline from the basis matrix"""
        ctrl_pts, residuals, rank, s = np.linalg.lstsq(a=self.basis_matrix, b=self.pts, rcond=None)
        return ctrl_pts
    
    def curve(self, t: float, pts) -> np.ndarray:
        """Return the point on the curve at parameter t
        @param t: float - parameter
        @return 3d point"""
        print(self.basis_matrix)
        t_idx = 
        res = self.basis_matrix[4] @ pts
        return res
    
    def radius(self, t):
        """Return radius at a point t along the spline"""
        return

    def pt_axis(self, t):
        """ Return a point along the bezier
        @param t in 0, 1
        @return 2 or 3d point"""
        self.pt0 = self.pts[0]
        self.pt1 = self.pts[1]
        self.pt2 = self.pts[2]
        pts_axis = np.array([self.pt0[i] * (1-t) ** 2 + 2 * (1-t) * t * self.pt1[i] + t ** 2 * self.pt2[i] for i in range(0, 3)])
        return pts_axis.transpose()

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
        
    def frenet_frame(self, t):
        """ Return the matrix that will take the point 0,0,0 to crv(t) with x axis along tangent, y along binormal
        @param t - t value
        @return 4x4 transformation matrix"""
        pt_center = self.pt_axis(t)
        vec_tang = self.tangent_axis(t)
        vec_tang = vec_tang / np.linalg.norm(vec_tang)
        vec_binormal = self.binormal_axis(t)
        vec_x = np.cross(vec_tang, vec_binormal)

        mat = np.identity(4)
        mat[0:3, 3] = pt_center[0:3]
        mat[0:3, 0] = vec_x.transpose()
        mat[0:3, 1] = vec_binormal.transpose()
        mat[0:3, 2] = vec_tang.transpose()

        return mat
    
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
        (-5, 25, 25),
        (-4, 16, 16),
        (-3, 9, 9),
        (-2, 4, 4),
        (-1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
        (2, 4, 4),
        (3, 9, 9),
        (4, 16, 16)
    ])

    bs.basis_mat()
    bs.ctrl_pts = bs.quadratic_bspline_control_points()
    fig = plot_ctrl_pts(bs.ctrl_pts)
    fig = plot_data_pts(bs.pts, fig=fig)

    print(bs.curve(0.5, bs.ctrl_pts))
    # fig.show()
    return

if __name__ == "__main__":
    main()