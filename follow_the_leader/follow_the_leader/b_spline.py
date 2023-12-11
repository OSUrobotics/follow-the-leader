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

        self.basis_matrix = np.zeros(shape=(self.degree, self.degree))
        return
    
    def add_point(self, point: np.ndarray) -> None:
        """Add a point to the sequnce"""
        self.pts.append(point)
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
    
    # P(t) = sum(b_ik(t) * p_i)
    # where b_ik(t) is the basis function
    # and p_i is the control point
    # and t is the parameter
    # and k is the degree of the curve
    # and i is the index of the control point
    # and n is the number of control points

    # b_ik(t) = (t - t_i) / (t_(i+k) - t_i) * b_i,k-1(t) + (t_(i+k+1) - t) / (t_(i+k+1) - t_(i+1)) * b_i+1,k-1(t)
    # b_i1(t) = 1 if t_i <= t < t_(i+1) else 0
    # b_ik(t) = 0 if t < t_i or t >= t_(i+k+1)

    def quadratic_bspline_control_points(self, end_pts: np.ndarray) -> np.ndarray:
        """Return the control points of a quadratic b-spline given the end points
        @param end_pts - 2x3 matrix of end points
        @return 3x3 matrix of control points"""


        return

    def cubic_bspline_control_points(self, end_pts: np.ndarray) -> np.ndarray:
        """Return the control points of a cubic b-spline given the end points
        @param end_pts - 2x3 matrix of end points
        @return 4x3 matrix of control points"""
        return
    
    def basis(self, i: int, t: float) -> float:
        """Return the basis function for the ith control point at parameter t
        @param i - index of the control point
        @param t - parameter
        https://core.ac.uk/download/pdf/82327690.pdf
        """
        if i <= t < i + 1:
            return ((t - i) / (i + 2 - i)) * ((t - i) / (i + 1 - i))
        elif i + 1 <= t < i + 2:
            return ((((i + 2) - t) / ((i + 2) - (i + 1))) * ((t - i) / ((i + 2) - i))) + ((((i + 3) - t) / ((i + 3) - (i + 1))) * ((t - (i + 1))) / ((i + 2) - (i + 1)))
        elif i + 2 <= t < i + 3:
            return ((i + 3 - t) / (i + 3 - (i + 1))) * ((i + 3 - t) / (i + 3 - (i + 2)))
        return 0.0

    # def basis_mat(self, i, t):
    #     """Return the basis matrix for the ith control point at parameter t
    #     @param i - index of the control point
    #     @param t - parameter
    #     """
    #     if i <= t < i + 1:
    #         return np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    #     elif i + 1 <= t < i + 2:
    #         return np.array([[(i + 2) - t, t - i, 0], [0, 2 - t + i, t - i - 1], [0, 0, 0]])
    #     elif i + 2 <= t < i + 3:
    #         return np.array([[0, 0, 0], [0, 3 - t - i, t - i - 2], [0, 0, t - i - 2]])
    #     return np.zeros(shape=(3,3))
    
    def curve(self, t: float) -> np.ndarray:
        """Return the point on the curve at parameter t
        @param t: float - parameter
        @return 3d point"""
        return
    
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
    
def plot(data):
    

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
    print(fig)
    return fig


def main():
    bs = BSplineCurve()
    fig = go.Figure()
    x = np.arange(-2, 1, 0.01)
    y = np.zeros(len(x))
    
    for _, j in enumerate(x):
        y[_] = bs.basis(-2, j)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers"
        )
    )

    x = np.arange(-1, 2, 0.01)
    y = np.zeros(len(x))
    
    for _, j in enumerate(x):
        y[_] = bs.basis(-1, j)


    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers"
        )
    )

    x = np.arange(0, 3, 0.01)
    y = np.zeros(len(x))
    
    for _, j in enumerate(x):
        y[_] = bs.basis(0, j)


    
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers"
        )
    )


    fig.show()
    # print(bs.basis(0, 1.25))
    # bs.add_point(np.array([0,0,0]))
    # bs.add_point(np.array([0,1,0]))
    # bs.add_point(np.array([1,1,1]))

    # # crv_data = np.zeros(shape=(1000,3))
    # # for i, t in enumerate(np.linspace(0,1,1000)):
    # #     crv_data[i,:] = bs.pt_axis(t)

    # fig = plot(bs.pts)
    
    # fig.show()
    return

if __name__ == "__main__":
    main()