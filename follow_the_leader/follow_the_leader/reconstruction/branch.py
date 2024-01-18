#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from b_spline_cyl import BSplineCyl

from typing import List, Union


class Branch():
    def __init__(self, name: str = "") -> None:
        # Parent/child branches
        self.name: str = name
        self.children: List[Union[Branch, None]] = []
        self._parent: Union[Branch, None] = None

        # Branch geometry
        self._t_value: float = None  # where along 0-1 index of the parent branch is this branch
        self._phi: float = None # rotation from the z-axis with respect to the parent branch
        self._theta: float = None # rotation around the z-axis with respect to the parent branch
        return
    
    def info(self) -> dict:
        """Generate a dict that contains the information of a branch"""
        info_dict = dict(
            name=self.name,
            parent=self.parent.name,
            t=self.t_value,
            phi=self.phi,
            theta=self.theta,
        )
        return info_dict

    def get_children(self) -> List[Branch]:
        """Return a list of branch children"""
        return self.children

    def add_child(self, child: Branch, t_value: float, phi: float, theta: float) -> None:
        """Add a child branch object"""
        child.set_parent(self, t_value, phi, theta)
        self.children.append(child)
        return

    @property
    def parent(self) -> Union[Branch, None]:
        """Get the parent branch object"""
        return self._parent

    def set_parent(self, parent: Branch, t_value: float, phi: float, theta: float) -> None:
        """Set the parent branch of the current branch."""
        self._parent = parent
        self.t_value = t_value
        self.phi = phi
        self.theta = theta
        return
    
    @property
    def t_value(self):
        """Get t-value with respect to the parent"""
        return self._t_value
    
    @t_value.setter
    def t_value(self, t_value: float):
        """Set t-value with respect to the parent"""
        self._t_value = t_value
        return

    @property
    def phi(self):
        """Get angle phi with respect to the parent"""
        return self._phi
    
    @phi.setter
    def phi(self, phi: float):
        """Set angle phi with respect to the parent"""
        if phi > np.pi or phi < -np.pi:
            raise ValueError("Angle phi cannot be greater than π or less than -π.")
        self._phi = phi
        return
    
    @property
    def theta(self):
        """Get angle theta with respect to the parent"""
        return self._theta

    @theta.setter
    def theta(self, theta: float):
        """Set angle theta with respect to the parent"""
        if theta > np.pi or theta < -np.pi:
            raise ValueError("Angle theta cannot be greater than π or less than -π.")
        self._theta = theta
        return

    def evaluate_branch_fitness(self):
        return

    def project_2d_image(self, camera_matrix):
        return

    def get_2d_image_pts(self, camera, pts):
        """Get the set of points from the image from reprojection"""
        return


def main():
    trunk = Branch(name="trunk")
    branch0 = Branch(name="branch0")
    trunk.add_child(branch0, t_value=0.1, phi=0.12, theta=1.15)
    
    print(trunk.children[0].parent.name)

    return


if __name__ == "__main__":
    main()
