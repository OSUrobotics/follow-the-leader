#!/usr/bin/env python3
from __future__ import annotations

from follow_the_leader.follow_the_leader.reconstruction.b_spline_cyl import BSplineCyl


from typing import List, Union


class Branch():
    def __init__(self, parent_index: int) -> None:
        super().__init__()
        # Parent/child branches
        self.name: str = ""
        self.children: Union[Branch, None] = []
        self.parent: Union[Branch, None] = None

        # Branch geometry
        self.t_value: float = None  # where along 0-1 index of the parent branch is this branch
        self.phi: float = None # rotation from the z-axis with respect to the parent branch
        self.theta: float = None # rotation around the z-axis with respect to the parent branch
        return
    
    def info(self) -> str:
        """Generate a string that contains the information of a branch"""
        return

    def get_children(self) -> List[Branch]:
        return self.children

    def add_child(self, child: Branch, phi: float, theta: float) -> None:
        """Add a child branch object"""
        child.set_parent(self, phi, theta)
        self.children.append(child)
        return

    def get_parent(self) -> Union[Branch, None]:
        """Get the parent branch object"""
        return self.parent

    def set_parent(self, parent: Branch, phi: float, theta: float) -> None:
        self.parent = parent
        self.phi = phi
        self.theta = theta
        return
    
    def set_phi(self, phi):
        """Set angle phi with respect to the parent"""
        self.phi = phi
        return
    
    def get_phi(self):
        """Get angle phi with respect to the parent"""
        return self.phi
    
    def set_phi(self, theta):
        """Set angle theta with respect to the parent"""
        self.theta = theta
        return
    
    def get_phi(self):
        """Get angle theta with respect to the parent"""
        return self.theta

    def evaluate_branch_fitness(self):
        return

    def project_2d_image(self, camera_matrix):
        return

    def get_2d_image_pts(self, camera, pts):
        """Get the set of points from the image from reprojection"""
        return


def main():
    tree = Branch(0)
    branch0 = Branch(0)

    tree.add_child(branch0)

    return


if __name__ == "__main__":
    main()
