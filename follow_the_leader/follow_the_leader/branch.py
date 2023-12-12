#!/usr/bin/env python3
from __future__ import annotations

from follow_the_leader.follow_the_leader.b_spline import BSplineCurve

from typing import List, Union


class Branch(BSplineCurve):
    def __init__(self, parent_index: int) -> None:
        super().__init__()
        # Parent/child branches
        self.children: List[Branch] = []
        self.parent: Union[Branch, None] = None
        self.parent_index = parent_index

        # Branch geometry
        self.t_value: float = None  # where along 0-1 index of the parent branch is this branch
        self.phi: float = None
        self.alpha: float = None

        return

    def get_children(self) -> List[Branch]:
        return self.children

    def add_child(self, child) -> None:
        self.children.append(child)
        child.set_parent(self)
        return

    def get_parent(self) -> Union[Branch, None]:
        return self.parent

    def set_parent(self, parent: Branch) -> None:
        self.parent = parent
        return

    def get_index(self) -> int:
        return self.parent_index

    def set_index(self, index: int) -> None:
        self.parent_index = index
        return

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
