#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from branch import Branch

from typing import List


class Tree(Node):
    def __init__(self) -> None:
        super().__init__(name="tree_node")

        self.tree: Branch = None
        self.score: float = 0.0
        self.size: float = 0.0  # (sum of the total length covered to calculate score)
        return

    def calculate_score(self):
        return

    def project_onto_image(self, image):

        return


def main():

    return


if __name__ == "__main__":
    main()
