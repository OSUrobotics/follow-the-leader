#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from branch import Branch

from typing import List

class Tree(Node):
    def __init__(self) -> None:
        super().__init__(name="tree_node")

        self.tree: List[Branch]

    def project_onto_image(self, image):

        return