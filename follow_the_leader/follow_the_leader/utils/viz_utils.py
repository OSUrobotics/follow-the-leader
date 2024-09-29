from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from enum import Enum

class Color(Enum):
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0)
    MAGENTA = (1.0, 0.0, 1.0)
    CYAN = (0.0, 1.0, 1.0)
    WHITE = (1.0, 1.0, 1.0)
    DARK_RED = (0.5, 0.0, 0.0)
    DARK_GREEN = (0.0, 0.5, 0.0)
    DARK_BLUE = (0.0, 0.0, 0.5)
    ORANGE = (1.0, 0.5, 0.0)
    PURPLE = (0.5, 0.0, 0.5)
    TEAL = (0.0, 0.5, 0.5)

def gen_points_marker(pts, frame_id="map", color=Color.MAGENTA, scale=(0.08, 0.08)):
    """
    Create a sphere list marker from a list of points
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = marker.POINTS
    marker.action = marker.ADD
    marker.scale.x, marker.scale.y = scale
    marker.color.a = 1.0
    marker.color.r, marker.color.g, marker.color.b = color
    marker.points = [Point(x=pt[0], y=pt[1], z=pt[2]) for pt in pts]
    return marker
