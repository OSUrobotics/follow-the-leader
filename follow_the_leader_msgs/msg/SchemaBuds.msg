# basic arrangement type
uint8 OPPOSITE=0
uint8 SPIRAL=1 # alternate is a subset of spiral
uint8 WHORLED=2

# type of opposite
uint8 DECUSSATE=0
uint8 SUPERPOSED=1

std_msgs/String summary_id
uint32 buds_per_branch
uint8 basic_arrangement_type
bool is_distichous
uint8 opposite_type # only for opposite
uint8 whorled_cluster_num # only for whorled
float64 angle_of_rotation # only for spiral
