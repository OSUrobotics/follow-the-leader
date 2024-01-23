uint8 TRUNK = 0
uint8 PRIMARY = 1
uint8 SECONDARY = 2
uint8 TERTIARY = 3

std_msgs/String summary_id
uint8 type
uint32 num_control_pts
shape_msgs/Plane constraint_plane
int8[3] growth_monoticity # 0 for no monoticity, +1 for nondecreasing or -1 nonincreasing in  x y z
follow_the_leader_msgs/SchemaBuds bud_arrangement