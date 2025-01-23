#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

#include <chrono>
#include <follow_the_leader_msgs/srv/follow_path.hpp>
#include <follow_the_leader_msgs/srv/move2_pose.hpp>
#include <follow_the_leader_msgs/srv/move2_state.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <std_srvs/srv/trigger.hpp>

using namespace std::chrono_literals;
static const rclcpp::Logger LOGGER =
    rclcpp::get_logger("ftl_move_group_server");

class FTLMoveGroupServer : public rclcpp::Node {
 public:
  std::string planning_group_;

  moveit::planning_interface::MoveGroupInterfacePtr move_group_;
  const moveit::core::JointModelGroup* joint_model_group_;
  std::shared_ptr<moveit_visual_tools::MoveItVisualTools> visual_tools_;
  rclcpp::Service<follow_the_leader_msgs::srv::Move2State>::SharedPtr
      move2state_server_;
  rclcpp::Service<follow_the_leader_msgs::srv::Move2Pose>::SharedPtr
      move2pose_server_;
  rclcpp::Service<follow_the_leader_msgs::srv::FollowPath>::SharedPtr
      followpath_server_;
  rclcpp::CallbackGroup::SharedPtr motion_cb_group_;

  FTLMoveGroupServer(const rclcpp::NodeOptions& options)
      : Node("ftl_move_group_server", options) {
    this->get_parameter("planning_group", planning_group_);
    RCLCPP_INFO(LOGGER, "Hello!");
  }

  bool setup_moveit() {
    RCLCPP_INFO(LOGGER, "Setting up MoveIt!");
    move_group_ =
        std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), planning_group_);
    RCLCPP_INFO(LOGGER, "Move Group Interface created");
    joint_model_group_ =
        move_group_->getCurrentState()->getJointModelGroup(planning_group_);

    // visual_tools_ = std::make_shared<moveit_visual_tools::MoveItVisualTools>(
    //     shared_from_this(), move_group_->getEndEffectorLink(), "ftl_move_group_server");

    RCLCPP_INFO(LOGGER, "Planning frame: %s",
                move_group_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "End effector link: %s",
                move_group_->getEndEffectorLink().c_str());
    motion_cb_group_ =
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    move2state_server_ =
        create_service<follow_the_leader_msgs::srv::Move2State>(
            "move2state",
            std::bind(&FTLMoveGroupServer::move2state, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, motion_cb_group_);
    move2pose_server_ = create_service<follow_the_leader_msgs::srv::Move2Pose>(
        "move2pose",
        std::bind(&FTLMoveGroupServer::move2pose, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default, motion_cb_group_);
    followpath_server_ =
        create_service<follow_the_leader_msgs::srv::FollowPath>(
            "followpath",
            std::bind(&FTLMoveGroupServer::followpath, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, motion_cb_group_);

    RCLCPP_INFO(LOGGER, "MOVE SERVICE MADE!");
    return true;
  }
  bool move2state(
      const std::shared_ptr<follow_the_leader_msgs::srv::Move2State::Request>
          request,
      const std::shared_ptr<follow_the_leader_msgs::srv::Move2State::Response>
          response) {
    move_group_->setStartStateToCurrentState();
    if (request->goal_state.joint_state.header.frame_id == "") {
      RCLCPP_WARN(this->get_logger(), "No frame id, malformed joint state");
      return false;
    }
    move_group_->setJointValueTarget(request->goal_state.joint_state);
    move_group_->setMaxVelocityScalingFactor(0.1);
    move_group_->setMaxAccelerationScalingFactor(0.1);
    if (request->planner_id != "")
      move_group_->setPlannerId(request->planner_id);
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    bool success =
        (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(LOGGER, "Plan to state %s",
                success ? "SUCCESS" : "FAILED");
    if (!success) return (response->state = success);

    //visual_tools_->deleteAllMarkers();
    //visual_tools_->publishTrajectoryLine(my_plan.trajectory_,
                                        //  joint_model_group_);
    //visual_tools_->trigger();
    rclcpp::sleep_for(std::chrono::seconds(5));

    success = (move_group_->execute(my_plan) ==
               moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(LOGGER, "Execute %s",
                success ? "SUCCESS" : "FAILED");
    if (!success) {
      int retry = 1;
      bool execute_success = false;
      while (!execute_success && retry < 4) {
        move_group_->setStartStateToCurrentState();
        execute_success =
            (move_group_->move() == moveit::core::MoveItErrorCode::SUCCESS);
        RCLCPP_INFO(LOGGER, "Replan %d and execute %s", retry,
                    execute_success ? "SUCCESS" : "FAILED");
        success = execute_success;
        retry++;
      }
    }

    return (response->state = success);
  }

  bool move2pose(
      const std::shared_ptr<follow_the_leader_msgs::srv::Move2Pose::Request>
          request,
      const std::shared_ptr<follow_the_leader_msgs::srv::Move2Pose::Response>
          response) {
    RCLCPP_INFO(LOGGER, "In move2pose server");
    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(request->goal_state);
    move_group_->setMaxVelocityScalingFactor(0.1);
    move_group_->setMaxAccelerationScalingFactor(0.1);
    move_group_->setPlanningTime(30.0);
    move_group_->setNumPlanningAttempts(100);
    if (request->planner_id != "")
      move_group_->setPlannerId(request->planner_id);

    // if (request->box_constraint)
    // {
    //   moveit_msgs::msg::PositionConstraint box_constraint;
    //   box_constraint.header.frame_id =
    //       move_group_interface.getPoseReferenceFrame();
    //   box_constraint.link_name = move_group_interface.getEndEffectorLink();
    //   shape_msgs::msg::SolidPrimitive box;
    //   box.type = shape_msgs::msg::SolidPrimitive::BOX;
    //   box.dimensions = {0.1, 0.4, 0.4};
    //   box_constraint.constraint_region.primitives.emplace_back(box);
    // }

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    bool success =
        (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(LOGGER, "Plan to pose %s",
                success ? "SUCCESS" : "FAILED");
    if (!success) return (response->state = success);

    //visual_tools_->deleteAllMarkers();
    //visual_tools_->publishTrajectoryLine(my_plan.trajectory_,
                                        //  joint_model_group_);
    //visual_tools_->trigger();
    rclcpp::sleep_for(std::chrono::seconds(10));

    success = (move_group_->execute(my_plan) ==
               moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(LOGGER, "Execute %s",
                success ? "SUCCESS" : "FAILED");
    if (!success) {
      int retry = 1;
      bool execute_success = false;
      while (!execute_success && retry < 4) {
        move_group_->setStartStateToCurrentState();
        execute_success =
            (move_group_->move() == moveit::core::MoveItErrorCode::SUCCESS);
        RCLCPP_INFO(LOGGER, "Replan %d and execute %s", retry,
                    execute_success ? "SUCCESS" : "FAILED");
        success = execute_success;
        retry++;
      }
    }

    return (response->state = success);
  }

  bool followpath(
      const std::shared_ptr<follow_the_leader_msgs::srv::FollowPath::Request>
          request,
      const std::shared_ptr<follow_the_leader_msgs::srv::FollowPath::Response>
          response) {
    RCLCPP_INFO(LOGGER, "In follow path server");
    move_group_->setStartStateToCurrentState();
    if (request->robot_trajectory.joint_trajectory.header.frame_id == "") {
      RCLCPP_WARN(this->get_logger(), "No frame id, malformed joint state");
      return false;
    }

    RCLCPP_INFO(LOGGER, "get current state");
    auto currRobotState = move_group_->getCurrentState(5.0);
    RCLCPP_INFO(LOGGER, "set RT");
    robot_trajectory::RobotTrajectory rt(currRobotState->getRobotModel(),
                                         planning_group_);
    rt.setRobotTrajectoryMsg(*currRobotState, request->robot_trajectory);

    RCLCPP_INFO(LOGGER, "do TOTG");
    moveit_msgs::msg::RobotTrajectory trajectory_msg;
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    bool success = totg.computeTimeStamps(rt, 0.1, 0.1);
    rt.getRobotTrajectoryMsg(trajectory_msg);
    RCLCPP_INFO(LOGGER,
                       "Parameterized trajectory length: %ld"
                           , trajectory_msg.joint_trajectory.points.size());
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    rclcpp::sleep_for(std::chrono::seconds(1));
    my_plan.trajectory_ = trajectory_msg;

    if (success) {
      bool execute_success = (move_group_->execute(my_plan) ==
                              moveit::core::MoveItErrorCode::SUCCESS);
      RCLCPP_INFO(LOGGER, "Trying to execute %s",
                  execute_success ? "" : "FAILED");
      success = execute_success;
    }

    return (response->state = success);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto const move_group_server = std::make_shared<FTLMoveGroupServer>(
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(
          true));
  move_group_server->setup_moveit();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(move_group_server);

  executor.spin();

  rclcpp::shutdown();
  return 0;
}
