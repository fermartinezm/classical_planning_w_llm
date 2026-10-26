from heuristics.heuristic_base import Heuristic

class sokoban8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It calculates the sum of Manhattan distances for each box to its goal (representing the minimal pushes needed) and adds the minimal robot distance to any box (adjusted for adjacency).

    # Assumptions:
    - Boxes can be moved in straight lines without considering obstacles.
    - The robot can reach the required position to push each box with minimal steps.
    - Each push action counts as 1, and each move action counts as 1.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task's goals.
    - Parses static information (adjacency) but in this heuristic, it's not used since Manhattan distance is grid-based.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, check if it's already at its goal. If not, compute its Manhattan distance to the goal.
    2. Sum all such distances to get the total estimated pushes needed.
    3. Find the robot's current location.
    4. For each box not at its goal, compute the Manhattan distance from the robot to the box's location, subtract 1 (since the robot needs to be adjacent to push).
    5. Take the minimal adjusted distance from step 4.
    6. The heuristic value is the sum of pushes and the minimal robot distance.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goal_locations = {}
        for goal in task.goals:
            parts = self._get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

    def _get_parts(self, fact):
        """Split a PDDL fact into its components."""
        return fact[1:-1].split()

    def _parse_location(self, loc_str):
        """Convert a location string (e.g., 'loc_3_4') into (x, y) coordinates."""
        parts = loc_str.split('_')
        x = int(parts[1])
        y = int(parts[2])
        return (x, y)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        # Find robot's current location
        robot_loc = None
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
                break
        assert robot_loc is not None, "Robot location not found in state"

        sum_pushes = 0
        boxes_not_at_goal = []
        for box, goal_loc in self.goal_locations.items():
            # Find current location of the box
            current_loc = None
            for fact in state:
                parts = self._get_parts(fact)
                if parts[0] == 'at' and parts[1] == box and len(parts) == 3:
                    current_loc = parts[2]
                    break
            assert current_loc is not None, f"Box {box} location not found in state"
            if current_loc == goal_loc:
                continue  # Box is already at goal
            # Compute Manhattan distance to goal
            current_x, current_y = self._parse_location(current_loc)
            goal_x, goal_y = self._parse_location(goal_loc)
            sum_pushes += abs(current_x - goal_x) + abs(current_y - goal_y)
            boxes_not_at_goal.append(current_loc)

        if not boxes_not_at_goal:
            return 0  # All boxes are at their goals

        # Compute minimal robot distance to any box not at goal
        robot_x, robot_y = self._parse_location(robot_loc)
        min_robot_distance = float('inf')
        for box_loc in boxes_not_at_goal:
            box_x, box_y = self._parse_location(box_loc)
            distance = abs(robot_x - box_x) + abs(robot_y - box_y)
            adjusted_distance = max(0, distance - 1)  # Need to be adjacent
            if adjusted_distance < min_robot_distance:
                min_robot_distance = adjusted_distance

        return sum_pushes + min_robot_distance
