from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their goal positions.
    It combines:
    1. The Manhattan distance from each box to its goal position
    2. The Manhattan distance from the robot to each box
    3. A penalty for boxes that are not on goal positions but blocking potential paths

    # Assumptions:
    - Each box has exactly one goal position (though multiple boxes can share the same goal in some variants).
    - The grid is rectangular and coordinates can be extracted from location names (format "loc_X_Y").
    - Only one box exists per location (standard Sokoban rules).

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from static facts to enable pathfinding.
    - Store clear locations to identify where boxes can be pushed.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not already at its goal:
        a. Calculate Manhattan distance from box to goal (minimum pushes needed).
        b. Find the robot's path to the box (minimum moves needed to reach pushing position).
        c. Check if the path from box to goal is clear (add penalty if blocked).
    2. Sum these values for all boxes to get the total estimate.
    3. For efficiency, we use Manhattan distance as a proxy for actual path lengths.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal positions for boxes
        self.box_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, box, loc = get_parts(goal)
                self.box_goals[box] = loc

        # Build adjacency graph for pathfinding
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.adjacency.setdefault(loc1, set()).add(loc2)
                self.adjacency.setdefault(loc2, set()).add(loc1)

    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state

        # Extract current positions
        robot_pos = None
        box_positions = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at-robot", "*"):
                robot_pos = parts[1]
            elif match(fact, "at", "*", "*"):
                box = parts[1]
                loc = parts[2]
                box_positions[box] = loc
            elif match(fact, "clear", "*"):
                clear_locations.add(parts[1])

        # If all boxes are at goals, return 0
        if all(box_positions.get(box) == goal for box, goal in self.box_goals.items()):
            return 0

        total_cost = 0

        for box, current_loc in box_positions.items():
            goal_loc = self.box_goals.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue

            # Extract coordinates from location names (format "loc_X_Y")
            try:
                _, curr_x, curr_y = current_loc.split('_')
                curr_x, curr_y = int(curr_x), int(curr_y)
                _, goal_x, goal_y = goal_loc.split('_')
                goal_x, goal_y = int(goal_x), int(goal_y)
            except:
                # Fallback if location names don't follow expected format
                box_dist = 1
                robot_dist = 1
            else:
                # Manhattan distance from box to goal
                box_dist = abs(curr_x - goal_x) + abs(curr_y - goal_y)

                # Manhattan distance from robot to box (approximate)
                if robot_pos:
                    _, robot_x, robot_y = robot_pos.split('_')
                    robot_x, robot_y = int(robot_x), int(robot_y)
                    robot_dist = abs(robot_x - curr_x) + abs(robot_y - curr_y)
                else:
                    robot_dist = 0

            # Add costs: robot to box + pushes to goal
            total_cost += robot_dist + box_dist * 2  # Pushing is more expensive than moving

            # Add penalty if box is blocking potential paths
            if current_loc in self.adjacency:
                for neighbor in self.adjacency[current_loc]:
                    if neighbor not in clear_locations and neighbor not in box_positions.values():
                        total_cost += 1  # Small penalty for blocking

        return total_cost
