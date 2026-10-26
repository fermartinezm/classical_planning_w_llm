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
    1. The Manhattan distance from each box to its goal position.
    2. The Manhattan distance from the robot to each box.
    3. A penalty for boxes that are not adjacent to clear spaces in the direction of their goal.

    # Assumptions:
    - Each box has exactly one goal position.
    - The grid is rectangular and coordinates follow the pattern loc_X_Y.
    - Pushing a box always requires moving the robot to an adjacent position first.

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Parse static adjacency information into a graph structure.
    - Precompute all clear locations from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a. Calculate Manhattan distance from box to goal (box_goal_dist).
        b. Calculate Manhattan distance from robot to box (robot_box_dist).
        c. Check if the box is blocked (no clear space in direction of goal).
    2. The heuristic value is the sum of:
        a. For each box: box_goal_dist * 2 (push + move)
        b. The minimum robot_box_dist among all boxes
        c. A penalty for each blocked box
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

        # Build adjacency graph
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, _ = get_parts(fact)
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
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

        # If all boxes are at their goals, return 0
        if all(box_positions.get(box) == goal_loc for box, goal_loc in self.box_goals.items()):
            return 0

        total_cost = 0
        min_robot_box_dist = float('inf')

        for box, box_loc in box_positions.items():
            goal_loc = self.box_goals.get(box)
            if not goal_loc or box_loc == goal_loc:
                continue

            # Parse coordinates from location names (assumes format loc_X_Y)
            _, box_x, box_y = box_loc.split('_')
            box_x, box_y = int(box_x), int(box_y)
            _, goal_x, goal_y = goal_loc.split('_')
            goal_x, goal_y = int(goal_x), int(goal_y)

            # Calculate Manhattan distances
            box_goal_dist = abs(box_x - goal_x) + abs(box_y - goal_y)
            robot_box_dist = self._manhattan_distance(robot_pos, box_loc)

            # Update minimum robot-box distance
            if robot_box_dist < min_robot_box_dist:
                min_robot_box_dist = robot_box_dist

            # Check if box is blocked in direction of goal
            penalty = 0
            if box_x < goal_x and f"loc_{box_x+1}_{box_y}" not in clear_locations:
                penalty += 1
            elif box_x > goal_x and f"loc_{box_x-1}_{box_y}" not in clear_locations:
                penalty += 1
            elif box_y < goal_y and f"loc_{box_x}_{box_y+1}" not in clear_locations:
                penalty += 1
            elif box_y > goal_y and f"loc_{box_x}_{box_y-1}" not in clear_locations:
                penalty += 1

            total_cost += box_goal_dist * 2 + penalty

        # Add the minimum robot-box distance (we'll need to reach at least one box)
        if min_robot_box_dist != float('inf'):
            total_cost += min_robot_box_dist

        return total_cost

    def _manhattan_distance(self, loc1, loc2):
        """Calculate Manhattan distance between two locations."""
        if not loc1 or not loc2:
            return 0
        _, x1, y1 = loc1.split('_')
        _, x2, y2 = loc2.split('_')
        return abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))
