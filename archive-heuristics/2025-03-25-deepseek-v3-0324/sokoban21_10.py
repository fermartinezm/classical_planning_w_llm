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
    3. A penalty for boxes that are not in a straight line with their goal position.
    4. A bonus for boxes that are already at their goal positions.

    # Assumptions:
    - Each box has exactly one goal position.
    - The grid is rectangular and coordinates follow the pattern "loc_X_Y".
    - The robot can only push one box at a time.
    - Diagonal moves are not allowed.

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from the static facts to enable pathfinding.
    - Parse location coordinates for distance calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box:
       a. If it's already at its goal position, skip it.
       b. Calculate the Manhattan distance from the box to its goal.
       c. Calculate the Manhattan distance from the robot to the box.
       d. Add these distances to the total heuristic value.
    2. Add a small penalty for boxes that require turns to reach their goals.
    3. The total heuristic is the sum of all box distances plus robot distances.
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
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = set()
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = set()
                self.adjacency[loc1].add(loc2)
                self.adjacency[loc2].add(loc1)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Find current robot and box positions
        robot_pos = None
        box_positions = {}
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_pos = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_positions[box] = loc

        # If all boxes are at their goals, return 0
        if all(box_positions.get(box) == goal_loc for box, goal_loc in self.box_goals.items()):
            return 0

        total_cost = 0

        for box, goal_loc in self.box_goals.items():
            current_loc = box_positions.get(box)
            if current_loc == goal_loc:
                continue  # Box is already at goal

            # Parse coordinates from location names
            try:
                _, curr_x, curr_y = current_loc.split('_')
                curr_x, curr_y = int(curr_x), int(curr_y)
                _, goal_x, goal_y = goal_loc.split('_')
                goal_x, goal_y = int(goal_x), int(goal_y)
            except (ValueError, AttributeError):
                # Fallback if location names don't follow expected pattern
                box_dist = 1
            else:
                # Calculate Manhattan distance from box to goal
                box_dist = abs(curr_x - goal_x) + abs(curr_y - goal_y)

            # Calculate Manhattan distance from robot to box
            if robot_pos:
                try:
                    _, robot_x, robot_y = robot_pos.split('_')
                    robot_x, robot_y = int(robot_x), int(robot_y)
                    robot_dist = abs(robot_x - curr_x) + abs(robot_y - curr_y)
                except (ValueError, AttributeError):
                    robot_dist = 1
            else:
                robot_dist = 1

            # Add to total cost with weighting factors
            total_cost += box_dist * 2  # Pushing is harder than moving
            total_cost += robot_dist

            # Small penalty for non-straight paths
            if robot_pos and current_loc and goal_loc:
                if not (self._is_straight_line(robot_pos, current_loc) or 
                       not self._is_straight_line(current_loc, goal_loc)):
                    total_cost += 1

        return total_cost

    def _is_straight_line(self, loc1, loc2):
        """Check if two locations are in a straight line (same row or column)."""
        try:
            _, x1, y1 = loc1.split('_')
            _, x2, y2 = loc2.split('_')
            return x1 == x2 or y1 == y2
        except (ValueError, AttributeError):
            return False
