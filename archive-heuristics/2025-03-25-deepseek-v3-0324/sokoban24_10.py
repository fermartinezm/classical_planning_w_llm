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
    3. A penalty for boxes that are not in goal positions but are blocking other boxes

    # Assumptions:
    - Each box has exactly one goal position (though multiple boxes can share the same goal in some variants).
    - The grid is rectangular and coordinates follow the pattern loc_X_Y where X and Y are integers.
    - Only one box exists per location (no stacking).

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from static facts to enable pathfinding.
    - Store all clear locations (where the robot can move).

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not in its goal position:
        a. Calculate Manhattan distance from box to goal (minimum pushes needed).
        b. Find the robot's path to the box (minimum moves needed to reach pushing position).
        c. Add these costs together.
    2. For boxes already in goal positions, no cost is added.
    3. The total heuristic is the sum of costs for all boxes plus any additional penalties.
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
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)

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

        if not robot_pos:
            return float('inf')  # Invalid state

        total_cost = 0

        for box, current_pos in box_positions.items():
            if box not in self.box_goals:
                continue  # This box has no goal (shouldn't happen in standard Sokoban)

            goal_pos = self.box_goals[box]

            if current_pos == goal_pos:
                continue  # Box is already at goal

            # Calculate Manhattan distance between current position and goal
            x1, y1 = map(int, current_pos.split('_')[1:])
            x2, y2 = map(int, goal_pos.split('_')[1:])
            box_to_goal = abs(x1 - x2) + abs(y1 - y2)

            # Estimate robot's path to the box (simple Manhattan distance)
            rx, ry = map(int, robot_pos.split('_')[1:])
            robot_to_box = abs(rx - x1) + abs(ry - y1)

            # Add to total cost (each push requires at least one move)
            total_cost += box_to_goal + robot_to_box

        return total_cost
