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
    1) The Manhattan distance from each box to its nearest goal.
    2) The Manhattan distance from the robot to each box (to account for pushing).
    3) A penalty for boxes that are not on a goal and adjacent to walls or corners where they might get stuck.

    # Assumptions:
    - Each box has exactly one goal location (as per the provided examples).
    - The grid is rectangular and coordinates follow the pattern loc_X_Y.
    - The heuristic does not need to be admissible since it's used with greedy best-first search.

    # Heuristic Initialization
    - Extract goal locations for boxes from the task goals.
    - Build an adjacency graph from static facts to compute distances efficiently.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not already at its goal:
        a) Compute Manhattan distance from box to its goal (how far it needs to be pushed).
        b) Compute Manhattan distance from robot to box (how far the robot needs to move to reach the box).
        c) Add these distances to the total cost.
    2. Add a small penalty for boxes adjacent to walls or corners to avoid deadlocks.
    3. If all boxes are at their goals, return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for boxes
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.box_goals[box] = location

        # Build adjacency graph for distance computation
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.adjacency.setdefault(loc1, set()).add(loc2)
                self.adjacency.setdefault(loc2, set()).add(loc1)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Extract current positions of boxes and robot
        box_positions = {}
        robot_pos = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                if parts[1].startswith("box"):
                    box_positions[parts[1]] = parts[2]
                elif parts[1] == "robot":
                    robot_pos = parts[2]
            elif parts[0] == "at-robot":
                robot_pos = parts[1]

        total_cost = 0

        for box, goal_loc in self.box_goals.items():
            current_loc = box_positions.get(box)
            if current_loc == goal_loc:
                continue  # Box is already at goal

            # Compute Manhattan distance from box to goal
            x1, y1 = map(int, current_loc.split('_')[1:])
            x2, y2 = map(int, goal_loc.split('_')[1:])
            box_to_goal = abs(x1 - x2) + abs(y1 - y2)

            # Compute Manhattan distance from robot to box
            rx, ry = map(int, robot_pos.split('_')[1:])
            robot_to_box = abs(rx - x1) + abs(ry - y1)

            # Add to total cost (each push requires robot to reach box first)
            total_cost += robot_to_box + box_to_goal

            # Add penalty if box is near wall/corner (deadlock risk)
            neighbors = self.adjacency.get(current_loc, set())
            if len(neighbors) < 4:  # Assuming 4-way movement, fewer neighbors means wall/corner
                total_cost += 2

        return total_cost
