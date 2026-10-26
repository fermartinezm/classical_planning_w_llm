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
    3. A penalty for boxes that are not adjacent to clear spaces in the direction of their goal

    # Assumptions:
    - Each box has exactly one goal position.
    - The robot can only push one box at a time.
    - Pushing a box requires the robot to be adjacent to it and have a clear space behind the box.

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from static facts to enable pathfinding.
    - Identify all clear locations (where boxes or robot can move).

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Calculate Manhattan distance from box to goal (minimum pushes needed)
        b) Find shortest path from robot to box (minimum moves needed to reach box)
        c) Check if box is pushable toward goal (if not, add penalty)
    2. Sum:
        a) The robot's distance to the farthest box
        b) The sum of all boxes' distances to their goals
        c) Penalties for boxes that can't be directly pushed toward goal
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal positions for boxes
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

        # Build adjacency graph
        self.adjacent = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2, _ = parts[1], parts[2], parts[3]
                if loc1 not in self.adjacent:
                    self.adjacent[loc1] = set()
                self.adjacent[loc1].add(loc2)

    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state

        # Get current positions of robot and boxes
        robot_pos = None
        box_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_pos = parts[1]
            elif parts[0] == "at" and parts[1] in self.box_goals:
                box_positions[parts[1]] = parts[2]

        # If all boxes are at goals, heuristic is 0
        if all(box_positions.get(box) == goal for box, goal in self.box_goals.items()):
            return 0

        total_cost = 0

        # Calculate distances and penalties for each box
        for box, current_pos in box_positions.items():
            goal_pos = self.box_goals[box]

            # Skip boxes already at goal
            if current_pos == goal_pos:
                continue

            # Manhattan distance from box to goal (minimum pushes needed)
            x1, y1 = map(int, current_pos.split('_')[1:])
            x2, y2 = map(int, goal_pos.split('_')[1:])
            box_to_goal = abs(x1 - x2) + abs(y1 - y2)

            # Find robot's distance to box (BFS would be better but Manhattan is faster)
            if robot_pos:
                rx, ry = map(int, robot_pos.split('_')[1:])
                robot_to_box = abs(rx - x1) + abs(ry - y1)
            else:
                robot_to_box = 0

            # Check if box can be pushed toward goal
            pushable = False
            if goal_pos in self.adjacent.get(current_pos, set()):
                pushable = True

            # Add to total cost
            total_cost += box_to_goal + robot_to_box
            if not pushable:
                total_cost += 2  # Penalty for needing to reposition

        return total_cost
