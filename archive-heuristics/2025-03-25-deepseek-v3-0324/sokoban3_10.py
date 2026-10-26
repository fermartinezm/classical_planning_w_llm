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
    3. A penalty for boxes that are not in goal positions but are blocking potential paths.

    # Assumptions:
    - Each box has exactly one goal position.
    - The grid is rectangular and obstacles are only walls (not other boxes).
    - Pushing a box requires the robot to be adjacent to it.

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from static facts to compute shortest paths.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not in its goal position:
        a. Calculate the Manhattan distance from the box to its goal.
        b. Calculate the shortest path distance from the robot to the box.
        c. Add these distances to the total heuristic value.
    2. For boxes already in goal positions, no cost is added.
    3. Add a small penalty for each box not in goal position to encourage progress.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building adjacency graph."""
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
                _, loc1, loc2, _ = get_parts(fact)
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = set()
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = set()
                self.adjacency[loc1].add(loc2)
                self.adjacency[loc2].add(loc1)

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
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

        if not robot_pos:
            return float('inf')  # Invalid state

        total_cost = 0

        for box, current_pos in box_positions.items():
            goal_pos = self.box_goals[box]
            
            if current_pos == goal_pos:
                continue  # Box is already at goal

            # Calculate Manhattan distance from box to goal
            x1, y1 = map(int, current_pos.split('_')[1:])
            x2, y2 = map(int, goal_pos.split('_')[1:])
            box_to_goal = abs(x1 - x2) + abs(y1 - y2)

            # Calculate Manhattan distance from robot to box
            rx, ry = map(int, robot_pos.split('_')[1:])
            robot_to_box = abs(rx - x1) + abs(ry - y1)

            # Add to total cost (each push counts as 1 action)
            total_cost += box_to_goal + robot_to_box + 1  # +1 for the push action

        return total_cost
