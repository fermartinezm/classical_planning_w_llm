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
    3. A penalty for boxes that are not in a goal position but are blocking potential paths

    # Assumptions:
    - Each box has exactly one goal position (though multiple boxes can share the same goal in some variants).
    - The grid is rectangular and movement is only in four cardinal directions.
    - Pushing a box requires the robot to be adjacent to it and have an empty space behind the box.

    # Heuristic Initialization
    - Extract goal positions for all boxes from the task goals.
    - Build an adjacency graph from the static facts to enable pathfinding.
    - Identify all clear locations (where boxes can be pushed to).

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not in its goal position:
        a. Calculate the Manhattan distance from its current position to its goal.
        b. Find the shortest path distance from the robot to the box (considering obstacles).
        c. Add these distances with appropriate weights.
    2. For boxes already in goal positions, no cost is added.
    3. Add a small penalty for each box not in goal position to encourage focusing on unsolved boxes.
    4. The total heuristic is the sum of all individual box costs plus penalties.
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
                _, loc1, loc2, _ = get_parts(fact)
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get current robot position
        robot_pos = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, loc = get_parts(fact)
                robot_pos = loc
                break

        if not robot_pos:
            return float('inf')  # Invalid state

        total_cost = 0

        # Get current box positions
        box_positions = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_positions[box] = loc

        # For each box, compute its contribution to the heuristic
        for box, goal_loc in self.box_goals.items():
            current_loc = box_positions.get(box, None)
            if not current_loc:
                return float('inf')  # Box missing - invalid state

            if current_loc == goal_loc:
                continue  # Box already at goal

            # Compute Manhattan distance from box to goal
            x1, y1 = map(int, current_loc.split('_')[1:])
            x2, y2 = map(int, goal_loc.split('_')[1:])
            box_to_goal = abs(x1 - x2) + abs(y1 - y2)

            # Estimate robot to box distance (simplified as Manhattan)
            rx, ry = map(int, robot_pos.split('_')[1:])
            robot_to_box = abs(rx - x1) + abs(ry - y1)

            # Add to total cost with weights
            total_cost += robot_to_box + 2 * box_to_goal

            # Small penalty for each box not at goal
            total_cost += 1

        return total_cost
