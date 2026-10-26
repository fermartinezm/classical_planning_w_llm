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
    This heuristic estimates the number of actions needed to solve a Sokoban puzzle by:
    1. Calculating the Manhattan distance from each box to its nearest goal position
    2. Adding the Manhattan distance from the robot to the nearest box
    3. Considering whether pushing moves are needed (each push counts as 1 action)

    # Assumptions:
    - Each box has exactly one goal position (though multiple boxes can share the same goal)
    - The grid is rectangular and movement is only in four cardinal directions
    - Pushing a box always moves it one square in the direction of the push

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals
    - Build an adjacency graph from static facts to enable pathfinding
    - Store all clear locations that can be moved to

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Calculate Manhattan distance to its goal position
        b) Add 1 for each required push (since each push is one action)
    2. For the robot:
        a) Find the nearest box that's not at its goal
        b) Calculate Manhattan distance to that box
    3. Sum all these distances to get the heuristic estimate
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
                    self.adjacency[loc1] = set()
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = set()
                self.adjacency[loc1].add(loc2)
                self.adjacency[loc2].add(loc1)

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
                box_positions[parts[1]] = parts[2]
            elif match(fact, "clear", "*"):
                clear_locations.add(parts[1])

        # If all boxes are at goals, return 0
        if all(box_positions.get(box, None) == goal for box, goal in self.box_goals.items()):
            return 0

        total_cost = 0

        # Calculate box movement costs
        for box, goal in self.box_goals.items():
            current_pos = box_positions.get(box, None)
            if current_pos != goal:
                # Estimate distance from current position to goal
                x1, y1 = map(int, current_pos.split('_')[1:])
                x2, y2 = map(int, goal.split('_')[1:])
                total_cost += abs(x1 - x2) + abs(y1 - y2)
                # Add cost for each push (1 per move)
                total_cost += abs(x1 - x2) + abs(y1 - y2)

        # Calculate robot movement cost to nearest box
        if robot_pos and box_positions:
            min_dist = float('inf')
            for box, pos in box_positions.items():
                if pos != self.box_goals.get(box, None):
                    x1, y1 = map(int, robot_pos.split('_')[1:])
                    x2, y2 = map(int, pos.split('_')[1:])
                    dist = abs(x1 - x2) + abs(y1 - y2)
                    if dist < min_dist:
                        min_dist = dist
            if min_dist != float('inf'):
                total_cost += min_dist

        return total_cost
