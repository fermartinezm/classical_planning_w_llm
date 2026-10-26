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
    3. A penalty for boxes that are not in a goal position but are blocking potential paths.

    # Assumptions:
    - Each box has exactly one goal position.
    - The grid is rectangular and coordinates follow the pattern loc_X_Y.
    - Pushing a box always requires moving the robot to an adjacent position first.

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from static facts to enable pathfinding.
    - Parse location coordinates for distance calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not in its goal position:
        a) Calculate Manhattan distance from box to goal (minimum pushes needed).
        b) Find the robot's distance to the box (minimum moves needed before pushing).
        c) Add these distances with appropriate weights.
    2. For boxes already in goal positions:
        a) No cost is added as they don't need to be moved.
    3. Add a small penalty for each box not in goal position to break ties.
    4. The total heuristic is the sum of all box movement estimates.
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
        """Compute heuristic estimate for the given state."""
        state = node.state

        # Get current robot and box positions
        robot_pos = None
        box_positions = {}
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_pos = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_positions[box] = loc

        # If all boxes are in goal positions, return 0
        if all(box_positions.get(box) == goal for box, goal in self.box_goals.items()):
            return 0

        total_cost = 0

        for box, goal_loc in self.box_goals.items():
            current_loc = box_positions.get(box)
            if current_loc == goal_loc:
                continue  # Box is already at goal

            # Calculate Manhattan distance between box and goal
            x1, y1 = map(int, current_loc.split('_')[1:])
            x2, y2 = map(int, goal_loc.split('_')[1:])
            box_to_goal = abs(x1 - x2) + abs(y1 - y2)

            # Calculate Manhattan distance between robot and box
            rx, ry = map(int, robot_pos.split('_')[1:])
            robot_to_box = abs(rx - x1) + abs(ry - y1)

            # Add to total cost (each push requires at least one move)
            total_cost += box_to_goal + robot_to_box

            # Small penalty for each box not in goal position
            total_cost += 1

        return total_cost
