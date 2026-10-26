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
    3. A penalty for boxes not on goal positions

    # Assumptions:
    - Each box has exactly one goal position (standard Sokoban).
    - The grid is rectangular and coordinates follow the loc_X_Y pattern.
    - Pushing a box always requires moving the robot to an adjacent position first.

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from static facts to enable pathfinding.
    - Parse location coordinates for distance calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Calculate Manhattan distance from box to goal (minimum pushes needed)
        b) Find the robot's distance to the box (minimum moves needed to reach it)
        c) Add these distances with appropriate weights
    2. Sum these values for all boxes
    3. Add a small penalty for each box not on goal to break ties
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
                self.adjacency.setdefault(loc1, set()).add(loc2)
                self.adjacency.setdefault(loc2, set()).add(loc1)

    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state

        # Find current positions of robot and boxes
        robot_pos = None
        box_positions = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_pos = parts[1]
            elif parts[0] == "at" and parts[1] in self.box_goals:
                box_positions[parts[1]] = parts[2]

        # If no boxes left to move, return 0
        if not box_positions:
            return 0

        total_cost = 0

        for box, current_pos in box_positions.items():
            goal_pos = self.box_goals[box]

            # Calculate box to goal distance
            if current_pos != goal_pos:
                # Parse coordinates (assuming loc_X_Y format)
                _, curr_x, curr_y = current_pos.split('_')
                _, goal_x, goal_y = goal_pos.split('_')
                box_dist = abs(int(curr_x) - int(goal_x)) + abs(int(curr_y) - int(goal_y))
                
                # Calculate robot to box distance (Manhattan as approximation)
                if robot_pos:
                    _, robot_x, robot_y = robot_pos.split('_')
                    robot_dist = abs(int(robot_x) - int(curr_x)) + abs(int(robot_y) - int(curr_y))
                else:
                    robot_dist = 0
                
                # Combine distances with weights
                total_cost += box_dist * 2  # Pushes are more expensive
                total_cost += robot_dist    # Moving to box
                
                # Small penalty for each box not at goal
                total_cost += 1

        return total_cost
