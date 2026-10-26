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
    - Each box has exactly one goal position (though multiple boxes can share the same goal).
    - The grid is rectangular and locations follow the naming convention loc_X_Y.
    - The robot can only push one box at a time.
    - Moving the robot without pushing a box costs 1 action.
    - Pushing a box costs 1 action (move + push).

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from the static facts to enable pathfinding.
    - Store clear locations that can be used for movement.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not in its goal position:
        a) Calculate the Manhattan distance from the box to its goal.
        b) Find the shortest path for the robot to reach the box (using BFS if needed).
        c) Add these distances to the total cost.
    2. For boxes already in goal positions, no cost is added.
    3. Add a small penalty for boxes that are blocking paths to other boxes.
    4. The total heuristic is the sum of these individual costs.
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
        """Compute the heuristic value for the given state."""
        state = node.state

        # Extract current positions
        robot_pos = None
        box_positions = {}
        clear_locations = set()

        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_pos = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_positions[box] = loc
            elif match(fact, "clear", "*"):
                clear_locations.add(get_parts(fact)[1])

        # If all boxes are in goal positions, return 0
        if all(box_positions.get(box) == goal for box, goal in self.box_goals.items()):
            return 0

        total_cost = 0

        for box, current_loc in box_positions.items():
            goal_loc = self.box_goals.get(box)
            
            # Skip boxes already at their goal
            if current_loc == goal_loc:
                continue

            # Calculate Manhattan distance from box to goal
            x1, y1 = map(int, current_loc.split('_')[1:])
            x2, y2 = map(int, goal_loc.split('_')[1:])
            box_to_goal = abs(x1 - x2) + abs(y1 - y2)

            # Estimate robot to box distance (simplified as Manhattan)
            if robot_pos:
                rx, ry = map(int, robot_pos.split('_')[1:])
                robot_to_box = abs(rx - x1) + abs(ry - y1)
            else:
                robot_to_box = 0

            # Each push action counts as 1 (move + push)
            total_cost += box_to_goal + robot_to_box

            # Add penalty if box is blocking path to other boxes
            if current_loc in self.adjacency:
                for neighbor in self.adjacency[current_loc]:
                    if neighbor not in clear_locations and neighbor not in box_positions.values():
                        total_cost += 1

        return total_cost
