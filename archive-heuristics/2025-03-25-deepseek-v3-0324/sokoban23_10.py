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
    - Each box has exactly one goal position (though the code handles multiple goals).
    - The grid is rectangular and coordinates follow the pattern loc_X_Y.
    - Pushing a box always takes 1 action (the push) plus the movement to get behind it.

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from static facts for pathfinding.
    - Identify all clear locations from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box that's not at its goal:
        a. Calculate Manhattan distance from box to goal (minimum pushes needed).
        b. Find the robot's distance to the box (movement needed before pushing).
        c. Check if the box is blocked (no clear space behind it in goal direction).
    2. Sum the distances and add penalties for blocked boxes.
    3. The heuristic is the sum of all box costs plus robot movement costs.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal positions for boxes
        self.box_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

        # Build adjacency graph
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                parts = get_parts(fact)
                loc1, loc2, _ = parts[1], parts[2], parts[3]
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)

    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state

        # Get current positions
        robot_pos = None
        box_positions = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at-robot", "*"):
                robot_pos = parts[1]
            elif match(fact, "at", "*", "*"):
                box, loc = parts[1], parts[2]
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

            # Estimate robot distance to box (Manhattan distance as approximation)
            rx, ry = map(int, robot_pos.split('_')[1:])
            robot_to_box = abs(rx - x1) + abs(ry - y1)

            # Check if box is blocked in the direction of goal
            blocked_penalty = 0
            dx = 1 if x2 > x1 else (-1 if x2 < x1 else 0)
            dy = 1 if y2 > y1 else (-1 if y2 < y1 else 0)
            
            # Find the position behind the box (where robot needs to be to push)
            if dx != 0:  # horizontal move needed
                behind_pos = f"loc_{x1-dx}_{y1}"
            else:  # vertical move needed
                behind_pos = f"loc_{x1}_{y1-dy}"
            
            if behind_pos not in clear_locations:
                blocked_penalty = 2  # Extra cost to clear the path

            total_cost += box_to_goal + robot_to_box + blocked_penalty

        return total_cost
