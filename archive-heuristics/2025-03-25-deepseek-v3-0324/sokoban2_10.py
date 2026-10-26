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
    3. A penalty for boxes that are not on goal positions but blocking potential paths

    # Assumptions:
    - Each box has exactly one goal position (standard Sokoban).
    - The grid is rectangular and coordinates follow the pattern loc_X_Y.
    - Only one box can be at a location at any time.
    - The robot can only push one box at a time.

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from the static facts to enable pathfinding.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Calculate Manhattan distance from box to goal (minimum pushes needed)
        b) Calculate Manhattan distance from robot to box (minimum moves needed to reach box)
        c) If the box is against a wall or corner in a way that might block movement, add penalty
    2. Sum these values for all boxes, prioritizing boxes farthest from their goals
    3. Add a small penalty for each box not on goal to encourage moving multiple boxes toward goals
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
        if all(box_positions.get(box) == goal for box, goal in self.box_goals.items()):
            return 0

        total_cost = 0

        for box, current_pos in box_positions.items():
            goal_pos = self.box_goals.get(box)
            if not goal_pos or current_pos == goal_pos:
                continue

            # Calculate coordinates from position names
            try:
                _, x1, y1 = current_pos.split('_')
                _, x2, y2 = goal_pos.split('_')
                box_dist = abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))
            except:
                box_dist = 1  # fallback if position format is unexpected

            # Calculate robot distance to this box
            if robot_pos:
                try:
                    _, rx, ry = robot_pos.split('_')
                    _, bx, by = current_pos.split('_')
                    robot_dist = abs(int(rx) - int(bx)) + abs(int(ry) - int(by))
                except:
                    robot_dist = 1
            else:
                robot_dist = 0

            # Add to total cost with weighting factors
            total_cost += box_dist * 2  # Pushes are more expensive than moves
            total_cost += robot_dist

            # Add penalty if box is not on goal and in a corner
            if self._is_in_corner(current_pos, clear_locations):
                total_cost += 2

        return total_cost

    def _is_in_corner(self, pos, clear_locations):
        """Check if a position is in a corner (less than 2 adjacent clear cells)."""
        if pos not in self.adjacency:
            return False
        
        clear_adjacent = sum(1 for adj in self.adjacency[pos] if adj in clear_locations)
        return clear_adjacent < 2
