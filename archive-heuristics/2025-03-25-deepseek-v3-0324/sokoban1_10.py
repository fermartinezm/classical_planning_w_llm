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
    3. A penalty for boxes that are not adjacent to clear spaces in the direction of their goal.

    # Assumptions:
    - Each box has exactly one goal position.
    - The grid is rectangular and coordinates follow the pattern loc_X_Y.
    - Pushing a box always requires moving the robot to an adjacent position first.

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from static facts for pathfinding.
    - Parse location coordinates for distance calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a. Calculate Manhattan distance from box to goal (box_goal_dist).
        b. Find the robot's distance to the box (robot_box_dist).
        c. Check if the box is blocked (no clear space in goal direction).
    2. Sum:
        a. The robot's distance to the farthest box.
        b. The sum of all box-to-goal distances.
        c. A penalty for each blocked box.
    3. The heuristic value is this sum, representing estimated pushes and moves.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal positions for boxes
        self.box_goals = {}
        for goal in self.goals:
            if goal.startswith('(at '):
                parts = get_parts(goal)
                box, loc = parts[1], parts[2]
                self.box_goals[box] = loc

        # Build adjacency graph
        self.adjacency = {}
        for fact in self.static:
            if fact.startswith('(adjacent '):
                parts = get_parts(fact)
                loc1, loc2, _ = parts[1], parts[2], parts[3]
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)

    def _parse_coords(self, loc):
        """Extract X,Y coordinates from location name (loc_X_Y)."""
        parts = loc.split('_')
        return int(parts[1]), int(parts[2])

    def _manhattan_distance(self, loc1, loc2):
        """Calculate Manhattan distance between two locations."""
        x1, y1 = self._parse_coords(loc1)
        x2, y2 = self._parse_coords(loc2)
        return abs(x1 - x2) + abs(y1 - y2)

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state

        # Extract current positions
        robot_pos = None
        box_positions = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_pos = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'):
                box_positions[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                clear_locations.add(parts[1])

        # If all boxes are at goals, heuristic is 0
        if all(box_positions.get(box) == goal for box, goal in self.box_goals.items()):
            return 0

        total_cost = 0
        max_robot_box_dist = 0

        for box, current_pos in box_positions.items():
            goal_pos = self.box_goals.get(box)
            if not goal_pos or current_pos == goal_pos:
                continue

            # Box to goal distance
            box_goal_dist = self._manhattan_distance(current_pos, goal_pos)
            total_cost += box_goal_dist * 2  # Each push counts as 2 actions (move + push)

            # Robot to box distance
            if robot_pos:
                robot_box_dist = self._manhattan_distance(robot_pos, current_pos)
                max_robot_box_dist = max(max_robot_box_dist, robot_box_dist)

            # Check if box is blocked in goal direction
            x, y = self._parse_coords(current_pos)
            gx, gy = self._parse_coords(goal_pos)
            dx = 1 if gx > x else (-1 if gx < x else 0)
            dy = 1 if gy > y else (-1 if gy < y else 0)
            
            # Check if next position in goal direction is clear
            next_pos = f"loc_{x+dx}_{y+dy}"
            if next_pos not in clear_locations and next_pos != goal_pos:
                total_cost += 2  # Penalty for blocked path

        total_cost += max_robot_box_dist  # Add robot's distance to farthest box

        return total_cost
