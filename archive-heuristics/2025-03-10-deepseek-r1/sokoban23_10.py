from collections import defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class sokoban23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. It combines the minimal pushes needed for each box (Manhattan distance) with the robot's distance to the nearest box's accessible side.

    # Assumptions
    - Boxes can only be pushed to adjacent clear cells.
    - The robot can move freely through clear cells.
    - The minimal pushes for a box are approximated by Manhattan distance to its goal.
    - The robot's path to a box is approximated by Manhattan distance to the nearest clear cell adjacent to the box.

    # Heuristic Initialization
    - Extract static adjacency information to build a graph of location connections.
    - Extract goal locations for each box from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a. Compute Manhattan distance from current position to goal (estimated pushes needed).
        b. Find all clear adjacent cells to the box's current position.
        c. Compute the minimal Manhattan distance from the robot's position to any clear adjacent cell.
    2. Sum all box pushes and add the minimal robot distance to any box's accessible side.
    """

    def __init__(self, task):
        self.goal_locations = {}
        for goal in task.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts
        self.adjacency = defaultdict(list)
        for fact in task.static:
            if fact.startswith('(adjacent '):
                parts = fact[1:-1].split()
                l1 = parts[1]
                l2 = parts[2]
                self.adjacency[l1].append(l2)

    def __call__(self, node):
        state = node.state
        robot_pos = None
        boxes = []

        # Extract robot's current position
        for fact in state:
            if fact.startswith('(at-robot '):
                robot_pos = fact[1:-1].split()[1]
                break
        if not robot_pos:
            return 0  # Robot position not found (invalid state)

        # Extract box positions
        for fact in state:
            if fact.startswith('(at ') and 'box' in fact:
                parts = fact[1:-1].split()
                if parts[0] == 'at' and parts[2].startswith('loc_'):
                    box_name = parts[1]
                    box_loc = parts[2]
                    boxes.append((box_name, box_loc))

        sum_pushes = 0
        robot_distances = []

        for box_name, box_loc in boxes:
            goal_loc = self.goal_locations.get(box_name)
            if not goal_loc or box_loc == goal_loc:
                continue

            # Compute Manhattan distance for pushes
            x1, y1 = self._parse_coords(box_loc)
            x2, y2 = self._parse_coords(goal_loc)
            pushes = abs(x1 - x2) + abs(y1 - y2)
            sum_pushes += pushes

            # Find clear adjacent cells to the box
            adjacent_cells = self.adjacency.get(box_loc, [])
            clear_adjacent = [loc for loc in adjacent_cells if f'(clear {loc})' in state]
            if not clear_adjacent:
                sum_pushes += 1000  # Penalize if no clear adjacent cells
                continue

            # Compute minimal Manhattan distance from robot to clear adjacent cells
            min_dist = min(self._manhattan_distance(robot_pos, cell) for cell in clear_adjacent)
            robot_distances.append(min_dist)

        if not robot_distances:
            return sum_pushes  # All boxes at goal or no movable boxes

        return sum_pushes + min(robot_distances)

    def _parse_coords(self, loc_str):
        parts = loc_str.split('_')
        x = int(parts[1])
        y = int(parts[2])
        return x, y

    def _manhattan_distance(self, loc1, loc2):
        x1, y1 = self._parse_coords(loc1)
        x2, y2 = self._parse_coords(loc2)
        return abs(x1 - x2) + abs(y1 - y2)
