import math
from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

class floortile2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles with their required colors. For each unpainted tile, it calculates the minimal cost considering the closest robot, including movement, color changes, and painting.

    # Assumptions
    - Robots can move freely between adjacent tiles (ignoring whether tiles are clear).
    - The grid is a perfect grid, allowing Manhattan distance for movement cost.
    - Adjacent tiles are clear for painting (ignoring current 'clear' state for simplicity).

    # Heuristic Initialization
    - Extract adjacency relations between tiles from static facts.
    - Parse tile coordinates from their names.
    - Extract goal conditions (tile and color pairs).

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile not yet painted:
        a. Find all adjacent tiles (possible positions to paint from).
        b. For each robot:
            i. Calculate Manhattan distance from robot's current tile to each adjacent tile.
            ii. Take the minimal distance as movement steps.
            iii. Add 1 if a color change is needed.
            iv. Add 1 for the paint action.
        c. Sum the minimal cost across all robots for this tile.
    2. Sum all tile costs for the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information from the task."""
        self.adjacent = defaultdict(list)
        self.tile_coords = {}
        self.goal_painted = []

        # Extract adjacency relations and tile coordinates
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] in ['up', 'down', 'left', 'right']:
                tile1, tile2 = parts[1], parts[2]
                self.adjacent[tile1].append(tile2)
                self.adjacent[tile2].append(tile1)
                # Parse coordinates for both tiles
                for tile in [tile1, tile2]:
                    if tile not in self.tile_coords:
                        coords = tile.split('_')[1:]
                        x, y = int(coords[0]), int(coords[1])
                        self.tile_coords[tile] = (x, y)

        # Extract goal painted conditions
        for goal in task.goals:
            if goal.startswith('(painted'):
                parts = goal[1:-1].split()
                tile, color = parts[1], parts[2]
                self.goal_painted.append((tile, color))

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        robots = {}
        # Collect robot positions and colors
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                robots[robot] = {'tile': tile, 'color': None}
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                if robot in robots:
                    robots[robot]['color'] = color

        total_cost = 0

        for tile, req_color in self.goal_painted:
            # Check if tile is already painted correctly
            if f'(painted {tile} {req_color})' in state:
                continue

            adjacent_tiles = self.adjacent.get(tile, [])
            if not adjacent_tiles:
                # No adjacent tiles, add a high penalty
                total_cost += 1000
                continue

            min_cost = math.inf

            for robot in robots.values():
                current_tile = robot['tile']
                current_color = robot['color']

                if current_tile not in self.tile_coords:
                    continue  # Invalid position

                rx, ry = self.tile_coords[current_tile]

                # Find minimal distance to any adjacent tile
                min_distance = math.inf
                for adj_tile in adjacent_tiles:
                    if adj_tile not in self.tile_coords:
                        continue
                    ax, ay = self.tile_coords[adj_tile]
                    distance = abs(rx - ax) + abs(ry - ay)
                    if distance < min_distance:
                        min_distance = distance

                if min_distance == math.inf:
                    continue  # No path

                # Check color change
                color_change = 0 if current_color == req_color else 1

                # Total cost for this robot
                cost = min_distance + color_change + 1  # movement + color + paint

                if cost < min_cost:
                    min_cost = cost

            if min_cost != math.inf:
                total_cost += min_cost
            else:
                # No robot can reach, add high penalty
                total_cost += 1000

        return total_cost
