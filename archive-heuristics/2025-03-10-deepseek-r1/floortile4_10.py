from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

class Floortile4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles with their required colors. For each tile not yet painted correctly, it calculates the minimal cost considering robot movement, color changes, and the paint action.

    # Assumptions:
    - The grid is regular, and tile coordinates can be derived from their names (tile_X_Y).
    - Each goal tile is initially clear (unpainted) or already painted correctly.
    - Robots can change colors freely as long as the target color is available.

    # Heuristic Initialization
    - Extract goal tiles and their required colors.
    - Parse tile coordinates from their names.
    - Build adjacency lists for each tile using static predicates.
    - Collect available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile, check if it's already painted correctly. If not, proceed.
    2. For each robot, calculate the minimal steps to reach any adjacent tile of the target:
        a. Compute Manhattan distance from the robot's current position to each adjacent tile.
        b. Determine if a color change is needed (1 action if required).
        c. Add 1 action for painting.
    3. Sum the minimal costs for all unpainted goal tiles.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static information."""
        self.goal_tiles = {}
        for fact in task.goals:
            parts = fact[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color

        self.tile_coords = {}
        self.adjacent_to = defaultdict(list)
        self.available_colors = set()

        for fact in task.static:
            if fact.startswith(('(up ', '(down ', '(left ', '(right ')):
                parts = fact[1:-1].split()
                adjacent_tile = parts[1]
                tile = parts[2]
                self.adjacent_to[tile].append(adjacent_tile)
                for t in [tile, adjacent_tile]:
                    if t not in self.tile_coords:
                        coords = t.split('_')[1:]
                        x, y = map(int, coords)
                        self.tile_coords[t] = (x, y)
            elif fact.startswith('(available-color '):
                parts = fact[1:-1].split()
                self.available_colors.add(parts[1])

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Check painted status for goal tiles
        unpainted = []
        for tile, color in self.goal_tiles.items():
            if f'(painted {tile} {color})' not in state:
                unpainted.append((tile, color))

        # Collect robot positions and colors
        robots = {}
        robot_colors = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'robot-at':
                robots[parts[1]] = parts[2]
            elif parts[0] == 'robot-has':
                robot_colors[parts[1]] = parts[2]

        for tile, target_color in unpainted:
            if target_color not in self.available_colors:
                continue  # Assume problem is solvable, so skip if color unavailable

            min_cost = float('inf')
            adj_tiles = self.adjacent_to.get(tile, [])

            for robot, pos in robots.items():
                current_color = robot_colors.get(robot, None)
                if current_color is None:
                    continue

                rx, ry = self.tile_coords[pos]
                for adj in adj_tiles:
                    ax, ay = self.tile_coords[adj]
                    distance = abs(rx - ax) + abs(ry - ay)
                    color_cost = 0 if current_color == target_color else 1
                    cost = distance + color_cost + 1  # move + color + paint
                    if cost < min_cost:
                        min_cost = cost

            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost
