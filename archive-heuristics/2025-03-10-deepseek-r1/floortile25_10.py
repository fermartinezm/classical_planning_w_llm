from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their required colors. For each tile not yet painted correctly, it calculates the minimal cost for any robot to reach an adjacent tile, change color if necessary, and paint the tile. The total heuristic value is the sum of these minimal costs.

    # Assumptions
    - Robots can move freely between tiles (ignores 'clear' predicate for efficiency).
    - Changing color is possible if required, assuming the target color is available.
    - Each robot can only paint one tile at a time (parallelism not considered).
    - Adjacency relationships are static and precomputed from the domain's 'up', 'down', 'left', 'right' predicates.

    # Heuristic Initialization
    - Extract adjacency relationships between tiles from static facts.
    - Parse tile coordinates from their names (e.g., 'tile_1_2' → (1,2)).
    - Identify goal tiles and their required colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile T:
        a. Check if T is already painted correctly in the current state.
        b. If not, find all tiles adjacent to T (from static adjacency data).
    2. For each such T, compute the minimal cost across all robots:
        a. For each robot, compute the Manhattan distance from its current position to each of T's adjacent tiles, taking the minimum.
        b. Add 1 action for painting T.
        c. Add 1 action if the robot's current color doesn't match T's required color.
    3. Sum the minimal costs for all unpainted goal tiles.
    """

    def __init__(self, task):
        """Initialize the heuristic with static data and goals."""
        self.goal_painted = {}
        self.adjacency_map = {}
        self.tile_coords = {}

        # Extract goal painted conditions
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color

        # Build adjacency map from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                tile_from = parts[1]
                tile_to = parts[2]
                if tile_to not in self.adjacency_map:
                    self.adjacency_map[tile_to] = []
                self.adjacency_map[tile_to].append(tile_from)

        # Parse tile coordinates from their names
        all_tiles = set(self.adjacency_map.keys())
        all_tiles.update(self.goal_painted.keys())
        for tile in all_tiles:
            if tile.startswith('tile_'):
                parts = tile.split('_')
                x = int(parts[1])
                y = int(parts[2])
                self.tile_coords[tile] = (x, y)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        robot_positions = {}
        robot_colors = {}

        # Extract robot positions and colors from state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                robot_positions[robot] = tile
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        total_cost = 0

        for tile, required_color in self.goal_painted.items():
            # Check if the tile is already correctly painted
            painted_correct = any(fact == f'(painted {tile} {required_color})' for fact in state)
            if painted_correct:
                continue

            adjacent_tiles = self.adjacency_map.get(tile, [])
            if not adjacent_tiles:
                continue

            min_cost = float('inf')
            for robot in robot_positions:
                current_pos = robot_positions[robot]
                current_color = robot_colors.get(robot, None)
                if current_color is None:
                    continue

                # Calculate minimal distance to any adjacent tile of the target
                min_distance = float('inf')
                for adj_tile in adjacent_tiles:
                    if current_pos not in self.tile_coords or adj_tile not in self.tile_coords:
                        continue
                    rx, ry = self.tile_coords[current_pos]
                    ax, ay = self.tile_coords[adj_tile]
                    distance = abs(rx - ax) + abs(ry - ay)
                    if distance < min_distance:
                        min_distance = distance

                color_cost = 1 if current_color != required_color else 0
                cost = min_distance + 1 + color_cost  # 1 for painting
                if cost < min_cost:
                    min_cost = cost

            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost
