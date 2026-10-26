from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(robot-at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all target tiles with the required colors. It considers both the movement of the robot and any necessary color changes.

    # Assumptions:
    - The robot can move up, down, left, or right on the grid.
    - The robot can change its color if an available color is selected.
    - Each target tile must be painted with a specific color.
    - The grid layout is static and defined by the static facts.

    # Heuristic Initialization
    - Extracts the goal conditions to identify target tiles and their required colors.
    - Builds a grid map from the static facts to compute distances between tiles.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Extract the current position of the robot and the required color for each target tile.
    2. For each target tile:
       - If already painted correctly, skip it.
       - Calculate the Manhattan distance from the robot's current position to the target tile.
       - If the robot doesn't have the required color, add a color change action.
    3. Sum the movement costs for all target tiles.
    4. Add the necessary color change actions.
    5. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build grid map from static facts
        self.grid = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                if x not in self.grid:
                    self.grid[x] = {}
                self.grid[x][y] = 'up'
            elif match(fact, "down", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                if x not in self.grid:
                    self.grid[x] = {}
                self.grid[x][y] = 'down'
            elif match(fact, "left", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                if x not in self.grid:
                    self.grid[x] = {}
                self.grid[x][y] = 'left'
            elif match(fact, "right", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                if x not in self.grid:
                    self.grid[x] = {}
                self.grid[x][y] = 'right'

        # Extract available colors
        self.available_colors = set()
        for fact in static_facts:
            if match(fact, "available-color", "*"):
                color = get_parts(fact)[1]
                self.available_colors.add(color)

    def __call__(self, node):
        """Estimate the minimum cost to achieve the goal state."""
        state = node.state

        # Extract robot's current position and held color
        robot_pos = None
        held_color = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_pos = get_parts(fact)[2]
            if match(fact, "robot-has", "*", "*"):
                held_color = get_parts(fact)[2]

        # If no robot position is found, return 0 (should not happen in valid state)
        if not robot_pos:
            return 0

        # Extract target tiles and their required colors
        target_tiles = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile, color = get_parts(goal)[1], get_parts(goal)[2]
                target_tiles[tile] = color

        # If no target tiles, return 0 (goal already achieved)
        if not target_tiles:
            return 0

        total_cost = 0
        tiles_to_paint = []

        # For each target tile, check if it's already painted correctly
        for tile, required_color in target_tiles.items():
            if not any(match(f, "painted", tile, required_color) for f in state):
                tiles_to_paint.append((tile, required_color))

        # If all target tiles are already correctly painted, return 0
        if not tiles_to_paint:
            return 0

        # Calculate distances and determine color changes
        for tile, color in tiles_to_paint:
            # Calculate Manhattan distance from robot's current position to the target tile
            x1, y1 = self._parse_tile(robot_pos)
            x2, y2 = self._parse_tile(tile)
            distance = abs(x1 - x2) + abs(y1 - y2)
            total_cost += distance

            # Check if color change is needed
            if held_color != color:
                # Check if the target color is available
                if color in self.available_colors:
                    total_cost += 1  # Add color change action
                    held_color = color  # Update held color
                else:
                    # If the required color is not available, heuristic returns infinity (unsolvable)
                    return float('inf')

        return total_cost

    def _parse_tile(self, tile_str):
        """
        Parse a tile string (e.g., "tile_0_1") into (x, y) coordinates.
        """
        parts = tile_str.split('_')
        return int(parts[1]), int(parts[2])
