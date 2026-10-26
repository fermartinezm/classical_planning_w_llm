from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The robot's current position.
    - The colors the robot holds.
    - The tiles that still need to be painted.
    - The distance to the nearest unpainted tile of the required color.

    # Assumptions:
    - The robot can move up, down, left, or right to adjacent tiles.
    - The robot can change colors if another color is available.
    - Each painting action requires the robot to be on the tile and hold the correct color.

    # Heuristic Initialization
    - Extracts goal conditions to know which tiles need to be painted and with what colors.
    - Builds a grid map to track tile connections (up, down, left, right) from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted and their required colors.
    2. For each required color:
       a. Count how many tiles still need to be painted with that color.
       b. Find the nearest unpainted tile of that color to the robot's current position.
       c. Calculate the Manhattan distance to that tile.
       d. If the robot doesn't hold the required color, add the cost to change colors.
    3. Sum the costs for all required colors, including movement and potential color changes.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build grid map from static facts
        self.grid = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] in ['up', 'down', 'left', 'right']:
                x, y = parts[1], parts[2]
                self.grid.setdefault(x, {})[y] = parts[0]

        # Precompute available colors
        self.available_colors = set()
        for fact in self.static:
            if fact.startswith('(available-color '):
                color = fact.split()[1]
                self.available_colors.add(color)

    def __call__(self, node):
        """Estimate the minimum cost to achieve the goal state."""
        state = node.state
        goal = self.goals

        # Parse current state
        robot_pos = None
        current_color = None
        painted = set()
        for fact in state:
            if fact.startswith('(robot-at '):
                robot_pos = fact.split()[2]
            elif fact.startswith('(robot-has '):
                current_color = fact.split()[2]
            elif fact.startswith('(painted '):
                painted.add((fact.split()[1], fact.split()[2]))

        # Parse goal conditions
        goal_tiles = {}
        for fact in goal:
            if fact.startswith('(painted '):
                tile, color = fact.split()[1], fact.split()[2]
                goal_tiles[(tile, color)] = True

        # If all goals are already met
        if not goal_tiles:
            return 0

        # For each required color, find the nearest unpainted tile
        total_cost = 0
        colors_needed = set(color for (_, color) in goal_tiles.keys())

        for color in colors_needed:
            # Count remaining tiles of this color
            remaining = []
            for (tile, c) in goal_tiles:
                if c == color and (tile, c) not in painted:
                    remaining.append(tile)

            if not remaining:
                continue

            # Find the nearest tile to robot_pos
            nearest = None
            min_dist = float('inf')
            for tile in remaining:
                if tile not in self.grid:
                    continue
                # Calculate Manhattan distance
                path = self.find_path(robot_pos, tile)
                if path:
                    dist = len(path) - 1
                    if dist < min_dist:
                        min_dist = dist
                        nearest = tile

            if nearest is None:
                continue  # No path found (should not happen in solvable states)

            # Add movement cost
            total_cost += min_dist

            # Add color change cost if needed
            if current_color != color:
                if 'free-color' in [f.split()[0] for f in state]:
                    total_cost += 1  # Change color action
                else:
                    # Need to drop current color first
                    total_cost += 2  # Change color action and move to available color tile

        return total_cost

    def find_path(self, start, end):
        """Find the shortest path using BFS considering grid connections."""
        from collections import deque
        visited = set()
        queue = deque([(start, [start])])
        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.grid.get(current, []):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        return None
