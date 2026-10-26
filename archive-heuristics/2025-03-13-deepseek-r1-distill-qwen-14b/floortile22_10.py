from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The current position and color of the robot.
    - The tiles that still need to be painted.
    - The shortest path between tiles.
    - The need to change colors when required.

    # Assumptions:
    - The robot can move up, down, left, or right on the grid.
    - Each move or paint action counts as one step.
    - The robot can carry only one color at a time.
    - If the robot starts with the wrong color, it must change colors before painting.

    # Heuristic Initialization
    - Extract static facts to build the grid layout and available colors.
    - Identify the goal tiles that need to be painted.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Extract the current state of the robot (position and color).
    2. Identify all tiles that need to be painted based on the goal conditions.
    3. For each required tile:
       a. Compute the Manhattan distance from the robot's current position.
       b. If the required color does not match the robot's current color, add an extra step for a color change.
    4. Sum all the distances and any additional color change steps to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and goal conditions.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build grid map from static facts
        self.grid = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid[(x, 'up')] = y
            elif match(fact, "down", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid[(x, 'down')] = y
            elif match(fact, "left", "*", "*"):
                x, y = get_parts(fact)[1], get_parts(fact)[2]
                self.grid[(y, 'left')] = x
            elif match(fact, "right", "*", "*"):
                x, y = get_parts(fact)[1], get_parts(fact)[2]
                self.grid[(y, 'right')] = x

        # Extract available colors
        self.available_colors = set()
        for fact in static_facts:
            if match(fact, "available-color", "*"):
                color = get_parts(fact)[1]
                self.available_colors.add(color)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Extract current robot state
        current_pos = None
        current_color = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                current_pos = (parts[1], parts[2])
            if match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                current_color = parts[1]

        # Identify tiles that need to be painted
        required_tiles = set()
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                required_tiles.add((tile, color))

        # If all goals are already achieved, return 0
        if not required_tiles:
            return 0

        # Find all tiles that still need painting
        painted = set()
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                painted.add((tile, color))

        remaining = required_tiles - painted
        if not remaining:
            return 0

        # Function to get neighbors of a tile
        def get_neighbors(tile):
            neighbors = []
            for dir in ['up', 'down', 'left', 'right']:
                key = (tile, dir)
                if key in self.grid:
                    neighbors.append(self.grid[key])
            return neighbors

        # Build grid graph
        graph = {}
        tiles = set()
        for fact in state:
            if match(fact, "clear", "*"):
                tiles.add(get_parts(fact)[1])
        for tile in tiles:
            graph[tile] = get_neighbors(tile)

        # Compute distances using BFS
        def shortest_path(start, goal):
            visited = {}
            queue = deque([(start, 0)])
            visited[start] = True
            while queue:
                node, dist = queue.popleft()
                if node == goal:
                    return dist
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        visited[neighbor] = True
                        queue.append((neighbor, dist + 1))
            return float('inf')

        total_cost = 0
        current_tile = current_pos[1] if current_pos else None
        if not current_tile:
            return float('inf')

        for (tile, color) in remaining:
            dist = shortest_path(current_tile, tile)
            if dist == float('inf'):
                return float('inf')
            total_cost += dist

            # Add cost for color change if needed
            if current_color != color:
                total_cost += 1

        return total_cost
