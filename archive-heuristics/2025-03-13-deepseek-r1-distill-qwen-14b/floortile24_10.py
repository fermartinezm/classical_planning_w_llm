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
    This heuristic estimates the number of actions needed to paint all required tiles by considering the robot's movement and color changes.

    # Assumptions:
    - The robot can move up, down, left, or right on the grid.
    - The robot can change colors, which takes one action.
    - Each painting action requires the robot to be at the tile's location with the correct color.

    # Heuristic Initialization
    - Extract the grid layout and precompute the shortest paths between tiles.
    - Store available colors from the static facts.

    # Step-by-Step Thinking for Computing Heuristic
    1. Extract the current state of the robots and the tiles.
    2. Identify all tiles that need to be painted and their required colors.
    3. For each tile, determine the minimal cost for any robot to paint it, considering movement and color changes.
    4. Sum the minimal costs for all tiles to estimate the total number of actions needed.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing necessary data structures.
        """
        super().__init__(task)
        static_facts = task.static

        # Extract grid information
        self.grid = {}
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.up[y] = x
            elif match(fact, "down", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.down[x] = y
            elif match(fact, "left", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.left[y] = x
            elif match(fact, "right", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.right[x] = y

        # Precompute distances between all pairs of tiles using BFS
        self.distances = {}
        tiles = set()
        for fact in static_facts:
            if match(fact, "up", "*", "*") or match(fact, "down", "*", "*"):
                tiles.add(get_parts(fact)[1])
                tiles.add(get_parts(fact)[2])
        for start in tiles:
            self.distances[start] = {}
            queue = deque()
            queue.append((start, 0))
            visited = {start}
            while queue:
                current, dist = queue.popleft()
                self.distances[start][current] = dist
                for neighbor in [self.up.get(current), self.down.get(current), self.left.get(current), self.right.get(current)]:
                    if neighbor and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

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
        goals = self.task.goals

        # Extract current robot positions and colors
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, x = get_parts(fact)
                robots[robot] = x
            elif match(fact, "robot-has", "*", "*"):
                robot, color = get_parts(fact)
                robots[robot] = (x := color)

        # Extract painted tiles
        painted = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                tile, color = get_parts(fact)
                painted[tile] = color

        # Identify goal tiles and their required colors
        goal_tiles = []
        for goal in goals:
            if match(goal, "painted", "*", "*"):
                tile, color = get_parts(goal)
                if tile not in painted or painted[tile] != color:
                    goal_tiles.append((tile, color))

        if not goal_tiles:
            return 0

        total_cost = 0

        # For each goal tile, find the minimal cost across all robots
        for tile, color in goal_tiles:
            min_cost = float('inf')
            for robot, info in robots.items():
                current_color = info[1] if isinstance(info, tuple) else None
                current_pos = info[0] if isinstance(info, tuple) else info

                # Check if the robot already has the correct color
                if current_color == color:
                    cost = self._calculate_distance(current_pos, tile) + 1
                else:
                    # Robot needs to change color
                    cost = 1 + self._calculate_distance(current_pos, tile) + 1

                if cost < min_cost:
                    min_cost = cost

            total_cost += min_cost

        return total_cost

    def _calculate_distance(self, from_tile, to_tile):
        """
        Calculate the precomputed distance between two tiles.
        """
        return self.distances.get(from_tile, {}).get(to_tile, float('inf'))

    def __eq__(self, other):
        return isinstance(other, FloortileHeuristic)

    def __hash__(self):
        return hash(FloortileHeuristic)
