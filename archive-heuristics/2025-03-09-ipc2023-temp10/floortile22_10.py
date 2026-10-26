from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the number of moves required to reach the unpainted tiles.

    # Assumptions:
    - Each robot can only hold one color at a time.
    - The robot must move to a tile before painting it.
    - Color changes are always possible if there are available colors.

    # Heuristic Initialization
    - Extract the goal conditions (tiles and their desired colors).
    - Extract the adjacency information between tiles (up, down, left, right).
    - Extract the initial robot locations and colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted and their target colors.
    2. For each robot, determine:
       - The current color of the robot.
       - The current location of the robot.
       - The number of tiles that the robot can paint with its current color.
    3. For each unpainted tile, determine the closest robot that can paint it.
       - If no robot has the correct color, estimate the cost of a color change.
       - Estimate the number of moves required for the robot to reach the tile.
    4. Sum the costs for all unpainted tiles, color changes, and movements.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.tile_adjacencies = {}
        self.available_colors = set()
        self.robots = set()
        self.tiles = set()

        for fact in self.static:
            fact_parts = self._extract_fact_parts(fact)
            if fact_parts[0] in ("up", "down", "left", "right"):
                tile1, tile2 = fact_parts[1], fact_parts[2]
                if tile1 not in self.tile_adjacencies:
                    self.tile_adjacencies[tile1] = []
                self.tile_adjacencies[tile1].append(tile2)
                if tile2 not in self.tile_adjacencies:
                    self.tile_adjacencies[tile2] = []
                self.tile_adjacencies[tile2].append(tile1)
            elif fact_parts[0] == "available-color":
                self.available_colors.add(fact_parts[1])

        for fact in task.initial_state:
            fact_parts = self._extract_fact_parts(fact)
            if fact_parts[0] == "robot-at":
                self.robots.add(fact_parts[1])
            if fact_parts[0] == "clear":
                self.tiles.add(fact_parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        if self._is_goal_state(state):
            return 0

        unpainted_tiles = []
        goal_colors = {}
        for goal in self.goals:
            goal_parts = self._extract_fact_parts(goal)
            if goal_parts[0] == "painted":
                tile, color = goal_parts[1], goal_parts[2]
                unpainted_tiles.append(tile)
                goal_colors[tile] = color

        robot_locations = {}
        robot_colors = {}
        for fact in state:
            fact_parts = self._extract_fact_parts(fact)
            if fact_parts[0] == "robot-at":
                robot, tile = fact_parts[1], fact_parts[2]
                robot_locations[robot] = tile
            elif fact_parts[0] == "robot-has":
                robot, color = fact_parts[1], fact_parts[2]
                robot_colors[robot] = color

        cost = 0
        for tile in unpainted_tiles:
            if self._is_painted(state, tile, goal_colors[tile]):
                continue

            min_cost = float('inf')
            for robot in self.robots:
                robot_location = robot_locations[robot]
                robot_color = robot_colors[robot]

                move_cost = self._calculate_move_cost(robot_location, tile)
                if robot_color == goal_colors[tile]:
                    total_cost = move_cost + 1  # Move + Paint
                else:
                    total_cost = move_cost + 2  # Move + Change Color + Paint
                min_cost = min(min_cost, total_cost)
            cost += min_cost

        return cost

    def _is_goal_state(self, state):
        """Check if the current state is a goal state."""
        for goal in self.goals:
            if goal not in state:
                return False
        return True

    def _extract_fact_parts(self, fact):
        """Extract the predicate and objects from a fact string."""
        return fact[1:-1].split()

    def _calculate_move_cost(self, start_tile, end_tile):
        """Calculate the move cost between two tiles using BFS."""
        if start_tile == end_tile:
            return 0

        queue = [(start_tile, 0)]
        visited = {start_tile}

        while queue:
            tile, distance = queue.pop(0)
            if tile == end_tile:
                return distance

            for neighbor in self.tile_adjacencies.get(tile, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf')  # Tiles are not connected

    def _is_painted(self, state, tile, color):
        """Check if a tile is painted with a specific color in the state."""
        return f"(painted {tile} {color})" in state
