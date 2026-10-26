from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the number of moves required to reach the tiles.

    # Assumptions:
    - Each robot can only hold one color at a time.
    - The robot must change color if it has the wrong color for the current tile.
    - The heuristic assumes that all clear tiles can be reached.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Identify the robots and their initial colors and locations.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted and their target colors.
    2. For each robot, determine:
       - Its current location and color.
       - The set of tiles it can potentially paint.
    3. For each tile that needs painting:
       - Find the closest robot that can paint it (considering the robot's current color and location).
       - Estimate the cost for that robot to paint the tile:
         - Cost of moving to the tile.
         - Cost of changing color (if necessary).
         - Cost of painting the tile.
    4. Sum up the costs for all tiles to get the overall heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal tile colors
        self.goal_tile_colors = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_tile_colors[tile] = color

        # Extract adjacency information
        self.adj = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] in ('up', 'down', 'left', 'right'):
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)

        # Extract robot information
        self.robots = {}
        self.available_colors = set()
        for fact in task.initial_state:
            parts = fact[1:-1].split()
            if parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                self.robots[robot] = {'location': tile, 'color': None}
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                if robot in self.robots:
                    self.robots[robot]['color'] = color
            elif parts[0] == 'available-color':
                color = parts[1]
                self.available_colors.add(color)

        # Store available colors
        self.available_colors = set()
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'available-color':
                color = parts[1]
                self.available_colors.add(color)

    def __call__(self, node):
        """Estimate the minimum cost to paint all tiles to their goal colors."""
        state = node.state

        # Check if the goal is already reached
        if all(goal in state for goal in self.goals):
            return 0

        # Extract current tile colors
        current_tile_colors = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                current_tile_colors[tile] = color

        # Extract robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'robot-at':
                robot = parts[1]
                tile = parts[2]
                robot_locations[robot] = tile
            elif parts[0] == 'robot-has':
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        # Calculate the cost for each tile that needs painting
        total_cost = 0
        for tile, goal_color in self.goal_tile_colors.items():
            if tile in current_tile_colors and current_tile_colors[tile] == goal_color:
                continue  # Tile is already painted correctly

            best_robot = None
            min_cost = float('inf')

            for robot, robot_info in self.robots.items():
                robot_location = robot_locations.get(robot, None)
                robot_color = robot_colors.get(robot, None)

                if robot_location is None or robot_color is None:
                    continue

                # Calculate the cost for this robot to paint the tile
                move_cost = self.calculate_move_cost(robot_location, tile)
                color_change_cost = 1 if robot_color != goal_color else 0
                paint_cost = 1

                cost = move_cost + color_change_cost + paint_cost

                if cost < min_cost:
                    min_cost = cost
                    best_robot = robot

            if best_robot is not None:
                total_cost += min_cost

        return total_cost

    def calculate_move_cost(self, start_tile, end_tile):
        """Calculate the move cost between two tiles using a simple BFS."""
        if start_tile == end_tile:
            return 0

        queue = [(start_tile, 0)]
        visited = {start_tile}

        while queue:
            tile, distance = queue.pop(0)

            if tile == end_tile:
                return distance

            if tile in self.adj:
                for neighbor in self.adj[tile]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
                        visited.add(neighbor)

        return float('inf')  # Tiles are not reachable
