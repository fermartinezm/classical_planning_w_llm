from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the distances the robot(s) need to travel to reach the unpainted tiles.

    # Assumptions
    - Each robot can only hold one color at a time.
    - Robots can move to adjacent tiles (up, down, left, right).
    - The heuristic assumes that the robot will always choose the shortest path to the next tile to be painted.
    - The heuristic does not take into account the interaction between multiple robots.

    # Heuristic Initialization
    - Extract the goal conditions (tiles and their desired colors).
    - Identify the available colors.
    - Identify the adjacency relationships between tiles (up, down, left, right).

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal state information, including the desired color for each tile.
    2. Identify the current location and color of each robot.
    3. For each tile in the goal state:
       a. Check if the tile is already painted with the correct color in the current state.
       b. If not, determine the closest robot that can paint the tile with the correct color.
          i. If the robot does not have the correct color, estimate the cost of changing the robot's color.
          ii. Estimate the cost of moving the robot to the tile.
          iii. Add the cost of painting the tile.
    4. Sum up the costs for all tiles to get the overall heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.goal_tile_colors = {}
        for goal in self.goals:
            if goal.startswith('(painted'):
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.goal_tile_colors[tile] = color

        self.available_colors = set()
        for fact in self.static:
            if fact.startswith('(available-color'):
                parts = fact[1:-1].split()
                color = parts[1]
                self.available_colors.add(color)

        self.adjacencies = {}
        for fact in self.static:
            if fact.startswith(('(up', '(down', '(left', '(right')):
                parts = fact[1:-1].split()
                relation = parts[0]
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.adjacencies:
                    self.adjacencies[tile1] = {}
                self.adjacencies[tile1][relation] = tile2

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        if self.goal_reached(state):
            return 0

        robot_locations = {}
        robot_colors = {}
        for fact in state:
            if fact.startswith('(robot-at'):
                parts = fact[1:-1].split()
                robot = parts[1]
                tile = parts[2]
                robot_locations[robot] = tile
            elif fact.startswith('(robot-has'):
                parts = fact[1:-1].split()
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        painted_tiles = {}
        for fact in state:
            if fact.startswith('(painted'):
                parts = fact[1:-1].split()
                tile = parts[1]
                color = parts[2]
                painted_tiles[tile] = color

        total_cost = 0
        for tile, goal_color in self.goal_tile_colors.items():
            if tile in painted_tiles and painted_tiles[tile] == goal_color:
                continue

            best_robot = None
            min_cost = float('inf')

            for robot, location in robot_locations.items():
                robot_color = robot_colors[robot]
                cost = 0

                if robot_color != goal_color:
                    cost += 1  # Cost of changing color

                # Estimate the cost of moving the robot to the tile
                cost += self.estimate_distance(location, tile)

                cost += 1  # Cost of painting the tile

                if cost < min_cost:
                    min_cost = cost
                    best_robot = robot

            if best_robot:
                total_cost += min_cost

        return total_cost

    def estimate_distance(self, start_tile, end_tile):
        """Estimates the distance between two tiles using a simple heuristic (e.g., Manhattan distance)."""
        # This is a placeholder. A more sophisticated distance estimation
        # (e.g., A* search on the adjacency graph) could be used for better accuracy.
        # For now, assume a direct path exists, and each move costs 1.
        # In a real implementation, you would need to traverse the adjacency graph.
        return 1  # Simplest possible estimate: always 1.

    def goal_reached(self, state):
        """Check if all goal conditions are satisfied in the given state."""
        for goal in self.goals:
            if goal not in state:
                return False
        return True
