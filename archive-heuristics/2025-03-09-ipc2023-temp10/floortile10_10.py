from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortile10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the movements needed to reach the unpainted tiles.

    # Assumptions:
    - Each robot can only hold one color at a time.
    - The heuristic assumes that robots can move freely between tiles if they are adjacent.
    - The heuristic assumes that robots can change colors freely if the target color is available.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles from static facts.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted and their target colors from the goal state.
    2. For each robot, find its current location and the color it is holding.
    3. For each tile that needs to be painted:
       a. If the tile is already painted with the correct color, ignore it.
       b. If the robot at the tile has the correct color, estimate the cost as 1 (paint action).
       c. If the robot at the tile has the wrong color, estimate the cost as 2 (1 color change + 1 paint action).
       d. If no robot is at the tile, find the closest robot and estimate the cost as the sum of:
          i. The number of moves required for the robot to reach the tile.
          ii. If the robot has the correct color, 1 (paint action).
          iii. If the robot has the wrong color, 2 (1 color change + 1 paint action).
    4. Sum up the costs for all tiles that need to be painted.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.tile_adjacencies = {}
        self.available_colors = set()

        for fact in static_facts:
            fact_str = str(fact)
            parts = fact_str[1:-1].split()
            predicate = parts[0]

            if predicate in ("up", "down", "left", "right"):
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.tile_adjacencies:
                    self.tile_adjacencies[tile1] = []
                self.tile_adjacencies[tile1].append(tile2)
            elif predicate == "available-color":
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach a goal state."""
        state = node.state

        def match(fact, *args):
            """Utility function to check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Extract goal information
        goal_tiles = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == "painted":
                tile = parts[1]
                color = parts[2]
                goal_tiles[tile] = color

        # Extract robot information
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = fact[1:-1].split()
                robot = parts[1]
                tile = parts[2]
                robot_locations[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                parts = fact[1:-1].split()
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        total_cost = 0
        for tile, target_color in goal_tiles.items():
            # Check if the tile is already painted with the correct color
            already_painted = False
            for fact in state:
                if match(fact, "painted", tile, target_color):
                    already_painted = True
                    break
            if already_painted:
                continue

            # Find the robot at the tile
            robot_at_tile = None
            for robot, location in robot_locations.items():
                if location == tile:
                    robot_at_tile = robot
                    break

            if robot_at_tile:
                robot_color = robot_colors[robot_at_tile]
                if robot_color == target_color:
                    total_cost += 1  # Paint action
                else:
                    total_cost += 2  # Color change + paint action
            else:
                # Find the closest robot
                min_distance = float('inf')
                closest_robot = None
                for robot, location in robot_locations.items():
                    distance = self.calculate_distance(location, tile)
                    if distance < min_distance:
                        min_distance = distance
                        closest_robot = robot

                robot_color = robot_colors[closest_robot]
                if robot_color == target_color:
                    total_cost += min_distance + 1  # Moves + paint action
                else:
                    total_cost += min_distance + 2  # Moves + color change + paint action

        return total_cost

    def calculate_distance(self, start_tile, end_tile):
        """Calculate the distance between two tiles using a simple BFS."""
        queue = [(start_tile, 0)]
        visited = {start_tile}

        while queue:
            tile, distance = queue.pop(0)
            if tile == end_tile:
                return distance

            if tile in self.tile_adjacencies:
                for neighbor in self.tile_adjacencies[tile]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
                        visited.add(neighbor)

        return float('inf')  # Tiles are not connected
