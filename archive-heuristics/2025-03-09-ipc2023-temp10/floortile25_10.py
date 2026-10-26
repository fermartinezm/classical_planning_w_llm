from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortile25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the number of moves required to reach the tiles.

    # Assumptions:
    - Each robot can only hold one color at a time.
    - Robots must change colors if they don't have the required color.
    - Robots must move to the adjacent tile to paint it.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted and their target colors from the goal state.
    2. For each robot, determine its current location and the color it is holding.
    3. For each tile that needs painting:
       a. Check if any robot is at an adjacent tile with the correct color.
       b. If not, estimate the cost of moving a robot to an adjacent tile and changing its color if necessary.
       c. Accumulate the costs for all tiles.
    4. The heuristic value is the sum of these estimated costs.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.tile_adjacencies = {}
        self.available_colors = set()
        self.goal_tile_colors = {}

        # Extract static information
        for fact in self.static:
            fact_parts = self._extract_objects_from_fact(fact)
            if fact_parts[0] in ("up", "down", "left", "right"):
                tile1, tile2 = fact_parts[1], fact_parts[2]
                if tile1 not in self.tile_adjacencies:
                    self.tile_adjacencies[tile1] = []
                self.tile_adjacencies[tile1].append(tile2)
            elif fact_parts[0] == "available-color":
                self.available_colors.add(fact_parts[1])

        # Extract goal information
        for goal in self.goals:
            goal_parts = self._extract_objects_from_fact(goal)
            if goal_parts[0] == "painted":
                tile, color = goal_parts[1], goal_parts[2]
                self.goal_tile_colors[tile] = color

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        robot_locations = {}
        robot_colors = {}
        clear_tiles = set()
        painted_tiles = {}

        # Extract current state information
        for fact in state:
            fact_parts = self._extract_objects_from_fact(fact)
            if fact_parts[0] == "robot-at":
                robot, tile = fact_parts[1], fact_parts[2]
                robot_locations[robot] = tile
            elif fact_parts[0] == "robot-has":
                robot, color = fact_parts[1], fact_parts[2]
                robot_colors[robot] = color
            elif fact_parts[0] == "clear":
                clear_tiles.add(fact_parts[1])
            elif fact_parts[0] == "painted":
                tile, color = fact_parts[1], fact_parts[2]
                painted_tiles[tile] = color

        # Check if the state is a goal state
        all_goals_achieved = True
        for tile, color in self.goal_tile_colors.items():
            if tile not in painted_tiles or painted_tiles[tile] != color:
                all_goals_achieved = False
                break

        if all_goals_achieved:
            return 0

        total_cost = 0
        for tile, target_color in self.goal_tile_colors.items():
            if tile in painted_tiles and painted_tiles[tile] == target_color:
                continue

            min_cost = float('inf')
            for robot, location in robot_locations.items():
                robot_color = robot_colors[robot]
                cost = 0

                # Check if robot has the correct color
                if robot_color != target_color:
                    cost += 1  # Cost for changing color

                # Check if robot is adjacent to the tile
                adjacent = False
                for adj_tile in self.tile_adjacencies.get(location, []):
                    if adj_tile == tile:
                        adjacent = True
                        break

                if not adjacent:
                    cost += 2 # Estimated cost to move to an adjacent tile

                min_cost = min(min_cost, cost)

            total_cost += min_cost

        return total_cost

    def _extract_objects_from_fact(self, fact):
        """Extract object names from a PDDL fact string."""
        return fact[1:-1].split()
