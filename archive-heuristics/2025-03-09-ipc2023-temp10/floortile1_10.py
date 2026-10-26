from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles
    according to the goal state, considering the robot's current position,
    color, and the colors of adjacent tiles. It prioritizes painting tiles
    that are close to the robot and considers the cost of moving and changing
    colors.

    # Assumptions:
    - The robot can only paint adjacent tiles (up, down, left, right).
    - The robot can only hold one color at a time.
    - The heuristic assumes that the shortest path to a tile is always the best.
    - Changing colors takes one action.
    - Moving to an adjacent tile takes one action.
    - Painting a tile takes one action.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Store the available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted according to the goal state.
    2. Determine the robot's current position and color.
    3. For each unpainted tile in the goal:
       a. Calculate the minimum number of moves required for the robot to reach an adjacent tile.
       b. If the robot has the wrong color, add one action for changing the color.
       c. Add one action for painting the tile.
    4. Sum the estimated actions for all unpainted tiles to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.adjacencies = {}
        self.available_colors = set()

        for fact in self.static:
            fact_parts = self._extract_objects_from_fact(fact)
            if fact_parts[0] in ("up", "down", "left", "right"):
                tile1, tile2 = fact_parts[1], fact_parts[2]
                if tile1 not in self.adjacencies:
                    self.adjacencies[tile1] = {}
                self.adjacencies[tile1][fact_parts[0]] = tile2
            elif fact_parts[0] == "available-color":
                self.available_colors.add(fact_parts[1])

    def __call__(self, node):
        """Estimate the minimum cost to paint all tiles according to the goal state."""
        state = node.state

        goal_tiles_colors = {}
        for goal in self.goals:
            parts = self._extract_objects_from_fact(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                goal_tiles_colors[tile] = color

        robot_location = None
        robot_color = None
        for fact in state:
            parts = self._extract_objects_from_fact(fact)
            if parts[0] == "robot-at":
                robot_location = parts[1]
            elif parts[0] == "robot-has":
                robot_color = parts[1]

        if robot_location is None or robot_color is None:
            return float('inf')

        unpainted_tiles = []
        for tile, color in goal_tiles_colors.items():
            painted = False
            for fact in state:
                parts = self._extract_objects_from_fact(fact)
                if parts[0] == "painted" and parts[1] == tile and parts[2] == color:
                    painted = True
                    break
            if not painted:
                unpainted_tiles.append((tile, color))

        if not unpainted_tiles:
            return 0

        total_cost = 0
        for tile, color in unpainted_tiles:
            adjacent = False
            for direction, adjacent_tile in self.adjacencies.get(robot_location, {}).items():
                if adjacent_tile == tile:
                    adjacent = True
                    break
            if not adjacent:
                for t1, adj in self.adjacencies.items():
                    for direction, adjacent_tile in adj.items():
                        if adjacent_tile == tile:
                            adjacent = True
                            break
                    if adjacent:
                        break
            if not adjacent:
                total_cost += 1 # Move to adjacent tile
            
            if robot_color != color:
                total_cost += 1  # Change color

            total_cost += 1  # Paint tile

        return total_cost

    def _extract_objects_from_fact(self, fact):
        """Extract the object names from a PDDL fact string."""
        return fact[1:-1].split()
