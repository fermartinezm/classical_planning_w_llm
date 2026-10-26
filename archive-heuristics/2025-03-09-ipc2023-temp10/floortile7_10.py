from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all the tiles
    to their goal colors, considering the robot's current color, location, and
    the availability of colors. It prioritizes painting tiles and changing colors
    when necessary.

    # Assumptions:
    - The robot can only paint adjacent tiles (up, down, left, right).
    - The robot can only hold one color at a time.
    - The robot must be at a tile adjacent to the tile to be painted.
    - The heuristic assumes that all colors are available.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Identify the adjacency relationships between tiles (up, down, left, right).
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted and their target colors.
    2. Determine the robot's current location and the color it is holding.
    3. For each tile that needs to be painted:
       a. If the robot is not at a tile adjacent to the target tile, estimate the
          number of moves required to reach an adjacent tile.  We use a simple
          Manhattan distance heuristic for this.
       b. If the robot has the correct color, estimate the cost to paint the tile (1 action).
       c. If the robot does not have the correct color, estimate the cost to change
          the robot's color to the correct color (1 action) and then paint the tile (1 action).
    4. Sum the costs for all tiles that need to be painted.
    5. If the current state is a goal state, return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.adjacencies = {}
        self.available_colors = set()

        for fact in static_facts:
            fact_parts = self._extract_objects_from_fact(fact)
            if fact_parts[0] in ("up", "down", "left", "right"):
                tile1, tile2 = fact_parts[1], fact_parts[2]
                if tile1 not in self.adjacencies:
                    self.adjacencies[tile1] = set()
                self.adjacencies[tile1].add(tile2)
                if tile2 not in self.adjacencies:
                    self.adjacencies[tile2] = set()
                self.adjacencies[tile2].add(tile1)
            elif fact_parts[0] == "available-color":
                self.available_colors.add(fact_parts[1])

        self.goal_tiles = {}
        for goal in self.goals:
            fact_parts = self._extract_objects_from_fact(goal)
            if fact_parts[0] == "painted":
                tile, color = fact_parts[1], fact_parts[2]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state."""
        state = node.state
        if self.goal_reached(state):
            return 0

        robot_location = None
        robot_color = None
        painted_tiles = {}

        for fact in state:
            fact_parts = self._extract_objects_from_fact(fact)
            if fact_parts[0] == "robot-at":
                robot_location = fact_parts[2]
            elif fact_parts[0] == "robot-has":
                robot_color = fact_parts[2]
            elif fact_parts[0] == "painted":
                tile, color = fact_parts[1], fact_parts[2]
                painted_tiles[tile] = color

        cost = 0
        for tile, goal_color in self.goal_tiles.items():
            if tile in painted_tiles and painted_tiles[tile] == goal_color:
                continue

            adjacent = False
            if robot_location in self.adjacencies:
                if tile in self.adjacencies[robot_location]:
                    adjacent = True

            if not adjacent:
                cost += 1  # Estimate cost to move to an adjacent tile. A better heuristic could compute shortest path.

            if robot_color != goal_color:
                cost += 1  # Cost to change color
            cost += 1  # Cost to paint

        return cost

    def goal_reached(self, state):
        """Check if the goal is reached in the given state."""
        for goal in self.goals:
            if goal not in state:
                return False
        return True

    def _extract_objects_from_fact(self, fact):
        """Extract object names from a PDDL fact string."""
        return fact[1:-1].split()
