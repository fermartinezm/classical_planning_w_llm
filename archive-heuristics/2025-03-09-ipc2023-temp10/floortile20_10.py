from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the number of moves required to reach the tiles.

    # Assumptions
    - The robot can only paint adjacent tiles.
    - The robot needs to have the correct color before painting.
    - The heuristic assumes that the robot can always reach any tile.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted according to the goal state.
    2. For each tile, check if it is already painted with the correct color in the current state.
    3. If a tile needs to be painted, estimate the cost:
       - If the robot is not at an adjacent tile, add a move cost.
       - If the robot does not have the correct color, add a color change cost.
       - Add a paint cost.
    4. The total heuristic value is the sum of the costs for all tiles that need to be painted.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.tile_adjacencies = {}
        self.available_colors = set()

        for fact in static_facts:
            fact_parts = self._extract_objects_from_fact(fact)
            predicate = fact_parts[0]

            if predicate in ("up", "down", "left", "right"):
                tile1 = fact_parts[1]
                tile2 = fact_parts[2]
                if tile1 not in self.tile_adjacencies:
                    self.tile_adjacencies[tile1] = []
                self.tile_adjacencies[tile1].append(tile2)
            elif predicate == "available-color":
                self.available_colors.add(fact_parts[1])

        self.goal_tiles = {}
        for goal in self.goals:
            goal_parts = self._extract_objects_from_fact(goal)
            predicate = goal_parts[0]
            if predicate == "painted":
                tile = goal_parts[1]
                color = goal_parts[2]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        if self._goal_reached(state):
            return 0

        robot_location = None
        robot_color = None
        for fact in state:
            fact_parts = self._extract_objects_from_fact(fact)
            predicate = fact_parts[0]
            if predicate == "robot-at":
                robot_location = fact_parts[2]
            elif predicate == "robot-has":
                robot_color = fact_parts[2]

        cost = 0
        for tile, goal_color in self.goal_tiles.items():
            painted_correctly = False
            for fact in state:
                fact_parts = self._extract_objects_from_fact(fact)
                predicate = fact_parts[0]
                if predicate == "painted" and fact_parts[1] == tile and fact_parts[2] == goal_color:
                    painted_correctly = True
                    break

            if not painted_correctly:
                # Estimate cost for painting this tile
                tile_cost = 1  # Painting cost

                # Check if robot is at an adjacent tile
                if robot_location is None or tile not in self.tile_adjacencies or robot_location not in self.tile_adjacencies:
                    tile_cost += 1  # Move cost (assuming it takes one move to get to an adjacent tile)
                else:
                    adjacent = False
                    if robot_location in self.tile_adjacencies and tile in self.tile_adjacencies[robot_location]:
                        adjacent = True
                    else:
                        for t, adj in self.tile_adjacencies.items():
                            if tile == t and robot_location in adj:
                                adjacent = True
                                break
                    if not adjacent:
                        tile_cost += 1 # Move cost

                # Check if robot has the correct color
                if robot_color != goal_color:
                    tile_cost += 1  # Color change cost

                cost += tile_cost

        return cost

    def _extract_objects_from_fact(self, fact):
        """Extract objects from a PDDL fact string."""
        return fact[1:-1].split()

    def _goal_reached(self, state):
        """Check if the goal state has been reached."""
        return self.goals <= state
