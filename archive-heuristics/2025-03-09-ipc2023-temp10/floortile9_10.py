from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the movements needed to reach the unpainted tiles.

    # Assumptions:
    - Each tile needs to be painted only once.
    - The robot can only paint adjacent tiles.
    - The robot may need to change colors multiple times.

    # Heuristic Initialization
    - Extract the goal conditions (tiles and their desired colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted according to the goal state.
    2. For each tile, check if it is already painted with the correct color.
    3. If a tile needs to be painted:
       - Check if the robot is at an adjacent tile. If not, estimate the number of moves to reach an adjacent tile.
       - Check if the robot has the correct color. If not, estimate the number of color changes needed.
       - Add 1 action for painting the tile.
    4. Sum up the estimated actions for all tiles to get the final heuristic value.
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
                    self.tile_adjacencies[tile1] = set()
                self.tile_adjacencies[tile1].add(tile2)
            elif predicate == "available-color":
                self.available_colors.add(fact_parts[1])

        self.goal_tile_colors = {}
        for goal in self.goals:
            goal_parts = self._extract_objects_from_fact(goal)
            predicate = goal_parts[0]
            if predicate == "painted":
                tile = goal_parts[1]
                color = goal_parts[2]
                self.goal_tile_colors[tile] = color

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        if self._is_goal_state(state):
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

        unpainted_tiles = 0
        color_changes = 0
        movement_cost = 0

        for tile, goal_color in self.goal_tile_colors.items():
            painted_correctly = False
            for fact in state:
                fact_parts = self._extract_objects_from_fact(fact)
                predicate = fact_parts[0]
                if predicate == "painted" and fact_parts[1] == tile and fact_parts[2] == goal_color:
                    painted_correctly = True
                    break

            if not painted_correctly:
                unpainted_tiles += 1

                # Check if robot is adjacent to the tile
                adjacent = False
                if robot_location in self.tile_adjacencies:
                    if tile in self.tile_adjacencies[robot_location]:
                        adjacent = True
                else:
                    for t, adj_tiles in self.tile_adjacencies.items():
                        if tile in adj_tiles and t == robot_location:
                            adjacent = True
                            break

                if not adjacent:
                    movement_cost += 1  # Estimate 1 move to get adjacent

                if robot_color != goal_color:
                    color_changes += 1  # Estimate 1 color change

        heuristic_value = unpainted_tiles + color_changes + movement_cost
        return heuristic_value

    def _is_goal_state(self, state):
        """Check if the current state satisfies all goal conditions."""
        return self.goals <= state

    def _extract_objects_from_fact(self, fact):
        """Extract object names from a PDDL fact string."""
        return fact[1:-1].split()
