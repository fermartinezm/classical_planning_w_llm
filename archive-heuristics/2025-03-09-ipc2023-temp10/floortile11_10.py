from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the number of moves required to reach the tiles.

    # Assumptions:
    - Each tile needs to be painted only once.
    - The robot can only paint adjacent tiles.
    - The robot needs to have the correct color before painting.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted according to the goal state.
    2. For each tile, check if it is already painted with the correct color in the current state.
    3. If a tile needs to be painted, estimate the cost as follows:
       - If the robot is not at an adjacent tile, estimate the number of moves to reach an adjacent tile.
       - If the robot does not have the correct color, add a cost of 1 for changing the color.
       - Add a cost of 1 for painting the tile.
    4. Sum the costs for all tiles that need to be painted.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.tile_adjacencies = {}
        self.available_colors = set()

        for fact in self.static:
            fact_parts = fact[1:-1].split()
            predicate = fact_parts[0]

            if predicate in ("up", "down", "left", "right"):
                tile1 = fact_parts[1]
                tile2 = fact_parts[2]
                if tile1 not in self.tile_adjacencies:
                    self.tile_adjacencies[tile1] = set()
                self.tile_adjacencies[tile1].add(tile2)
            elif predicate == "available-color":
                self.available_colors.add(fact_parts[1])

    def __call__(self, node):
        """Estimate the minimum cost to paint all tiles to their goal colors."""
        state = node.state

        def match(fact, *args):
            """Utility function to check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Extract goal tile colors
        goal_tile_colors = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                goal_tile_colors[tile] = color

        # Extract current robot location and color
        robot_location = None
        robot_color = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = fact[1:-1].split()
                robot_location = parts[2]
            if match(fact, "robot-has", "*", "*"):
                parts = fact[1:-1].split()
                robot_color = parts[2]

        # Calculate heuristic cost
        total_cost = 0
        for tile, goal_color in goal_tile_colors.items():
            correctly_painted = False
            for fact in state:
                if match(fact, "painted", tile, goal_color):
                    correctly_painted = True
                    break

            if not correctly_painted:
                # Check if robot is adjacent to the tile
                adjacent = False
                if robot_location in self.tile_adjacencies:
                    if tile in self.tile_adjacencies[robot_location]:
                        adjacent = True
                
                if not adjacent:
                    total_cost += 1 # Estimate cost to move to an adjacent tile

                # Check if robot has the correct color
                if robot_color != goal_color:
                    total_cost += 1 # Cost to change color

                total_cost += 1 # Cost to paint the tile

        return total_cost
