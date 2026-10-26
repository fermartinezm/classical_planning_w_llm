from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all the tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the number of moves required to reach the tiles.

    # Assumptions
    - Each tile needs to be painted only once.
    - The robot can only paint adjacent tiles.
    - The robot may need to change colors multiple times.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles from the static facts.
    - Extract available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted according to the goal state.
    2. For each tile, check if it is already painted with the correct color in the current state.
    3. If a tile needs to be painted, estimate the cost to paint it:
       - Check if the robot is at an adjacent tile. If not, estimate the number of moves to reach an adjacent tile.
       - Check if the robot has the correct color. If not, add a color change cost.
       - Add a painting cost (1 action).
    4. Sum the costs for all tiles that need to be painted.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.tile_colors_goal = {}
        for goal in self.goals:
            if goal.startswith('(painted'):
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.tile_colors_goal[tile] = color

        self.adjacencies = {}
        for fact in static_facts:
            if fact.startswith('(up') or fact.startswith('(down') or fact.startswith('(left') or fact.startswith('(right'):
                parts = fact[1:-1].split()
                relation = parts[0]
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.adjacencies:
                    self.adjacencies[tile1] = []
                self.adjacencies[tile1].append((relation, tile2))

        self.available_colors = set()
        for fact in static_facts:
            if fact.startswith('(available-color'):
                parts = fact[1:-1].split()
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        robot_tile = None
        robot_color = None
        painted_tiles = {}

        for fact in state:
            if fact.startswith('(robot-at'):
                parts = fact[1:-1].split()
                robot_tile = parts[2]
            elif fact.startswith('(robot-has'):
                parts = fact[1:-1].split()
                robot_color = parts[2]
            elif fact.startswith('(painted'):
                parts = fact[1:-1].split()
                tile = parts[1]
                color = parts[2]
                painted_tiles[tile] = color

        cost = 0
        for tile, goal_color in self.tile_colors_goal.items():
            if tile in painted_tiles and painted_tiles[tile] == goal_color:
                continue

            adjacent = False
            if robot_tile in self.adjacencies:
                for _, adj_tile in self.adjacencies[robot_tile]:
                    if tile == adj_tile:
                        adjacent = True
                        break
            if not adjacent:
                cost += 1  # Move to an adjacent tile

            if robot_color != goal_color:
                cost += 1  # Change color

            cost += 1  # Paint the tile

        return cost
