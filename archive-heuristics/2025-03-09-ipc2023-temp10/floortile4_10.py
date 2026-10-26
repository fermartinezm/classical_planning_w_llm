from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the number of moves required to reach the tiles.

    # Assumptions:
    - Each tile needs to be painted only once.
    - The robot can only paint adjacent tiles.
    - The robot can change colors as needed.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted according to the goal state.
    2. For each tile, check if it is already painted with the correct color in the current state.
    3. If a tile needs to be painted, estimate the cost to paint it:
       - Check if the robot is at the tile. If not, estimate the number of moves to reach it.
       - Check if the robot has the correct color. If not, add a color change action.
       - Add one paint action.
    4. Sum the costs for all tiles to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.tile_connections = {}
        self.available_colors = set()

        for fact in static_facts:
            fact = fact[1:-1]
            parts = fact.split()
            if parts[0] in ("up", "down", "left", "right"):
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.tile_connections:
                    self.tile_connections[tile1] = []
                self.tile_connections[tile1].append(tile2)
                if tile2 not in self.tile_connections:
                    self.tile_connections[tile2] = []
                self.tile_connections[tile2].append(tile1)
            elif parts[0] == "available-color":
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        goal_tiles = {}
        for goal in self.goals:
            goal = goal[1:-1]
            parts = goal.split()
            if parts[0] == "painted":
                tile = parts[1]
                color = parts[2]
                goal_tiles[tile] = color

        robot_location = None
        robot_color = None
        for fact in state:
            fact = fact[1:-1]
            parts = fact.split()
            if parts[0] == "robot-at":
                robot_location = parts[2]
            elif parts[0] == "robot-has":
                robot_color = parts[2]

        cost = 0
        for tile, goal_color in goal_tiles.items():
            painted_correctly = False
            for fact in state:
                fact = fact[1:-1]
                parts = fact.split()
                if parts[0] == "painted" and parts[1] == tile and parts[2] == goal_color:
                    painted_correctly = True
                    break

            if not painted_correctly:
                # Cost to move to the tile
                if robot_location != tile:
                    cost += 1  # Assume one move is needed. A better heuristic would calculate the actual distance.

                # Cost to change color
                if robot_color != goal_color:
                    cost += 1  # Color change action

                # Cost to paint
                cost += 1  # Paint action

        return cost
