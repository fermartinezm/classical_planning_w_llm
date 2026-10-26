from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the number of moves required to reach unpainted tiles.

    # Assumptions
    - Each tile needs to be painted only once.
    - The robot can only hold one color at a time.
    - Moving to an adjacent tile takes one action.
    - Changing color takes one action.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles from static facts.
    - Extract available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted according to the goal state.
    2. For each tile, check if it is already painted with the correct color in the current state.
    3. Count the number of tiles that are not painted correctly.
    4. Identify the current color of the robot.
    5. For each incorrectly painted tile, check if the robot has the correct color. If not, increment the color change count.
    6. Estimate the number of moves required to reach the incorrectly painted tiles. This is a simplification and does not consider optimal paths.
    7. The heuristic value is the sum of the number of incorrectly painted tiles, the number of color changes, and a simplified move cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.goal_tiles = {}
        for goal in self.goals:
            if goal.startswith("(painted"):
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color

        self.adj = {}
        for fact in self.static:
            if fact.startswith("(up") or fact.startswith("(down") or fact.startswith("(left") or fact.startswith("(right"):
                parts = fact[1:-1].split()
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)

        self.available_colors = set()
        for fact in self.static:
            if fact.startswith("(available-color"):
                parts = fact[1:-1].split()
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach a goal state."""
        state = node.state

        incorrect_tiles = 0
        color_changes = 0
        robot_tile = None
        robot_color = None

        for fact in state:
            if fact.startswith("(robot-at"):
                parts = fact[1:-1].split()
                robot_tile = parts[2]
            if fact.startswith("(robot-has"):
                parts = fact[1:-1].split()
                robot_color = parts[2]

        for tile, goal_color in self.goal_tiles.items():
            painted_correctly = False
            for fact in state:
                if fact == f"(painted {tile} {goal_color})":
                    painted_correctly = True
                    break
            if not painted_correctly:
                incorrect_tiles += 1
                if robot_color != goal_color:
                    color_changes += 1

        if not self.goal_reached(state):
            return incorrect_tiles + color_changes
        else:
            return 0
