from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the distances the robot(s) need to travel.

    # Assumptions:
    - Each robot can only hold one color at a time.
    - The heuristic assumes that the robots can move freely between tiles.
    - The heuristic does not consider the 'clear' predicate, assuming that robots can always move to a tile.

    # Heuristic Initialization
    - Extract the goal conditions (tiles and their desired colors).
    - Identify the available colors.
    - Store the connectivity information between tiles (up, down, left, right).
    - Store the initial robot positions and colors.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the tiles that need to be painted and their target colors based on the goal state.
    2.  For each robot, determine the tiles it needs to paint and the colors it needs to have.
    3.  Calculate the number of color changes each robot needs to perform.
    4.  Estimate the movement cost for each robot to reach the tiles it needs to paint.
        - This is a simplified estimation, assuming the robot can move directly to any tile.
    5.  Sum the number of paint actions required.
    6.  Sum the number of color change actions required.
    7.  Sum the movement costs for all robots.
    8.  Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        self.goal_tile_colors = {}
        for goal in self.goals:
            if "painted" in goal:
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.goal_tile_colors[tile] = color

        self.available_colors = set()
        for fact in self.static:
            if "available-color" in fact:
                parts = fact[1:-1].split()
                color = parts[1]
                self.available_colors.add(color)

        self.tile_connections = {}
        for fact in self.static:
            if "up" in fact or "down" in fact or "left" in fact or "right" in fact:
                parts = fact[1:-1].split()
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.tile_connections:
                    self.tile_connections[tile1] = {}
                self.tile_connections[tile1][parts[0]] = tile2

        self.initial_robot_positions = {}
        self.initial_robot_colors = {}

        for fact in task.initial_state:
            if "robot-at" in fact:
                parts = fact[1:-1].split()
                robot = parts[1]
                tile = parts[2]
                self.initial_robot_positions[robot] = tile
            if "robot-has" in fact:
                parts = fact[1:-1].split()
                robot = parts[1]
                color = parts[2]
                self.initial_robot_colors[robot] = color

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state."""
        state = node.state

        if self.goal_reached(state):
            return 0

        robot_positions = {}
        robot_colors = {}
        for fact in state:
            if "robot-at" in fact:
                parts = fact[1:-1].split()
                robot = parts[1]
                tile = parts[2]
                robot_positions[robot] = tile
            if "robot-has" in fact:
                parts = fact[1:-1].split()
                robot = parts[1]
                color = parts[2]
                robot_colors[robot] = color

        painted_tiles = {}
        for fact in state:
            if "painted" in fact:
                parts = fact[1:-1].split()
                tile = parts[1]
                color = parts[2]
                painted_tiles[tile] = color

        unpainted_tiles = {}
        for tile, color in self.goal_tile_colors.items():
            if tile not in painted_tiles or painted_tiles[tile] != color:
                unpainted_tiles[tile] = color

        paint_actions = len(unpainted_tiles)
        color_change_actions = 0
        movement_cost = 0

        for robot, color in robot_colors.items():
            target_tiles = {}
            for tile, target_color in unpainted_tiles.items():
                if target_color == color:
                    target_tiles[tile] = target_color

            if not target_tiles:
                # Robot needs to change color
                needed_colors = set(unpainted_tiles.values())
                if color not in needed_colors:
                    color_change_actions += 1
            else:
                # Robot needs to paint tiles
                movement_cost += 1 # Simplified movement cost

        return paint_actions + color_change_actions + movement_cost

    def goal_reached(self, state):
        """Check if the goal has been reached in the given state."""
        for goal in self.goals:
            if goal not in state:
                return False
        return True
