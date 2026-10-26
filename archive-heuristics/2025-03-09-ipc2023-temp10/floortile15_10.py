from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that are not painted correctly, the number of color changes needed,
    and the distance the robot needs to move to reach the unpainted tiles.

    # Assumptions:
    - The robot can only paint adjacent tiles.
    - The robot can only hold one color at a time.
    - The robot can change colors at any time if an available color exists.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that are not painted correctly according to the goal state.
    2. For each incorrectly painted tile:
       a. Determine the color needed for that tile.
       b. Check if the robot currently has the correct color. If not, estimate the cost of changing colors.
       c. Estimate the Manhattan distance (number of moves) from the robot's current position to the incorrectly painted tile.
       d. Add the cost of painting the tile (1 action).
    3. Sum up the costs for all incorrectly painted tiles.
    4. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.goal_colors = {}
        for goal in self.goals:
            if goal.startswith('(painted'):
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.goal_colors[tile] = color

        self.adjacencies = {}
        for fact in static_facts:
            if fact.startswith('(up') or fact.startswith('(down') or \
               fact.startswith('(left') or fact.startswith('(right'):
                parts = fact[1:-1].split()
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.adjacencies:
                    self.adjacencies[tile1] = []
                self.adjacencies[tile1].append(tile2)

        self.available_colors = set()
        for fact in static_facts:
            if fact.startswith('(available-color'):
                parts = fact[1:-1].split()
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach a goal state."""
        state = node.state

        # Check if the current state is a goal state
        if all(goal in state for goal in self.goals):
            return 0

        robot_location = None
        robot_color = None
        painted_tiles = {}

        for fact in state:
            if fact.startswith('(robot-at'):
                parts = fact[1:-1].split()
                robot_location = parts[2]
            elif fact.startswith('(robot-has'):
                parts = fact[1:-1].split()
                robot_color = parts[2]
            elif fact.startswith('(painted'):
                parts = fact[1:-1].split()
                tile = parts[1]
                color = parts[2]
                painted_tiles[tile] = color

        incorrectly_painted_tiles = []
        for tile, color in self.goal_colors.items():
            if tile not in painted_tiles or painted_tiles[tile] != color:
                incorrectly_painted_tiles.append(tile)

        total_cost = 0
        for tile in incorrectly_painted_tiles:
            goal_color = self.goal_colors[tile]

            color_change_cost = 0
            if robot_color != goal_color:
                color_change_cost = 1  # Assume one action to change color

            # Simple heuristic for distance: assume 1 move action
            # to reach any adjacent tile
            move_cost = 0
            if robot_location != tile:
                move_cost = 1

            total_cost += color_change_cost + move_cost + 1  # 1 for painting

        return total_cost
