from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class floortile13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the number of moves required to reach the tiles.

    # Assumptions
    - Each tile needs to be painted only once.
    - The robot can only hold one color at a time.
    - The robot can move to adjacent tiles.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Identify the available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted and their target colors.
    2. For each tile that needs painting:
       a. Check if the robot is at the tile. If not, estimate the number of moves to reach the tile.
       b. Check if the robot has the correct color. If not, estimate the number of color changes needed.
       c. Add 1 to the heuristic value for the paint action itself.
    3. Sum the estimated costs for all tiles to get the final heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal tiles and colors
        self.goal_tiles = {}
        for goal in self.goals:
            if goal.startswith('(painted'):
                parts = goal[1:-1].split()
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color

        # Extract adjacency information
        self.adj = {}
        for fact in self.static:
            if fact.startswith('(up') or fact.startswith('(down') or fact.startswith('(left') or fact.startswith('(right'):
                parts = fact[1:-1].split()
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)

        # Extract available colors
        self.available_colors = set()
        for fact in self.static:
            if fact.startswith('(available-color'):
                parts = fact[1:-1].split()
                color = parts[1]
                self.available_colors.add(color)

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal state.
        """
        state = node.state

        # Check if the state is a goal state
        if all(goal in state for goal in self.goals):
            return 0

        total_cost = 0

        # Get robot location
        robot_location = None
        for fact in state:
            if fact.startswith('(robot-at'):
                parts = fact[1:-1].split()
                robot_location = parts[2]
                break

        # Get robot color
        robot_color = None
        for fact in state:
            if fact.startswith('(robot-has'):
                parts = fact[1:-1].split()
                robot_color = parts[2]
                break

        # Calculate cost for each goal tile
        for tile, color in self.goal_tiles.items():
            # Check if the tile is already painted with the correct color
            if f'(painted {tile} {color})' in state:
                continue

            # Estimate cost to move to the tile
            if robot_location != tile:
                total_cost += self.estimate_moves(robot_location, tile)

            # Estimate cost to change color if needed
            if robot_color != color:
                total_cost += 1  # Assume one color change is sufficient

            # Add cost for painting the tile
            total_cost += 1

        return total_cost

    def estimate_moves(self, start, end):
        """
        Estimate the number of moves required to reach the end tile from the start tile.
        This is a simple heuristic that returns 1 if the tiles are adjacent, and 2 otherwise.
        """
        if start is None or end is None:
            return 100  # large cost if robot location is unknown

        if end in self.adj.get(start, []):
            return 1
        else:
            return 2
