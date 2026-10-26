from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortile18Heuristic(Heuristic):
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
    - Extract the goal conditions (tiles and their desired colors).
    - Extract the adjacency information between tiles (up, down, left, right).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted and their target colors from the goal state.
    2. For each tile that needs painting:
       a. Determine the robot's current location and the color it is holding.
       b. Calculate the minimum number of moves required for the robot to reach the tile.
       c. If the robot is not holding the correct color, estimate the cost of changing colors.
       d. Add the cost of painting the tile (1 action).
    3. Sum the costs for all tiles to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions (tiles and their desired colors).
        - Adjacency information between tiles (up, down, left, right).
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal colors for each tile.
        self.goal_colors = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile = get_parts(goal)[1]
                color = get_parts(goal)[2]
                self.goal_colors[tile] = color

        # Store adjacency information between tiles.
        self.adj = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)
                if tile2 not in self.adj:
                    self.adj[tile2] = []
                self.adj[tile2].append(tile1)
            elif match(fact, "down", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)
                if tile2 not in self.adj:
                    self.adj[tile2] = []
                self.adj[tile2].append(tile1)
            elif match(fact, "left", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)
                if tile2 not in self.adj:
                    self.adj[tile2] = []
                self.adj[tile2].append(tile1)
            elif match(fact, "right", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                self.adj[tile1].append(tile2)
                if tile2 not in self.adj:
                    self.adj[tile2] = []
                self.adj[tile2].append(tile1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get the robot's current location.
        robot_location = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_location = get_parts(fact)[2]
                break

        # Get the color the robot is holding.
        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break

        total_cost = 0

        for tile, goal_color in self.goal_colors.items():
            # Check if the tile is already painted with the correct color.
            painted_correctly = False
            for fact in state:
                if match(fact, "painted", tile, goal_color):
                    painted_correctly = True
                    break

            if painted_correctly:
                continue

            # Calculate the minimum number of moves to reach the tile.
            if robot_location is None:
                return float('inf')

            queue = [(robot_location, 0)]
            visited = {robot_location}
            min_moves = float('inf')

            while queue:
                curr_tile, moves = queue.pop(0)
                if curr_tile == tile:
                    min_moves = moves
                    break

                if curr_tile in self.adj:
                    for neighbor in self.adj[curr_tile]:
                        is_clear = False
                        for fact in state:
                            if match(fact, "clear", neighbor):
                                is_clear = True
                                break
                        if neighbor not in visited and is_clear:
                            queue.append((neighbor, moves + 1))
                            visited.add(neighbor)

            if min_moves == float('inf'):
                return float('inf')

            # Check if the robot has the correct color.
            if robot_color != goal_color:
                total_cost += 1  # Cost of changing color

            total_cost += min_moves  # Cost of moving to the tile
            total_cost += 1  # Cost of painting the tile

        return total_cost
