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


class floortile3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that are not painted correctly and estimates the cost of moving the robot,
    changing colors, and painting tiles.

    # Assumptions
    - The robot can only paint tiles that are adjacent to its current location.
    - The robot can only hold one color at a time.
    - The heuristic assumes that the robot always has access to the required colors.

    # Heuristic Initialization
    - Extract the goal conditions (tiles and their desired colors).
    - Store the adjacency information between tiles (up, down, left, right).
    - Store available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that are not painted correctly in the current state.
    2. For each incorrectly painted tile, estimate the cost as follows:
       a. If the robot is not at a tile adjacent to the incorrectly painted tile, estimate the cost of moving the robot to an adjacent tile.
       b. If the robot does not have the correct color, estimate the cost of changing the robot's color.
       c. Estimate the cost of painting the tile.
    3. Sum the estimated costs for all incorrectly painted tiles.
    4. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal tile colors
        self.goal_tile_colors = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile = get_parts(goal)[1]
                color = get_parts(goal)[2]
                self.goal_tile_colors[tile] = color

        # Extract adjacency information
        self.adjacencies = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*") or match(fact, "down", "*", "*") or \
               match(fact, "left", "*", "*") or match(fact, "right", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                if tile1 not in self.adjacencies:
                    self.adjacencies[tile1] = []
                self.adjacencies[tile1].append(tile2)

        # Extract available colors
        self.available_colors = set()
        for fact in static_facts:
            if match(fact, "available-color", "*"):
                color = get_parts(fact)[1]
                self.available_colors.add(color)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Get the robot's current location
        robot_location = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_location = get_parts(fact)[2]
                break

        # Get the robot's current color
        robot_color = None
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
                break

        # Identify incorrectly painted tiles
        incorrectly_painted_tiles = []
        for tile, goal_color in self.goal_tile_colors.items():
            painted_correctly = False
            for fact in state:
                if match(fact, "painted", tile, goal_color):
                    painted_correctly = True
                    break
            if not painted_correctly:
                incorrectly_painted_tiles.append(tile)

        # Estimate the cost for each incorrectly painted tile
        total_cost = 0
        for tile in incorrectly_painted_tiles:
            # Check if the robot is adjacent to the tile
            adjacent = False
            if robot_location in self.adjacencies:
                if tile in self.adjacencies[robot_location]:
                    adjacent = True
            else:
                for t, adj in self.adjacencies.items():
                    if tile in adj and t == robot_location:
                        adjacent = True
                        break

            # Estimate the cost of moving the robot
            if not adjacent:
                total_cost += 1  # Cost of moving

            # Estimate the cost of changing the robot's color
            goal_color = self.goal_tile_colors[tile]
            if robot_color != goal_color:
                total_cost += 1  # Cost of changing color

            # Estimate the cost of painting the tile
            total_cost += 1  # Cost of painting

        # Return 0 if the goal is reached
        if not incorrectly_painted_tiles:
            return 0

        return total_cost
