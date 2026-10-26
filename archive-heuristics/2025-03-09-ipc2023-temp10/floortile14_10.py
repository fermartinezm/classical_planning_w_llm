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


class floortile14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that are not painted correctly and estimates the number of color changes and movements required.

    # Assumptions
    - The robot can only paint adjacent tiles.
    - The robot needs to have the correct color before painting.
    - Moving to an adjacent tile requires one action.
    - Changing color requires one action.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles) from the task.
    - Extract the adjacency information (up, down, left, right) from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal state information, i.e., which tile should have which color.
    2. For each tile in the goal state, check if it is painted with the correct color in the current state.
    3. If a tile is not painted correctly, estimate the cost to paint it correctly:
       - Check if the robot has the correct color. If not, add 1 to the cost (color change).
       - Calculate the Manhattan distance between the robot's current position and the incorrectly painted tile. Add this distance to the cost (movement).
       - Add 1 to the cost for the painting action itself.
    4. Sum up the costs for all incorrectly painted tiles.
    5. Return the total estimated cost.
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
            if match(fact, "up", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                self.adjacencies.setdefault(tile1, []).append(tile2)
                self.adjacencies.setdefault(tile2, []).append(tile1)  # Assuming bidirectional

            if match(fact, "down", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                self.adjacencies.setdefault(tile1, []).append(tile2)
                self.adjacencies.setdefault(tile2, []).append(tile1)

            if match(fact, "left", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                self.adjacencies.setdefault(tile1, []).append(tile2)
                self.adjacencies.setdefault(tile2, []).append(tile1)

            if match(fact, "right", "*", "*"):
                tile1 = get_parts(fact)[1]
                tile2 = get_parts(fact)[2]
                self.adjacencies.setdefault(tile1, []).append(tile2)
                self.adjacencies.setdefault(tile2, []).append(tile1)

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state."""
        state = node.state
        total_cost = 0

        # Get robot's current location and color
        robot_location = None
        robot_color = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_location = get_parts(fact)[2]
            if match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]

        if robot_location is None or robot_color is None:
            return float('inf')

        # Check each goal tile
        for tile, goal_color in self.goal_tile_colors.items():
            current_color = None
            for fact in state:
                if match(fact, "painted", tile, "*"):
                    current_color = get_parts(fact)[2]
                    break

            if (tile not in [get_parts(fact)[1] for fact in state if match(fact, "painted", "*", "*")]) or current_color != goal_color:
                tile_cost = 0

                # Color change cost
                if robot_color != goal_color:
                    tile_cost += 1

                # Movement cost (Manhattan distance - simplified)
                if robot_location != tile:
                    # Assuming a simple grid structure, estimate movement cost as 1
                    tile_cost += 1

                # Painting cost
                tile_cost += 1

                total_cost += tile_cost

        return total_cost
