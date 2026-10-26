from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(up tile_1_1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortile8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors,
    considering the robot's current location, color, and the colors of adjacent tiles.

    # Assumptions
    - The robot can only paint adjacent tiles.
    - The robot can only hold one color at a time.
    - The robot can change colors at any time if the desired color is available.
    - Moving to an adjacent tile requires one action.
    - Painting a tile requires one action.
    - Changing the robot's color requires one action.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles from static facts.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current state information:
       - Robot's location.
       - Robot's current color.
       - Painted tiles and their colors.
       - Clear tiles.

    2. Identify unpainted tiles in the goal and tiles painted with the wrong color.

    3. For each such tile, estimate the cost:
       - If the robot is not at an adjacent tile, estimate the minimum number of moves to reach an adjacent tile.
       - If the robot has the wrong color, add 1 to the cost to account for changing the color.
       - Add 1 to the cost to account for painting the tile.

    4. Sum the costs for all wrongly colored/unpainted tiles to get the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract adjacency information (up, down, left, right)
        self.adjacencies = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ("up", "down", "left", "right"):
                tile1, tile2 = parts[1], parts[2]
                if tile1 not in self.adjacencies:
                    self.adjacencies[tile1] = []
                self.adjacencies[tile1].append(tile2)

        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach a goal state from the given state.
        """
        state = node.state

        # Extract robot's location and color
        robot_location = next(
            (get_parts(fact)[1] for fact in state if match(fact, "robot-at", "*", "*")), None
        )
        robot_color = next(
            (get_parts(fact)[1] for fact in state if match(fact, "robot-has", "*", "*")), None
        )

        # Extract painted tiles and their colors
        painted_tiles = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                tile, color = get_parts(fact)[1], get_parts(fact)[2]
                painted_tiles[tile] = color

        # Extract goal tiles and their colors
        goal_tiles = {}
        for fact in self.goals:
            if match(fact, "painted", "*", "*"):
                tile, color = get_parts(fact)[1], get_parts(fact)[2]
                goal_tiles[tile] = color

        # Identify unpainted tiles in the goal and tiles painted with the wrong color
        unpainted_or_wrong_tiles = []
        for tile, color in goal_tiles.items():
            if tile not in painted_tiles or painted_tiles[tile] != color:
                unpainted_or_wrong_tiles.append((tile, color))

        # Calculate the heuristic value
        total_cost = 0
        for tile, goal_color in unpainted_or_wrong_tiles:
            cost = 0

            # Check if the robot is adjacent to the tile
            adjacent = False
            for t1, adj_tiles in self.adjacencies.items():
                if t1 == robot_location and tile in adj_tiles:
                    adjacent = True
                    break
                
            if not adjacent:
                cost += 1  # Estimate cost to move to an adjacent tile (simplified)

            # Check if the robot has the correct color
            if robot_color != goal_color:
                cost += 1  # Cost to change color

            cost += 1  # Cost to paint the tile

            total_cost += cost

        return total_cost
