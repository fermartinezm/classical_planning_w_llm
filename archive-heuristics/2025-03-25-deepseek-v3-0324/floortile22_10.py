from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all tiles according to the goal pattern.
    It considers:
    - The distance robots need to move to reach unpainted tiles.
    - The number of color changes required.
    - The number of tiles that still need to be painted.

    # Assumptions
    - Robots can move freely between adjacent tiles.
    - Each robot can carry only one color at a time.
    - Changing color is a separate action from painting.
    - The grid structure is static (no moving tiles).

    # Heuristic Initialization
    - Extract the goal painting pattern.
    - Build adjacency maps for tiles (up, down, left, right relations).
    - Identify available colors and initial robot positions.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (not matching goal):
        a. Find the nearest robot that can paint it (with correct color or can change color).
        b. Compute Manhattan distance from robot to tile (movement cost).
        c. Add cost for color change if needed.
    2. Sum the costs for all tiles, considering parallel actions by multiple robots.
    3. Add a penalty for remaining tiles to prioritize states closer to the goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract adjacency relations
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}

        for fact in self.static:
            if match(fact, "up", "*", "*"):
                _, y, x = get_parts(fact)
                self.up[x] = y
            elif match(fact, "down", "*", "*"):
                _, y, x = get_parts(fact)
                self.down[x] = y
            elif match(fact, "left", "*", "*"):
                _, x, y = get_parts(fact)
                self.left[x] = y
            elif match(fact, "right", "*", "*"):
                _, x, y = get_parts(fact)
                self.right[x] = y

        # Extract goal painting pattern
        self.goal_paint = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_paint[tile] = color

        # Available colors
        self.available_colors = {
            get_parts(fact)[1] for fact in self.static if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state

        # If all goals are satisfied, return 0
        if self.goals <= state:
            return 0

        # Extract current robot positions and colors
        robots = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robots[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robot_colors[robot] = color

        # Extract currently painted tiles
        current_paint = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_paint[tile] = color

        # Find all tiles that need to be painted or repainted
        tiles_to_paint = []
        for tile, goal_color in self.goal_paint.items():
            current_color = current_paint.get(tile, None)
            if current_color != goal_color:
                tiles_to_paint.append((tile, goal_color))

        # For each tile to paint, find the best robot to do it
        total_cost = 0

        for tile, goal_color in tiles_to_paint:
            min_cost = float('inf')

            for robot, robot_pos in robots.items():
                # Compute movement cost (Manhattan distance)
                x1, y1 = map(int, robot_pos.split('_')[1:])
                x2, y2 = map(int, tile.split('_')[1:])
                distance = abs(x1 - x2) + abs(y1 - y2)

                # Compute color change cost if needed
                current_color = robot_colors.get(robot, None)
                color_cost = 0 if current_color == goal_color else 1

                total_robot_cost = distance + color_cost + 1  # +1 for paint action
                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost

            total_cost += min_cost

        return total_cost
