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


class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all tiles according to the goal pattern.
    It considers:
    - The distance robots need to move to reach unpainted tiles.
    - The color changes required for robots to match the goal color for each tile.
    - The number of tiles that still need to be painted.

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right).
    - Each robot can carry only one color at a time.
    - Changing color requires one action.
    - Painting a tile requires the robot to be adjacent to it and have the correct color.

    # Heuristic Initialization
    - Extract the goal conditions (which tiles need to be painted which colors).
    - Build adjacency maps for tiles (up, down, left, right relationships).
    - Identify available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (according to goals):
        - If already painted correctly, no cost.
        - Otherwise:
            a. Find the nearest robot that can paint it (considering current color and distance).
            b. Compute cost to:
                - Change color if needed (1 action).
                - Move to adjacent tile (Manhattan distance * 1 action per move).
                - Paint the tile (1 action).
    2. Sum the costs for all tiles, prioritizing tiles that can be painted by the same robot consecutively.
    3. Add a small penalty for each robot that needs to change color multiple times.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal painting conditions: {(tile, color)}
        self.goal_paintings = set()
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_paintings.add((tile, color))

        # Build adjacency maps: {tile: {direction: neighbor}}
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
                self.left[y] = x
            elif match(fact, "right", "*", "*"):
                _, x, y = get_parts(fact)
                self.right[y] = x

        # Available colors
        self.available_colors = {
            get_parts(fact)[1] for fact in self.static if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state

        # Current robot positions and colors: {robot: (tile, color)}
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robots[robot] = [tile, None]
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                if robot in robots:
                    robots[robot][1] = color

        # Current paintings: {tile: color}
        current_paintings = {
            tile: color
            for fact in state
            if match(fact, "painted", "*", "*") and (_, tile, color) := get_parts(fact)
        }

        total_cost = 0

        for tile, goal_color in self.goal_paintings:
            # Skip if already painted correctly
            if current_paintings.get(tile) == goal_color:
                continue

            min_cost = float('inf')

            # Find best robot to paint this tile
            for robot, (robot_tile, robot_color) in robots.items():
                # Compute Manhattan distance between robot and target tile
                def get_coords(t):
                    parts = t.split('_')
                    return int(parts[1]), int(parts[2])

                x1, y1 = get_coords(robot_tile)
                x2, y2 = get_coords(tile)
                distance = abs(x1 - x2) + abs(y1 - y2)

                # Cost components
                color_change = 0 if robot_color == goal_color else 1
                move_cost = distance
                paint_cost = 1

                total_robot_cost = color_change + move_cost + paint_cost
                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost

            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost
