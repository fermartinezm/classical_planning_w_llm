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
    - The number of unpainted tiles that need to be painted.
    - The distance robots need to travel to reach unpainted tiles.
    - The color changes required for robots to match the goal colors.

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right).
    - Each robot can carry only one color at a time.
    - Changing color requires an action.
    - Painting a tile requires the robot to be adjacent and have the correct color.

    # Heuristic Initialization
    - Extract goal conditions to identify which tiles need to be painted and their target colors.
    - Extract static facts to build a map of tile adjacencies (up, down, left, right).
    - Identify available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (according to goals):
        - If already painted correctly, no cost.
        - Otherwise:
            a. Find the nearest robot that can paint it (considering current color and distance).
            b. Compute cost to:
                - Change color if needed (1 action).
                - Move to the tile (Manhattan distance actions).
                - Paint the tile (1 action).
    2. Sum the costs for all tiles, prioritizing tiles that can be painted by the same robot consecutively.
    3. Add a penalty if multiple robots are needed to cover distant tiles quickly.
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
        self.adjacency = {}
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                direction, tile1, tile2 = parts
                if tile1 not in self.adjacency:
                    self.adjacency[tile1] = {}
                self.adjacency[tile1][direction] = tile2

        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1] for fact in self.static if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Extract current robot positions and colors: {robot: (tile, color)}
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robots[robot] = {"tile": tile, "color": None}
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                if robot in robots:
                    robots[robot]["color"] = color

        # Extract currently painted tiles: {tile: color}
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_paintings[tile] = color

        total_cost = 0

        for tile, goal_color in self.goal_paintings:
            # Skip if already painted correctly
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue

            # Find the best robot to paint this tile
            min_cost = float('inf')
            for robot, data in robots.items():
                robot_tile = data["tile"]
                robot_color = data["color"]

                # Compute Manhattan distance between robot and tile
                def get_coords(t):
                    # Extract coordinates from tile name (e.g., "tile_1_2" -> (1, 2))
                    parts = t.split('_')
                    return int(parts[1]), int(parts[2])

                x1, y1 = get_coords(robot_tile)
                x2, y2 = get_coords(tile)
                distance = abs(x1 - x2) + abs(y1 - y2)

                # Cost to change color if needed
                color_cost = 0 if robot_color == goal_color else 1

                # Total cost for this robot: move + color change + paint
                cost = distance + color_cost + 1
                if cost < min_cost:
                    min_cost = cost

            total_cost += min_cost

        return total_cost
