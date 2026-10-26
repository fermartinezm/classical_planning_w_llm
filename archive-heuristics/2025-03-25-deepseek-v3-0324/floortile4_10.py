from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(up tile1 tile2)".
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
    - The distance robots need to travel to reach unpainted tiles.
    - The number of color changes required.
    - The number of tiles that still need to be painted.

    # Assumptions
    - Robots can move freely between adjacent tiles.
    - Each robot can carry only one color at a time.
    - Changing color requires one action.
    - Painting a tile requires one action if the robot has the correct color.

    # Heuristic Initialization
    - Extract the goal painting conditions.
    - Build adjacency maps for tiles (up, down, left, right relations).
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (not yet matching goal):
        a. Find the nearest robot that can paint it (has the correct color or can change color).
        b. Compute the Manhattan distance between the robot and the tile.
        c. Add the distance plus any required color changes to the heuristic.
    2. Sum the costs for all tiles that need to be painted.
    3. If multiple robots can paint a tile, choose the one with the minimal cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract adjacency relations between tiles
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "up", "*", "*"):
                self.up[parts[2]] = parts[1]  # up[y] = x means y is above x
            elif match(fact, "down", "*", "*"):
                self.down[parts[2]] = parts[1]
            elif match(fact, "left", "*", "*"):
                self.left[parts[2]] = parts[1]
            elif match(fact, "right", "*", "*"):
                self.right[parts[2]] = parts[1]

        # Extract available colors
        self.available_colors = {
            parts[1] for fact in static_facts if match(fact, "available-color", "*")
        }

        # Store goal painting conditions
        self.goal_paintings = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile, color = get_parts(goal)[1:]
                self.goal_paintings[tile] = color

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Extract current robot positions and colors
        robot_positions = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "robot-at", "*", "*"):
                robot, tile = parts[1:]
                robot_positions[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                robot, color = parts[1:]
                robot_colors[robot] = color

        # Extract currently painted tiles
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                tile, color = get_parts(fact)[1:]
                current_paintings[tile] = color

        total_cost = 0

        # For each tile that needs to be painted (or repainted)
        for tile, goal_color in self.goal_paintings.items():
            current_color = current_paintings.get(tile)
            if current_color == goal_color:
                continue  # Already correctly painted

            min_robot_cost = float('inf')

            # Find the best robot to paint this tile
            for robot, robot_tile in robot_positions.items():
                robot_color = robot_colors.get(robot)

                # Compute Manhattan distance between robot and tile
                distance = self.manhattan_distance(robot_tile, tile)

                # Cost to change color if needed
                color_change_cost = 0
                if robot_color != goal_color:
                    color_change_cost = 1  # One action to change color

                total_robot_cost = distance + color_change_cost + 1  # +1 for painting

                if total_robot_cost < min_robot_cost:
                    min_robot_cost = total_robot_cost

            if min_robot_cost != float('inf'):
                total_cost += min_robot_cost

        return total_cost

    def manhattan_distance(self, tile1, tile2):
        """Compute Manhattan distance between two tiles based on their coordinates."""
        # Extract coordinates from tile names (assuming format tile_X_Y)
        try:
            x1, y1 = map(int, tile1.split('_')[1:])
            x2, y2 = map(int, tile2.split('_')[1:])
            return abs(x1 - x2) + abs(y1 - y2)
        except:
            # Fallback for unexpected tile naming
            return 0
