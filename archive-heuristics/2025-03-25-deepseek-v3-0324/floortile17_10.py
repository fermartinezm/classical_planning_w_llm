from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(robot-at robot1 tile_0_1)".
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
    - The distance robots must travel to reach unpainted tiles.
    - The color changes required for robots to match the goal colors.

    # Assumptions
    - Robots can move freely between adjacent tiles.
    - Each robot can carry only one color at a time.
    - Changing color requires one action.
    - Painting a tile requires the robot to be adjacent with the correct color.

    # Heuristic Initialization
    - Extract goal conditions to identify which tiles need which colors.
    - Extract static facts to build a map of tile adjacencies.
    - Identify available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (goal condition):
        a. If already painted correctly, no cost.
        b. If not painted correctly:
            i. Find the nearest robot that can paint it (has correct color or can change color).
            ii. Compute Manhattan distance from robot to tile (movement cost).
            iii. Add cost for color change if needed.
            iv. Add cost for painting (1 action).
    2. Sum costs for all tiles, considering parallel actions by multiple robots.
    3. The heuristic value is the total estimated actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal painting conditions: {tile: color}
        self.goal_paintings = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_paintings[tile] = color

        # Build adjacency maps for each direction
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}
        for fact in static_facts:
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

        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state

        # Extract current robot positions and colors
        robot_positions = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robot_positions[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robot_colors[robot] = color

        total_cost = 0

        # For each tile that needs painting
        for tile, goal_color in self.goal_paintings.items():
            # Check if already painted correctly
            already_painted = False
            for fact in state:
                if match(fact, "painted", tile, goal_color):
                    already_painted = True
                    break
            if already_painted:
                continue

            # Find the best robot to paint this tile
            min_robot_cost = float('inf')

            for robot, robot_tile in robot_positions.items():
                # Compute movement cost (Manhattan distance)
                movement_cost = self.manhattan_distance(robot_tile, tile)

                # Compute color change cost if needed
                current_color = robot_colors.get(robot, None)
                color_cost = 0 if current_color == goal_color else 1

                total_robot_cost = movement_cost + color_cost + 1  # +1 for painting

                if total_robot_cost < min_robot_cost:
                    min_robot_cost = total_robot_cost

            total_cost += min_robot_cost

        return total_cost

    def manhattan_distance(self, tile1, tile2):
        """Compute Manhattan distance between two tiles based on their coordinates."""
        # Extract coordinates from tile names (format tile_X_Y)
        x1, y1 = map(int, tile1.split('_')[1:])
        x2, y2 = map(int, tile2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)
