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
    - Changing color requires one action.
    - Painting a tile requires one action if the robot is adjacent and has the correct color.

    # Heuristic Initialization
    - Extract goal conditions to identify which tiles need to be painted and their colors.
    - Extract static facts to build a graph of tile adjacencies (up, down, left, right).
    - Extract available colors for robots.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (goal conditions):
        - If already painted correctly, no cost.
        - Otherwise:
            a. Find the nearest robot that can paint it (considering current color and distance).
            b. Compute cost to change color (if needed) + move to tile + paint it.
    2. Sum costs for all tiles, considering parallel actions by multiple robots.
    3. Add penalty for tiles that require color changes but are far from the robot's current position.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract adjacency relationships between tiles
        self.adjacency = {
            'up': {},
            'down': {},
            'left': {},
            'right': {}
        }
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] in self.adjacency:
                self.adjacency[parts[0]][parts[1]] = parts[2]

        # Extract goal painting conditions
        self.goal_paintings = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_paintings[tile] = color

        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state

        # Extract current robot positions and colors
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robots[robot] = {'pos': tile, 'color': None}
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                if robot in robots:
                    robots[robot]['color'] = color

        # Extract currently painted tiles
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_paintings[tile] = color

        total_cost = 0

        # For each tile that needs painting in the goal
        for tile, goal_color in self.goal_paintings.items():
            # Skip if already painted correctly
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue

            min_robot_cost = float('inf')

            # Find the best robot to paint this tile
            for robot, data in robots.items():
                robot_pos = data['pos']
                robot_color = data['color']

                # Compute movement cost (Manhattan distance)
                x1, y1 = map(int, robot_pos.split('_')[1:])
                x2, y2 = map(int, tile.split('_')[1:])
                move_cost = abs(x1 - x2) + abs(y1 - y2)

                # Compute color change cost if needed
                color_cost = 0 if robot_color == goal_color else 1

                # Total cost for this robot: move + color change + paint (1 action)
                total_robot_cost = move_cost + color_cost + 1

                if total_robot_cost < min_robot_cost:
                    min_robot_cost = total_robot_cost

            if min_robot_cost != float('inf'):
                total_cost += min_robot_cost

        return total_cost
