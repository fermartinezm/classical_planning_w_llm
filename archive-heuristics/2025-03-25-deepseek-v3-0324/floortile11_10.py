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
    This heuristic estimates the number of actions needed to paint all required tiles
    with their goal colors. It considers:
    - The distance robots need to move to reach unpainted tiles
    - The color changes required for robots to have the correct paint color
    - The painting actions needed for each tile

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each robot can carry only one color at a time
    - Changing color takes one action
    - Painting a tile takes one action
    - Moving to an adjacent tile takes one action

    # Heuristic Initialization
    - Extract goal painting conditions for each tile
    - Build adjacency graph for tiles from static facts
    - Identify available colors from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (from goal conditions):
        a. If already painted correctly, no cost
        b. If not painted correctly:
            i. Find the nearest robot that either:
               - Already has the correct color, or
               - Can change to the correct color
            ii. Calculate Manhattan distance from robot to tile
            iii. Add cost for:
                 - Color change (if needed)
                 - Movement to tile
                 - Painting action
    2. Sum costs for all tiles, considering parallel actions by multiple robots
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal painting conditions
        self.goal_paintings = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Build adjacency graph for tiles
        self.adjacency = {}
        directions = ['up', 'down', 'left', 'right']
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] in directions:
                from_tile, to_tile = parts[1], parts[2]
                if from_tile not in self.adjacency:
                    self.adjacency[from_tile] = {}
                self.adjacency[from_tile][parts[0]] = to_tile

        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1] for fact in self.static 
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state

        # Extract current robot positions and colors
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot, tile = parts[1], parts[2]
                robots[robot] = {'tile': tile, 'color': None}
            elif match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                robot, color = parts[1], parts[2]
                if robot in robots:
                    robots[robot]['color'] = color

        # Extract current paintings
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                tile, color = parts[1], parts[2]
                current_paintings[tile] = color

        total_cost = 0

        for tile, goal_color in self.goal_paintings.items():
            # Skip if already painted correctly
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue

            min_robot_cost = float('inf')

            for robot, robot_info in robots.items():
                robot_tile = robot_info['tile']
                robot_color = robot_info['color']

                # Calculate movement cost (Manhattan distance)
                x1, y1 = map(int, robot_tile.split('_')[1:])
                x2, y2 = map(int, tile.split('_')[1:])
                movement_cost = abs(x1 - x2) + abs(y1 - y2)

                # Calculate color change cost
                color_cost = 0 if robot_color == goal_color else 1

                # Total cost for this robot to paint this tile
                total_robot_cost = movement_cost + color_cost + 1  # +1 for paint action

                if total_robot_cost < min_robot_cost:
                    min_robot_cost = total_robot_cost

            total_cost += min_robot_cost

        return total_cost
