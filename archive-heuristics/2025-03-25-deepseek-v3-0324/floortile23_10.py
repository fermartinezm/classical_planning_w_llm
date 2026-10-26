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
    This heuristic estimates the number of actions needed to paint all required tiles
    with their goal colors. It considers:
    - The number of unpainted tiles that need to be painted
    - The distance robots need to move to reach unpainted tiles
    - The color changes needed for robots to have the correct paint color

    # Assumptions
    - Robots can only paint adjacent tiles (up/down/left/right)
    - Each robot can carry only one color at a time
    - Changing color takes one action
    - Moving to an adjacent tile takes one action
    - Painting a tile takes one action

    # Heuristic Initialization
    - Extract goal paint conditions for each tile
    - Build adjacency graph for tiles from static facts
    - Identify available colors from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (not yet at goal color):
        a. Find the nearest robot that can paint it (has correct color or can change color)
        b. Calculate Manhattan distance from robot to tile
        c. Add 1 for painting action
        d. Add 1 if robot needs to change color
    2. Sum these costs for all unpainted tiles
    3. If multiple robots can paint a tile, use the one with minimal cost
    4. For efficiency, we don't track robot collisions but assume optimal coordination
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal paint conditions
        self.goal_paint = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile, color = parts[1], parts[2]
                self.goal_paint[tile] = color

        # Build adjacency graph
        self.adjacent = {}
        directions = ['up', 'down', 'left', 'right']
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in directions:
                dir, tile1, tile2 = parts[0], parts[1], parts[2]
                if tile1 not in self.adjacent:
                    self.adjacent[tile1] = {}
                self.adjacent[tile1][dir] = tile2

        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1] 
            for fact in static_facts 
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state

        # If all goals are satisfied, return 0
        if self.goals <= state:
            return 0

        # Extract current robot positions and colors
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot, tile = parts[1], parts[2]
                robots[robot] = {'pos': tile, 'color': None}
            elif match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                robot, color = parts[1], parts[2]
                if robot in robots:
                    robots[robot]['color'] = color

        # Extract currently painted tiles
        current_paint = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                tile, color = parts[1], parts[2]
                current_paint[tile] = color

        total_cost = 0

        # For each tile that needs painting
        for tile, goal_color in self.goal_paint.items():
            # Skip if already correctly painted
            if tile in current_paint and current_paint[tile] == goal_color:
                continue

            min_cost = float('inf')

            # Find the best robot to paint this tile
            for robot, info in robots.items():
                robot_pos = info['pos']
                robot_color = info['color']

                # Calculate Manhattan distance
                try:
                    x1, y1 = map(int, robot_pos.split('_')[1:])
                    x2, y2 = map(int, tile.split('_')[1:])
                    distance = abs(x1 - x2) + abs(y1 - y2)
                except:
                    # Fallback if tile naming doesn't follow expected pattern
                    distance = 1  # Assume adjacent

                # Cost components
                move_cost = distance
                color_cost = 0 if robot_color == goal_color else 1
                paint_cost = 1

                total_robot_cost = move_cost + color_cost + paint_cost

                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost

            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost
