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
    This heuristic estimates the number of actions needed to paint all required tiles
    with their goal colors. It considers:
    - The distance robots need to move to reach unpainted tiles
    - The color changes needed for robots to have the correct paint color
    - The painting actions themselves

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each painting action requires the robot to be adjacent to the tile
    - Color changes take one action and can be done anywhere
    - Multiple robots can work in parallel if available

    # Heuristic Initialization
    - Extract goal painting conditions
    - Build adjacency graph from static facts for movement calculations
    - Identify available colors

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (goal conditions):
       a. If already painted correctly, no cost
       b. Otherwise:
          i. Find nearest robot that can paint it (with correct color or can change color)
          ii. Calculate movement distance (Manhattan distance)
          iii. Add cost for color change if needed
          iv. Add cost for painting action
    2. Sum costs for all tiles, divided by number of robots (parallelism)
    3. Add minimum color changes needed (at least one per robot that needs to change color)
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

        # Build adjacency graph for movement
        self.adjacency = {}
        directions = ['up', 'down', 'left', 'right']
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] in directions:
                direction, tile1, tile2 = parts
                if tile1 not in self.adjacency:
                    self.adjacency[tile1] = {}
                self.adjacency[tile1][direction] = tile2

        # Identify available colors
        self.available_colors = {
            parts[1] for fact in self.static 
            if match(fact, "available-color", "*") 
            for parts in [get_parts(fact)]
        }

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state

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

        # Extract already painted tiles
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                tile, color = parts[1], parts[2]
                current_paintings[tile] = color

        total_cost = 0

        # For each tile that needs painting
        for tile, goal_color in self.goal_paintings.items():
            # Skip if already painted correctly
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue

            # Find nearest robot that can paint this tile
            min_cost = float('inf')
            for robot, data in robots.items():
                robot_pos = data['pos']
                robot_color = data['color']

                # Calculate movement distance (Manhattan distance approximation)
                x1, y1 = map(int, robot_pos.split('_')[1:])
                x2, y2 = map(int, tile.split('_')[1:])
                distance = abs(x1 - x2) + abs(y1 - y2)

                # Cost for color change if needed
                color_cost = 0
                if robot_color != goal_color and goal_color in self.available_colors:
                    color_cost = 1

                # Total cost for this robot
                cost = distance + color_cost + 1  # +1 for painting action
                if cost < min_cost:
                    min_cost = cost

            if min_cost != float('inf'):
                total_cost += min_cost

        # Normalize by number of robots (parallel work)
        if robots:
            total_cost = max(1, total_cost / len(robots))

        return total_cost
