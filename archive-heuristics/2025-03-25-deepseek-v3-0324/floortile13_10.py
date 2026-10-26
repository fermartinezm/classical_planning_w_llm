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
    with their correct colors. It considers:
    - The number of unpainted tiles that need to be painted
    - The distance robots need to move to reach unpainted tiles
    - The color changes needed for robots to have the correct paint color

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each robot can carry only one color at a time
    - Changing color takes one action
    - Painting a tile takes one action
    - Moving to an adjacent tile takes one action

    # Heuristic Initialization
    - Extract goal conditions (which tiles need which colors)
    - Build adjacency graph from static facts (up, down, left, right relations)
    - Extract available colors from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (from goal conditions):
        a. If already painted correctly, no cost
        b. If not painted correctly:
            i. Find nearest robot that can paint it (has correct color or can change color)
            ii. Calculate movement distance (Manhattan distance)
            iii. Add cost for movement, color change (if needed), and painting
    2. Sum all costs for all tiles that need painting
    3. Add minimal coordination cost when multiple robots could interfere
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

        # Build adjacency graph
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                up, down = get_parts(fact)[1], get_parts(fact)[2]
                self.adjacency.setdefault(down, {}).setdefault('up', []).append(up)
                self.adjacency.setdefault(up, {}).setdefault('down', []).append(down)
            elif match(fact, "left", "*", "*"):
                left, right = get_parts(fact)[1], get_parts(fact)[2]
                self.adjacency.setdefault(right, {}).setdefault('left', []).append(left)
                self.adjacency.setdefault(left, {}).setdefault('right', []).append(right)

        # Extract available colors
        self.available_colors = {get_parts(fact)[1] for fact in self.static 
                               if match(fact, "available-color", "*")}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current robot positions and colors
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, tile = get_parts(fact)[1], get_parts(fact)[2]
                robots[robot] = {'tile': tile, 'color': None}
            elif match(fact, "robot-has", "*", "*"):
                robot, color = get_parts(fact)[1], get_parts(fact)[2]
                if robot in robots:
                    robots[robot]['color'] = color

        # Extract currently painted tiles
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                tile, color = get_parts(fact)[1], get_parts(fact)[2]
                current_paintings[tile] = color

        total_cost = 0

        # For each tile that needs painting in the goal
        for tile, goal_color in self.goal_paintings.items():
            current_color = current_paintings.get(tile, None)
            
            # If already painted correctly, no cost
            if current_color == goal_color:
                continue

            # Find nearest robot that can paint this tile
            min_cost = float('inf')
            
            for robot, info in robots.items():
                robot_tile = info['tile']
                robot_color = info['color']
                
                # Calculate movement distance (Manhattan distance)
                x1, y1 = map(int, robot_tile.split('_')[1:])
                x2, y2 = map(int, tile.split('_')[1:])
                distance = abs(x1 - x2) + abs(y1 - y2)
                
                # Calculate color change cost if needed
                color_cost = 0 if robot_color == goal_color else 1
                
                # Total cost for this robot: move + color change + paint
                cost = distance + color_cost + 1
                
                if cost < min_cost:
                    min_cost = cost
            
            total_cost += min_cost

        return total_cost
