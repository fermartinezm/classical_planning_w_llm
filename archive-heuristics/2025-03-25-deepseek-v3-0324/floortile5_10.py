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
    - The distance robots need to move to reach unpainted tiles
    - The need to change colors when the current color doesn't match the required color
    - The painting actions themselves

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each painting action requires the robot to be adjacent to the tile
    - Color changes take one action and can be done at any time
    - Multiple robots can work in parallel if available

    # Heuristic Initialization
    - Extract goal conditions (which tiles need which colors)
    - Build adjacency graphs for movement between tiles
    - Identify available colors and robot starting positions

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (goal condition):
        a. If already correctly painted, no cost
        b. Else:
            i. Find the nearest robot that either:
                - Has the correct color, or
                - Can change to the correct color
            ii. Calculate Manhattan distance from robot to tile
            iii. Add cost for movement (1 per tile)
            iv. Add cost for color change if needed (1)
            v. Add cost for painting action (1)
    2. For multiple robots, distribute the work to minimize total cost
    3. Return the sum of all required actions
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
        for fact in self.static:
            if match(fact, "up", "*", "*") or match(fact, "down", "*", "*") or \
               match(fact, "left", "*", "*") or match(fact, "right", "*", "*"):
                parts = get_parts(fact)
                direction, tile1, tile2 = parts[0], parts[1], parts[2]
                if tile1 not in self.adjacency:
                    self.adjacency[tile1] = []
                self.adjacency[tile1].append(tile2)
        
        # Extract available colors
        self.available_colors = set()
        for fact in self.static:
            if match(fact, "available-color", "*"):
                parts = get_parts(fact)
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        
        # If all goals are satisfied, return 0
        if all(goal in state for goal in self.goals):
            return 0
        
        # Track robot positions and colors
        robot_info = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot, tile = parts[1], parts[2]
                robot_info[robot] = {"pos": tile, "color": None}
            elif match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                robot, color = parts[1], parts[2]
                if robot in robot_info:
                    robot_info[robot]["color"] = color
        
        # For each tile that needs painting, find the minimal cost
        total_cost = 0
        for tile, req_color in self.goal_paintings.items():
            # Skip if already correctly painted
            if f"(painted {tile} {req_color})" in state:
                continue
                
            min_cost = float('inf')
            
            # Find the best robot for this tile
            for robot, info in robot_info.items():
                robot_pos = info["pos"]
                robot_color = info["color"]
                
                # Calculate movement cost (Manhattan distance)
                x1, y1 = map(int, robot_pos.split('_')[1:])
                x2, y2 = map(int, tile.split('_')[1:])
                move_cost = abs(x1 - x2) + abs(y1 - y2)
                
                # Calculate color change cost if needed
                color_cost = 0 if robot_color == req_color else 1
                
                # Painting action cost
                paint_cost = 1
                
                total_robot_cost = move_cost + color_cost + paint_cost
                
                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost
            
            # If no robot can reach (shouldn't happen in valid states), use a base cost
            if min_cost == float('inf'):
                min_cost = 1  # At least need to paint it
                
            total_cost += min_cost
        
        return total_cost
