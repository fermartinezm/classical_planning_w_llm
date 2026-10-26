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
    This heuristic estimates the number of actions needed to paint all tiles according to the goal pattern.
    It considers:
    - The number of tiles that still need to be painted (or repainted)
    - The distance robots need to move to reach unpainted tiles
    - The color changes needed for robots to have the correct colors

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each painting action requires the robot to be adjacent to the tile
    - Color changes take one action each
    - Multiple robots can work in parallel if available

    # Heuristic Initialization
    - Extract the goal painting pattern (which tiles need which colors)
    - Build adjacency graphs for movement between tiles
    - Extract available colors and initial robot positions

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (or repainted):
       a) Find the nearest robot that can paint it (has correct color or can change color)
       b) Calculate Manhattan distance between robot and tile
       c) Add 1 action for painting and distance actions for movement
       d) If robot needs color change, add 1 action
    2. Sum all these costs across all tiles
    3. For multiple robots, divide the work among them (approximate parallelization)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal painting pattern
        self.goal_paintings = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color
        
        # Build adjacency graph for movement
        self.adjacent = {}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                parts = get_parts(fact)
                self.adjacent.setdefault(parts[1], set()).add(parts[2])
            elif match(fact, "down", "*", "*"):
                parts = get_parts(fact)
                self.adjacent.setdefault(parts[1], set()).add(parts[2])
            elif match(fact, "left", "*", "*"):
                parts = get_parts(fact)
                self.adjacent.setdefault(parts[1], set()).add(parts[2])
            elif match(fact, "right", "*", "*"):
                parts = get_parts(fact)
                self.adjacent.setdefault(parts[1], set()).add(parts[2])
        
        # Extract available colors
        self.available_colors = {
            parts[1] for fact in self.static 
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        
        # Extract current robot positions and colors
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robots[parts[1]] = {'pos': parts[2], 'color': None}
            elif match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                if parts[1] in robots:
                    robots[parts[1]]['color'] = parts[2]
        
        # Extract current paintings
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                current_paintings[parts[1]] = parts[2]
        
        # Calculate required work
        total_cost = 0
        robot_costs = {robot: 0 for robot in robots}
        
        for tile, goal_color in self.goal_paintings.items():
            current_color = current_paintings.get(tile)
            
            # Skip if already correctly painted
            if current_color == goal_color:
                continue
                
            # Find nearest robot that can paint this tile
            min_cost = float('inf')
            best_robot = None
            
            for robot, data in robots.items():
                robot_pos = data['pos']
                robot_color = data['color']
                
                # Calculate movement cost (Manhattan distance)
                x1, y1 = map(int, robot_pos.split('_')[1:])
                x2, y2 = map(int, tile.split('_')[1:])
                distance = abs(x1 - x2) + abs(y1 - y2)
                
                # Calculate color change cost if needed
                color_cost = 0 if robot_color == goal_color else 1
                
                total_robot_cost = distance + color_cost + 1  # +1 for painting
                
                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost
                    best_robot = robot
            
            if best_robot:
                robot_costs[best_robot] += min_cost
        
        # Sum up costs, approximating parallel work by taking max robot cost
        if robot_costs:
            total_cost = max(robot_costs.values())
        
        return total_cost
