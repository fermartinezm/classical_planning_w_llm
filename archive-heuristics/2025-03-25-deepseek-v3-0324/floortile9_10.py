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
    This heuristic estimates the number of actions needed to paint all tiles according to the goal pattern.
    It considers:
    - The distance robots need to move to reach unpainted tiles
    - The color changes required for robots to have the correct paint color
    - The painting actions needed for each tile

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each robot can carry only one color at a time
    - Changing color takes one action
    - Moving to an adjacent tile takes one action
    - Painting a tile takes one action

    # Heuristic Initialization
    - Extract the goal painting pattern (which tiles need which colors)
    - Build adjacency graphs for tile movement (up/down/left/right relationships)
    - Identify available colors from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (according to goals):
        a. If already painted correctly, no cost
        b. Else:
            i. Find nearest robot that either:
                - Already has the correct color, or
                - Can change to the correct color
            ii. Calculate movement distance (Manhattan distance)
            iii. Add cost for:
                - Color change (if needed)
                - Movement to tile
                - Painting action
    2. Sum all costs while trying to optimize robot assignments
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal painting requirements
        self.goal_paintings = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color
        
        # Build adjacency graph for movement
        self.adjacency = {}  # tile -> {direction: neighbor}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                _, y, x = get_parts(fact)
                self.adjacency.setdefault(x, {})["up"] = y
                self.adjacency.setdefault(y, {})["down"] = x
            elif match(fact, "left", "*", "*"):
                _, x, y = get_parts(fact)
                self.adjacency.setdefault(y, {})["left"] = x
                self.adjacency.setdefault(x, {})["right"] = y
        
        # Get available colors
        self.available_colors = {
            get_parts(fact)[1] 
            for fact in self.static 
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        
        # Extract current robot positions and colors
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robots[robot] = {"tile": tile, "color": None}
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                if robot in robots:
                    robots[robot]["color"] = color
        
        # Extract already painted tiles
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_paintings[tile] = color
        
        total_cost = 0
        assigned_robots = set()
        
        for tile, goal_color in self.goal_paintings.items():
            # Skip if already painted correctly
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue
            
            # Find best robot for this tile
            best_cost = float('inf')
            best_robot = None
            
            for robot, data in robots.items():
                if robot in assigned_robots:
                    continue
                
                robot_tile = data["tile"]
                robot_color = data["color"]
                
                # Calculate movement distance (Manhattan distance)
                x1, y1 = map(int, robot_tile.split('_')[1:])
                x2, y2 = map(int, tile.split('_')[1:])
                distance = abs(x1 - x2) + abs(y1 - y2)
                
                # Calculate color change cost
                color_cost = 0 if robot_color == goal_color else 1
                
                total_robot_cost = distance + color_cost + 1  # +1 for painting
                
                if total_robot_cost < best_cost:
                    best_cost = total_robot_cost
                    best_robot = robot
            
            if best_robot is not None:
                total_cost += best_cost
                assigned_robots.add(best_robot)
            else:
                # No unassigned robot left - assign to closest robot with highest cost
                min_distance = float('inf')
                for robot, data in robots.items():
                    robot_tile = data["tile"]
                    x1, y1 = map(int, robot_tile.split('_')[1:])
                    x2, y2 = map(int, tile.split('_')[1:])
                    distance = abs(x1 - x2) + abs(y1 - y2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        robot_color = data["color"]
                        color_cost = 0 if robot_color == goal_color else 1
                        best_cost = distance + color_cost + 1
                
                total_cost += best_cost
        
        return total_cost
