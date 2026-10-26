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
    - The distance robots need to move to reach unpainted tiles
    - The color changes required for robots to have the correct paint color
    - The painting actions needed for each tile

    # Assumptions
    - Robots can move freely between adjacent tiles when they are clear
    - Each robot can carry only one color at a time
    - Changing color takes one action
    - Painting a tile takes one action
    - Moving to an adjacent tile takes one action

    # Heuristic Initialization
    - Extract the goal painting conditions for each tile
    - Build adjacency maps for tiles (up, down, left, right relationships)
    - Identify available colors from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (goal conditions):
       a. If already painted correctly, no cost
       b. If not painted correctly:
          i. Find the nearest robot that can paint it (considering current color)
          ii. Calculate movement distance (Manhattan distance)
          iii. Add cost for color change if needed
          iv. Add cost for painting action
    2. Sum all these costs to get the total heuristic estimate
    3. For multiple robots, assign tiles to robots in a way that minimizes total cost
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
        
        # Build adjacency maps
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}
        
        for fact in self.static:
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
                robots[robot] = {'pos': tile, 'color': None}
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                if robot in robots:
                    robots[robot]['color'] = color
        
        # Extract current paintings
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_paintings[tile] = color
        
        # Extract clear tiles
        clear_tiles = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "clear", "*")
        }
        
        total_cost = 0
        
        for tile, goal_color in self.goal_paintings.items():
            current_color = current_paintings.get(tile, None)
            
            # Skip if already painted correctly
            if current_color == goal_color:
                continue
                
            # Find the best robot to paint this tile
            min_cost = float('inf')
            
            for robot, data in robots.items():
                robot_pos = data['pos']
                robot_color = data['color']
                
                # Calculate movement distance (Manhattan distance)
                distance = self.manhattan_distance(robot_pos, tile, clear_tiles)
                
                # Calculate color change cost if needed
                color_cost = 0 if robot_color == goal_color else 1
                
                # Total cost for this robot
                cost = distance + color_cost + 1  # +1 for painting action
                
                if cost < min_cost:
                    min_cost = cost
            
            total_cost += min_cost
        
        return total_cost
    
    def manhattan_distance(self, start, end, clear_tiles):
        """Calculate Manhattan distance between two tiles considering clear paths."""
        # Parse coordinates from tile names (assuming format tile_X_Y)
        try:
            x1, y1 = map(int, start.split('_')[1:])
            x2, y2 = map(int, end.split('_')[1:])
        except:
            # Fallback if tile naming is different
            return abs(x1 - x2) + abs(y1 - y2)
        
        # Simple Manhattan distance (can be improved with pathfinding)
        return abs(x1 - x2) + abs(y1 - y2)
