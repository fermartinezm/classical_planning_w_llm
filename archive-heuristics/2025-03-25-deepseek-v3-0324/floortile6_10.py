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
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each paint action requires the robot to be adjacent to the tile
    - Color changes are only needed when the robot doesn't have the required color
    - Multiple robots can work in parallel if available

    # Heuristic Initialization
    - Extract goal paint conditions for each tile
    - Build adjacency graph for tile movement
    - Identify available colors and robots

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (not yet at goal color):
        a. Find the nearest robot that can paint it (considering current color)
        b. Calculate Manhattan distance from robot to tile
        c. Add 1 action for painting
        d. If robot needs color change, add 1 action
    2. Sum all actions across all tiles
    3. Divide by number of robots (accounting for parallel work)
    4. Add minimum color changes needed (each robot may need at most one color change)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal paint conditions
        self.goal_paint = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile, color = parts[1], parts[2]
                self.goal_paint[tile] = color
        
        # Build adjacency graph
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                _, y, x = get_parts(fact)
                self.adjacency.setdefault(x, {}).setdefault('up', y)
            elif match(fact, "down", "*", "*"):
                _, y, x = get_parts(fact)
                self.adjacency.setdefault(x, {}).setdefault('down', y)
            elif match(fact, "left", "*", "*"):
                _, y, x = get_parts(fact)
                self.adjacency.setdefault(x, {}).setdefault('left', y)
            elif match(fact, "right", "*", "*"):
                _, y, x = get_parts(fact)
                self.adjacency.setdefault(x, {}).setdefault('right', y)
        
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
                robots[robot] = {'tile': tile}
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robots[robot]['color'] = color
        
        # Extract current paint status
        current_paint = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_paint[tile] = color
        
        # Identify tiles that need painting
        tiles_to_paint = []
        for tile, goal_color in self.goal_paint.items():
            if current_paint.get(tile) != goal_color:
                tiles_to_paint.append((tile, goal_color))
        
        if not tiles_to_paint:
            return 0  # Goal state
        
        # For each tile to paint, find the best robot to do it
        total_cost = 0
        
        for tile, goal_color in tiles_to_paint:
            min_cost = float('inf')
            
            for robot, info in robots.items():
                robot_tile = info['tile']
                robot_color = info['color']
                
                # Calculate movement distance (Manhattan distance)
                x1, y1 = self._parse_tile_coords(robot_tile)
                x2, y2 = self._parse_tile_coords(tile)
                distance = abs(x1 - x2) + abs(y1 - y2)
                
                # Calculate color change cost
                color_cost = 0 if robot_color == goal_color else 1
                
                # Total cost for this robot: move + paint + color change
                cost = distance + 1 + color_cost
                
                if cost < min_cost:
                    min_cost = cost
            
            total_cost += min_cost
        
        # Account for parallel work by dividing by number of robots (but at least 1)
        num_robots = max(1, len(robots))
        return (total_cost + num_robots - 1) // num_robots  # Ceiling division
    
    def _parse_tile_coords(self, tile_name):
        """Parse tile coordinates from tile name (e.g., 'tile_1_2' -> (1, 2))"""
        parts = tile_name.split('_')
        return int(parts[1]), int(parts[2])
