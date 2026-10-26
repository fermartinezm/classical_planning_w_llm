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
    This heuristic estimates the number of actions needed to paint all tiles according to the goal pattern.
    It considers:
    - The number of unpainted tiles that need to be painted
    - The distance robots need to move to reach unpainted tiles
    - The color changes required for robots to have the correct paint color

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each painting action requires the robot to be adjacent to the tile
    - Color changes are only needed when the robot doesn't have the required color
    - Multiple robots can work in parallel if available

    # Heuristic Initialization
    - Extract the goal painting pattern from task.goals
    - Build adjacency maps (up, down, left, right) from static facts
    - Store available colors from static facts

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (from goal conditions):
        a. If already painted correctly in current state, no action needed
        b. Else:
            i. Find nearest robot that can paint it (considering current color)
            ii. Calculate Manhattan distance from robot to tile
            iii. Add 1 action for painting
            iv. If robot needs color change, add 1 action
    2. Sum all required actions:
        - Movement actions (distance)
        - Painting actions (1 per tile)
        - Color change actions (if needed)
    3. Divide by number of robots (since they can work in parallel)
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
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.up[x] = y
            elif match(fact, "down", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.down[x] = y
            elif match(fact, "left", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.left[x] = y
            elif match(fact, "right", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.right[x] = y
        
        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1] 
            for fact in self.static 
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        
        # Extract current robot positions and colors
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, tile = get_parts(fact)[1], get_parts(fact)[2]
                robots[robot] = {'pos': tile, 'color': None}
            elif match(fact, "robot-has", "*", "*"):
                robot, color = get_parts(fact)[1], get_parts(fact)[2]
                if robot in robots:
                    robots[robot]['color'] = color
        
        # Count number of robots
        num_robots = len(robots)
        if num_robots == 0:
            return float('inf')  # No robots available
        
        # Extract current paintings
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                tile, color = get_parts(fact)[1], get_parts(fact)[2]
                current_paintings[tile] = color
        
        total_cost = 0
        
        for tile, goal_color in self.goal_paintings.items():
            # Skip if already painted correctly
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue
                
            # Find nearest robot that can paint this tile
            min_dist = float('inf')
            best_robot = None
            color_change_needed = False
            
            for robot_name, robot in robots.items():
                # Calculate Manhattan distance
                dist = self._manhattan_distance(robot['pos'], tile)
                
                # Check if color change is needed
                needs_color_change = (robot['color'] != goal_color)
                
                # Prefer robots that don't need color change
                effective_dist = dist + (1 if needs_color_change else 0)
                
                if effective_dist < min_dist:
                    min_dist = effective_dist
                    best_robot = robot_name
                    color_change_needed = needs_color_change
            
            # Add costs: movement + painting + color change (if needed)
            total_cost += min_dist + 1  # 1 for painting action
        
        # Divide by number of robots to account for parallelism
        if num_robots > 0:
            total_cost = max(1, total_cost // num_robots)
        
        return total_cost

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two tiles based on their coordinates."""
        # Extract coordinates from tile names (format tile_X_Y)
        try:
            x1, y1 = map(int, pos1.split('_')[1:])
            x2, y2 = map(int, pos2.split('_')[1:])
            return abs(x1 - x2) + abs(y1 - y2)
        except:
            # Fallback for unexpected tile naming
            return 1
