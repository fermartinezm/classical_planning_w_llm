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
    - The number of unpainted tiles that need painting
    - The distance robots need to move to reach unpainted tiles
    - The color changes needed for robots to have the correct paint color

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each paint action requires the robot to be adjacent to the tile
    - Color changes take one action each
    - Multiple robots can work in parallel if available

    # Heuristic Initialization
    - Extract goal paint conditions for each tile
    - Build adjacency maps for tile movement
    - Identify available colors and robots

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs painting (not yet correct color in goal):
       a. Find the nearest robot that can paint it (has correct color or can change)
       b. Calculate Manhattan distance from robot to tile
       c. Add cost for any required color changes
    2. Sum costs for all unpainted tiles
    3. Add minimal movement costs (assuming optimal parallel movement)
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
                self.goal_paint[parts[1]] = parts[2]
        
        # Build adjacency maps
        self.adjacent = {}  # tile -> set of adjacent tiles
        self.all_tiles = set()
        
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacent.setdefault(tile2, set()).add(tile1)
                self.all_tiles.update([tile1, tile2])
            elif match(fact, "down", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacent.setdefault(tile1, set()).add(tile2)
                self.all_tiles.update([tile1, tile2])
            elif match(fact, "left", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacent.setdefault(tile1, set()).add(tile2)
                self.all_tiles.update([tile1, tile2])
            elif match(fact, "right", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacent.setdefault(tile2, set()).add(tile1)
                self.all_tiles.update([tile1, tile2])
        
        # Extract available colors
        self.available_colors = {
            parts[1] for fact in self.static 
            if match(fact, "available-color", "*") 
            for parts in [get_parts(fact)]
        }

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        
        # Extract current robot positions and colors
        robot_pos = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot_pos[parts[1]] = parts[2]
            elif match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                robot_colors[parts[1]] = parts[2]
        
        # Extract currently painted tiles
        current_paint = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                current_paint[parts[1]] = parts[2]
        
        # Find tiles that need painting (not painted or wrong color)
        tiles_to_paint = []
        for tile, goal_color in self.goal_paint.items():
            current_color = current_paint.get(tile, None)
            if current_color != goal_color:
                tiles_to_paint.append((tile, goal_color))
        
        if not tiles_to_paint:
            return 0  # Goal state
        
        # For each tile to paint, find minimal cost robot
        total_cost = 0
        
        for tile, goal_color in tiles_to_paint:
            min_cost = float('inf')
            
            for robot, pos in robot_pos.items():
                current_color = robot_colors.get(robot, None)
                
                # Calculate movement cost (Manhattan distance)
                x1, y1 = map(int, pos.split('_')[1:])
                x2, y2 = map(int, tile.split('_')[1:])
                move_cost = abs(x1 - x2) + abs(y1 - y2)
                
                # Calculate color change cost
                color_cost = 0 if current_color == goal_color else 1
                
                total_robot_cost = move_cost + color_cost + 1  # +1 for paint action
                
                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost
            
            total_cost += min_cost
        
        return total_cost
