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
    - Unpainted tiles that need to be painted
    - Tiles painted with wrong colors that need repainting
    - Robot movements required to reach unpainted tiles
    - Color changes needed for robots

    # Assumptions:
    - Robots can only paint adjacent tiles (up/down/left/right)
    - Each painting action requires the robot to be adjacent to the tile
    - Changing color takes one action
    - Moving to an adjacent tile takes one action

    # Heuristic Initialization
    - Extract goal painting conditions (which tiles need which colors)
    - Extract static information about tile adjacency relationships
    - Identify available colors

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (or repainted):
       - Calculate the Manhattan distance from the nearest robot
       - Add 1 action for the painting operation
       - If the robot doesn't have the required color, add 1 action for color change
    2. For tiles that are already correctly painted, no actions are needed
    3. The heuristic value is the sum of all these estimated actions
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
        
        # Build adjacency graph for movement estimation
        self.adjacency = {}
        directions = ['up', 'down', 'left', 'right']
        for fact in self.static:
            for direction in directions:
                if match(fact, direction, "*", "*"):
                    parts = get_parts(fact)
                    from_tile, to_tile = parts[1], parts[2]
                    if from_tile not in self.adjacency:
                        self.adjacency[from_tile] = []
                    self.adjacency[from_tile].append((to_tile, direction))
        
        # Extract available colors
        self.available_colors = set()
        for fact in self.static:
            if match(fact, "available-color", "*"):
                parts = get_parts(fact)
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state
        
        # Find all robots and their current positions/colors
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
        
        # Find current paintings
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                tile, color = parts[1], parts[2]
                current_paintings[tile] = color
        
        total_cost = 0
        
        # For each tile that needs to be painted in the goal
        for tile, goal_color in self.goal_paintings.items():
            current_color = current_paintings.get(tile, None)
            
            # If already correctly painted, no cost
            if current_color == goal_color:
                continue
                
            # Find the closest robot to this tile
            min_distance = float('inf')
            closest_robot = None
            closest_robot_color = None
            
            for robot, info in robots.items():
                robot_pos = info['pos']
                distance = self.manhattan_distance(robot_pos, tile)
                if distance < min_distance:
                    min_distance = distance
                    closest_robot = robot
                    closest_robot_color = info['color']
            
            # Add movement cost (Manhattan distance)
            total_cost += min_distance
            
            # Add painting cost (1 action)
            total_cost += 1
            
            # Add color change cost if needed (1 action)
            if closest_robot_color != goal_color:
                total_cost += 1
        
        return total_cost
    
    def manhattan_distance(self, tile1, tile2):
        """Calculate Manhattan distance between two tiles based on their names."""
        try:
            # Parse coordinates from tile names (assuming format tile_X_Y)
            x1, y1 = map(int, tile1.split('_')[1:])
            x2, y2 = map(int, tile2.split('_')[1:])
            return abs(x1 - x2) + abs(y1 - y2)
        except:
            # Fallback for unexpected tile naming - use BFS
            return self.bfs_distance(tile1, tile2)
    
    def bfs_distance(self, start, goal):
        """Calculate shortest path distance between two tiles using BFS."""
        if start == goal:
            return 0
            
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            if current == goal:
                return distance
                
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor, _ in self.adjacency.get(current, []):
                queue.append((neighbor, distance + 1))
        
        # If no path found (shouldn't happen in valid problems)
        return float('inf')
