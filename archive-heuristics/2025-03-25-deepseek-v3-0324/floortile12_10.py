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
    - The color changes needed for robots to have the correct paint color
    - The painting actions themselves

    # Assumptions
    - Robots can move freely between adjacent tiles (up/down/left/right)
    - Each robot can carry only one color at a time
    - Changing color takes one action
    - Painting a tile takes one action
    - Moving to an adjacent tile takes one action

    # Heuristic Initialization
    - Extract goal painting conditions
    - Build adjacency graph from static facts (up/down/left/right relationships)
    - Identify available colors

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (goal conditions):
        a. If already painted correctly, no cost
        b. Else:
            i. Find the nearest robot that either:
                - Has the correct color, or
                - Can change to the correct color
            ii. Calculate Manhattan distance from robot to tile
            iii. Add cost for:
                - Moving to tile (distance)
                - Changing color if needed (1 action)
                - Painting (1 action)
    2. Sum costs for all tiles, considering robot assignments to minimize total cost
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
        self.adjacent = {}
        for fact in self.static:
            if match(fact, "up", "*", "*") or match(fact, "down", "*", "*") or \
               match(fact, "left", "*", "*") or match(fact, "right", "*", "*"):
                parts = get_parts(fact)
                tile1, tile2 = parts[1], parts[2]
                if tile1 not in self.adjacent:
                    self.adjacent[tile1] = []
                self.adjacent[tile1].append(tile2)
        
        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1] 
            for fact in self.static 
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state
        
        # Get current robot positions and colors
        robots = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot, tile = parts[1], parts[2]
                robots[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
        
        # Get currently painted tiles
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                tile, color = parts[1], parts[2]
                current_paintings[tile] = color
        
        total_cost = 0
        
        # For each tile that needs painting in the goal
        for tile, goal_color in self.goal_paintings.items():
            # Skip if already painted correctly
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue
            
            min_robot_cost = float('inf')
            
            # Find best robot to paint this tile
            for robot, robot_pos in robots.items():
                robot_color = robot_colors.get(robot, None)
                
                # Calculate movement cost (Manhattan distance)
                movement_cost = self.manhattan_distance(robot_pos, tile)
                
                # Calculate color change cost if needed
                color_cost = 0
                if robot_color != goal_color:
                    color_cost = 1  # change_color action
                
                # Painting cost
                paint_cost = 1
                
                total_robot_cost = movement_cost + color_cost + paint_cost
                
                if total_robot_cost < min_robot_cost:
                    min_robot_cost = total_robot_cost
            
            total_cost += min_robot_cost
        
        return total_cost
    
    def manhattan_distance(self, tile1, tile2):
        """Calculate Manhattan distance between two tiles based on their names."""
        try:
            # Parse coordinates from tile names (format tile_X_Y)
            x1, y1 = map(int, tile1.split('_')[1:])
            x2, y2 = map(int, tile2.split('_')[1:])
            return abs(x1 - x2) + abs(y1 - y2)
        except:
            # Fallback to BFS if tile names don't follow expected format
            return self.bfs_distance(tile1, tile2)
    
    def bfs_distance(self, start, goal):
        """Calculate shortest path distance using BFS on the adjacency graph."""
        if start == goal:
            return 0
            
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == goal:
                return dist
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.adjacent.get(current, []):
                queue.append((neighbor, dist + 1))
        
        # If no path found (shouldn't happen in valid problems)
        return float('inf')
