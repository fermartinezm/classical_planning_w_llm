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
    - The distance robots need to move to reach unpainted tiles
    - The color changes required for robots to have the correct paint color
    - The painting actions needed for each tile

    # Assumptions
    - Robots can move freely between adjacent tiles (up/down/left/right)
    - Each painting action requires the robot to be adjacent to the tile
    - Color changes take one action each
    - Multiple robots can work in parallel if available

    # Heuristic Initialization
    - Extract goal paint conditions for each tile
    - Build adjacency graph for tile movement
    - Identify available colors and robots

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs painting (not matching goal):
        a. Find the nearest robot that either:
            - Has the correct color already
            - Can change to the correct color
        b. Calculate Manhattan distance from robot to tile
        c. Add distance + 1 (for painting action)
        d. If robot needs color change, add 1 action
    2. For multiple robots, distribute tiles to minimize maximum cost per robot
    3. Sum all required actions (movement, painting, color changes)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal paint conditions
        self.goal_paint = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_paint[tile] = color
        
        # Build adjacency graph
        self.adjacent = {}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacent.setdefault(tile1, set()).add(tile2)
                self.adjacent.setdefault(tile2, set()).add(tile1)
            elif match(fact, "down", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacent.setdefault(tile1, set()).add(tile2)
                self.adjacent.setdefault(tile2, set()).add(tile1)
            elif match(fact, "left", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacent.setdefault(tile1, set()).add(tile2)
                self.adjacent.setdefault(tile2, set()).add(tile1)
            elif match(fact, "right", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacent.setdefault(tile1, set()).add(tile2)
                self.adjacent.setdefault(tile2, set()).add(tile1)
        
        # Available colors
        self.available_colors = {
            get_parts(fact)[1] for fact in self.static 
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        
        # Extract current robot positions and colors
        robots = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robots[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robot_colors[robot] = color
        
        # Find tiles that need painting
        tiles_to_paint = []
        current_paint = {}
        
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_paint[tile] = color
        
        for tile, goal_color in self.goal_paint.items():
            if current_paint.get(tile) != goal_color:
                tiles_to_paint.append((tile, goal_color))
        
        if not tiles_to_paint:
            return 0  # Goal state
        
        # For each tile, find closest robot that can paint it
        total_cost = 0
        
        for tile, goal_color in tiles_to_paint:
            min_cost = float('inf')
            
            for robot, robot_pos in robots.items():
                # Calculate movement distance
                distance = self._manhattan_distance(robot_pos, tile)
                
                # Check if color change is needed
                color_cost = 0
                if robot_colors.get(robot) != goal_color:
                    color_cost = 1  # One action to change color
                
                total_robot_cost = distance + 1 + color_cost  # +1 for painting
                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost
            
            total_cost += min_cost
        
        return total_cost
    
    def _manhattan_distance(self, tile1, tile2):
        """Estimate distance between two tiles using their coordinates."""
        try:
            # Parse coordinates from tile names (format tile_X_Y)
            x1, y1 = map(int, tile1.split('_')[1:])
            x2, y2 = map(int, tile2.split('_')[1:])
            return abs(x1 - x2) + abs(y1 - y2)
        except:
            # Fallback to BFS if tile naming doesn't follow expected pattern
            return self._bfs_distance(tile1, tile2)
    
    def _bfs_distance(self, start, goal):
        """Calculate shortest path distance between two tiles using BFS."""
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
            
            for neighbor in self.adjacent.get(current, set()):
                queue.append((neighbor, dist + 1))
        
        return float('inf')  # No path found (shouldn't happen in valid problems)
