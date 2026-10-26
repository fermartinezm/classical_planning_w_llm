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
    - The number of unpainted tiles that need to be painted
    - The distance robots need to move to reach unpainted tiles
    - The color changes needed for robots to match required colors

    # Assumptions
    - Robots can move freely between adjacent tiles (up/down/left/right)
    - Each painting action requires the robot to be adjacent to the tile
    - Color changes take one action each
    - Multiple robots can work in parallel if available

    # Heuristic Initialization
    - Extract goal painting conditions
    - Build adjacency graph for tiles from static facts
    - Identify available colors

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (goal condition):
       a. If already correctly painted, no cost
       b. If not painted or wrong color:
          i. Find nearest robot that can paint it (has correct color or can change)
          ii. Calculate movement distance (Manhattan distance)
          iii. Add cost for color change if needed
    2. Sum costs for all tiles, dividing by number of robots (parallelism)
    3. Add minimum color changes needed (at least one per color change)
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
        directions = ['up', 'down', 'left', 'right']
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] in directions:
                from_tile, to_tile = parts[1], parts[2]
                if from_tile not in self.adjacent:
                    self.adjacent[from_tile] = []
                self.adjacent[from_tile].append(to_tile)
        
        # Extract available colors
        self.available_colors = set()
        for fact in self.static:
            if match(fact, "available-color", "*"):
                parts = get_parts(fact)
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Compute heuristic estimate for given state."""
        state = node.state
        
        # Extract current robot positions and colors
        robots = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robots[parts[1]] = parts[2]
            elif match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                robot_colors[parts[1]] = parts[2]
        
        # Extract current paintings
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                current_paintings[parts[1]] = parts[2]
        
        total_cost = 0
        
        for tile, goal_color in self.goal_paintings.items():
            current_color = current_paintings.get(tile, None)
            
            # Skip if already correctly painted
            if current_color == goal_color:
                continue
                
            min_robot_cost = float('inf')
            
            # Find best robot for this tile
            for robot, robot_pos in robots.items():
                robot_color = robot_colors.get(robot, None)
                
                # Calculate movement distance (BFS)
                visited = {robot_pos: 0}
                queue = [robot_pos]
                found = False
                
                while queue and not found:
                    current = queue.pop(0)
                    if current == tile:
                        found = True
                        break
                    
                    for neighbor in self.adjacent.get(current, []):
                        if neighbor not in visited:
                            visited[neighbor] = visited[current] + 1
                            queue.append(neighbor)
                
                if not found:
                    continue  # Robot can't reach this tile
                
                movement_cost = visited[tile]
                color_cost = 0 if robot_color == goal_color else 1
                
                total_robot_cost = movement_cost + color_cost + 1  # +1 for paint action
                min_robot_cost = min(min_robot_cost, total_robot_cost)
            
            if min_robot_cost != float('inf'):
                total_cost += min_robot_cost
        
        # Normalize by number of robots (parallel work)
        if robots:
            total_cost = max(1, total_cost / len(robots))
        
        return total_cost
