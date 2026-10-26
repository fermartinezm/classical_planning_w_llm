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
    - The number of unpainted tiles that need to be painted
    - The distance between robots and unpainted tiles
    - Whether color changes are needed
    - The current color of robots

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each painting action requires the robot to be adjacent to the tile
    - Color changes take one action per change
    - Multiple robots can work in parallel (though heuristic sums all costs)

    # Heuristic Initialization
    - Extract goal painting conditions (which tiles need which colors)
    - Build adjacency graph from static facts (up/down/left/right relationships)
    - Extract available colors and initial robot positions/colors

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (goal conditions):
        a. If already painted correctly, no cost
        b. Else:
            i. Find closest robot that can paint it (either has correct color or can change)
            ii. Calculate movement cost (Manhattan distance)
            iii. Add color change cost if needed
            iv. Add painting cost (1 action)
    2. Sum all costs across all tiles
    3. For robots that need to change color but aren't assigned to paint anything:
        a. Add color change cost if they're not holding a color needed elsewhere
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
                from_tile, to_tile = parts[1], parts[2]
                if from_tile not in self.adjacent:
                    self.adjacent[from_tile] = []
                self.adjacent[from_tile].append(to_tile)
                if to_tile not in self.adjacent:
                    self.adjacent[to_tile] = []
                self.adjacent[to_tile].append(from_tile)
        
        # Extract available colors
        self.available_colors = set()
        for fact in self.static:
            if match(fact, "available-color", "*"):
                parts = get_parts(fact)
                self.available_colors.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        
        # Extract current robot positions and colors
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
        
        # Extract currently painted tiles
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                parts = get_parts(fact)
                tile, color = parts[1], parts[2]
                current_paintings[tile] = color
        
        total_cost = 0
        used_robots = set()
        
        # For each tile that needs painting
        for tile, goal_color in self.goal_paintings.items():
            # Skip if already painted correctly
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue
                
            # Find closest robot that can paint this tile
            min_cost = float('inf')
            best_robot = None
            
            for robot, data in robots.items():
                if robot in used_robots:
                    continue
                    
                pos = data['pos']
                color = data['color']
                
                # Calculate movement cost (BFS for shortest path)
                visited = {pos: 0}
                queue = [pos]
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
                    continue  # No path from robot to tile
                    
                movement_cost = visited[tile]
                color_cost = 0 if color == goal_color else 1
                total_robot_cost = movement_cost + color_cost + 1  # +1 for painting
                
                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost
                    best_robot = robot
            
            if best_robot is not None:
                total_cost += min_cost
                used_robots.add(best_robot)
        
        # Add cost for robots that need to change color but aren't assigned
        for robot, data in robots.items():
            if robot not in used_robots and data['color'] is not None:
                # Check if this color is needed elsewhere
                color_needed = any(
                    c == data['color'] 
                    for t, c in self.goal_paintings.items() 
                    if t not in current_paintings or current_paintings[t] != c
                )
                if not color_needed and self.available_colors:
                    # Change to any available color (minimum 1 action)
                    total_cost += 1
        
        return total_cost
