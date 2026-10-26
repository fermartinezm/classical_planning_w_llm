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
    with their correct colors. It considers:
    - The distance robots need to move to reach unpainted tiles
    - The color changes needed for robots to have the correct paint color
    - The painting actions themselves

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right)
    - Each robot can carry only one color at a time
    - Changing color takes one action
    - Painting a tile takes one action
    - Moving between adjacent tiles takes one action

    # Heuristic Initialization
    - Extract the goal painting conditions
    - Build adjacency graphs for tile movement
    - Identify available colors and robot capabilities

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (goal conditions):
       a. If already correctly painted, no cost
       b. Otherwise:
          i. Find the nearest robot that can paint it (either has correct color or can change)
          ii. Calculate movement distance from robot to tile
          iii. Add cost for any required color changes
          iv. Add cost for the painting action
    2. Sum all these costs to get the total heuristic estimate
    3. For multiple robots, distribute the work optimally (assign each tile to closest robot)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract adjacency relationships
        self.adjacency = {}
        self.tiles = set()
        
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacency.setdefault(tile2, {}).setdefault('up', []).append(tile1)
                self.tiles.update([tile1, tile2])
            elif match(fact, "down", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacency.setdefault(tile2, {}).setdefault('down', []).append(tile1)
                self.tiles.update([tile1, tile2])
            elif match(fact, "left", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacency.setdefault(tile2, {}).setdefault('left', []).append(tile1)
                self.tiles.update([tile1, tile2])
            elif match(fact, "right", "*", "*"):
                _, tile1, tile2 = get_parts(fact)
                self.adjacency.setdefault(tile2, {}).setdefault('right', []).append(tile1)
                self.tiles.update([tile1, tile2])
            elif match(fact, "available-color", "*"):
                _, color = get_parts(fact)
                self.available_colors = color

        # Extract goal painting conditions
        self.goal_paintings = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_paintings[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
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
        
        # Extract currently painted tiles
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_paintings[tile] = color
        
        total_cost = 0
        
        # For each tile that needs to be painted in the goal
        for tile, goal_color in self.goal_paintings.items():
            # Skip if already correctly painted
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue
                
            min_robot_cost = float('inf')
            
            # Find the best robot to paint this tile
            for robot, robot_pos in robots.items():
                robot_color = robot_colors.get(robot, None)
                
                # Calculate movement distance (BFS)
                visited = set()
                queue = [(robot_pos, 0)]
                movement_cost = float('inf')
                
                while queue:
                    current_tile, distance = queue.pop(0)
                    
                    if current_tile == tile:
                        movement_cost = distance
                        break
                    
                    if current_tile in visited:
                        continue
                    
                    visited.add(current_tile)
                    
                    # Explore adjacent tiles
                    for direction in ['up', 'down', 'left', 'right']:
                        for neighbor in self.adjacency.get(current_tile, {}).get(direction, []):
                            # Check if neighbor is clear (can move there)
                            if f'(clear {neighbor})' in state:
                                queue.append((neighbor, distance + 1))
                
                # If we can't reach the tile from this robot, skip
                if movement_cost == float('inf'):
                    continue
                
                # Calculate color change cost if needed
                color_change_cost = 0
                if robot_color != goal_color:
                    color_change_cost = 1
                
                # Total cost for this robot to paint this tile
                total_robot_cost = movement_cost + color_change_cost + 1  # +1 for paint action
                
                if total_robot_cost < min_robot_cost:
                    min_robot_cost = total_robot_cost
            
            if min_robot_cost != float('inf'):
                total_cost += min_robot_cost
        
        return total_cost
