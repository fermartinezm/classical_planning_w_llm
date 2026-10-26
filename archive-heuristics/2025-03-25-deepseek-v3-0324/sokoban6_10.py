from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to solve a Sokoban puzzle by:
    1. Calculating the Manhattan distance between each box and its goal position
    2. Calculating the distance between the robot and each box
    3. Considering whether boxes are already in goal positions
    4. Adding penalties for boxes that are not in goal positions but block the path

    # Assumptions:
    - Each box has exactly one goal position (standard Sokoban)
    - The grid is rectangular and coordinates can be extracted from location names
    - Pushing a box always moves it one step closer to its goal
    - The robot needs to reach each box at least once

    # Heuristic Initialization
    - Extract goal positions for boxes from task.goals
    - Build adjacency graph from static facts to enable pathfinding
    - Parse location coordinates from their names (format loc_X_Y)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not in its goal position:
        a) Calculate Manhattan distance from current position to goal
        b) Find the robot's distance to the box (using BFS if needed)
        c) Add these distances to the total cost
    2. For boxes already in goal positions, no cost is added
    3. Add a small penalty for each box that might block the path
    4. Return the sum of all distances and penalties as the heuristic value
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building adjacency graph."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal locations for boxes
        self.box_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, box, loc = get_parts(goal)
                self.box_goals[box] = loc
        
        # Build adjacency graph from static facts
        self.adjacency = defaultdict(set)
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.adjacency[loc1].add(loc2)
                self.adjacency[loc2].add(loc1)
        
        # Precompute coordinates for locations
        self.loc_coords = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                for loc in (get_parts(fact)[1], get_parts(fact)[2]):
                    if loc not in self.loc_coords:
                        try:
                            # Parse coordinates from location name (format loc_X_Y)
                            parts = loc.split('_')
                            x, y = int(parts[1]), int(parts[2])
                            self.loc_coords[loc] = (x, y)
                        except:
                            # Fallback if location names don't follow expected format
                            self.loc_coords[loc] = (0, 0)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        
        # Find current positions of robot and boxes
        robot_pos = None
        box_positions = {}
        
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_pos = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_positions[box] = loc
        
        if not robot_pos or not box_positions:
            return 0  # No boxes or robot not found
        
        total_cost = 0
        
        for box, current_loc in box_positions.items():
            if box not in self.box_goals:
                continue  # Box has no goal (shouldn't happen in standard Sokoban)
                
            goal_loc = self.box_goals[box]
            
            if current_loc == goal_loc:
                continue  # Box is already at goal
                
            # Calculate Manhattan distance from current position to goal
            if current_loc in self.loc_coords and goal_loc in self.loc_coords:
                x1, y1 = self.loc_coords[current_loc]
                x2, y2 = self.loc_coords[goal_loc]
                box_distance = abs(x1 - x2) + abs(y1 - y2)
            else:
                # Fallback: assume distance is 1 if coordinates can't be parsed
                box_distance = 1
                
            # Estimate robot's distance to the box
            robot_to_box = self.estimate_distance(robot_pos, current_loc)
            
            # Add to total cost
            total_cost += robot_to_box + box_distance
            
            # Small penalty for boxes that might block paths
            if box_distance > 0:
                total_cost += 0.5
        
        return total_cost
    
    def estimate_distance(self, start, end):
        """Estimate distance between two locations using BFS."""
        if start == end:
            return 0
            
        # Try Manhattan distance first if coordinates are available
        if start in self.loc_coords and end in self.loc_coords:
            x1, y1 = self.loc_coords[start]
            x2, y2 = self.loc_coords[end]
            return abs(x1 - x2) + abs(y1 - y2)
        
        # Fallback to BFS if no coordinates or for more accurate distance
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == end:
                return dist
                
            if current in visited:
                continue
                
            visited.add(current)
            
            for neighbor in self.adjacency.get(current, []):
                queue.append((neighbor, dist + 1))
        
        # If no path found (shouldn't happen in valid states)
        return float('inf')
