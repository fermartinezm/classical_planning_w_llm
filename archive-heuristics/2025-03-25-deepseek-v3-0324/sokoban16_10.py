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
    1. Calculating the Manhattan distance from each box to its goal position
    2. Calculating the distance from the robot to each box
    3. Adding these distances with appropriate weights to account for pushing costs

    # Assumptions:
    - Each box has exactly one goal position (standard Sokoban)
    - The grid is rectangular and coordinates can be extracted from location names
    - Pushing a box requires the robot to be adjacent to it
    - Moving the robot without pushing has cost 1
    - Pushing a box has cost 1 (same as moving)

    # Heuristic Initialization
    - Extract goal positions for boxes from task goals
    - Build an adjacency graph from static facts to compute distances
    - Parse location coordinates from their names (format loc_X_Y)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Calculate Manhattan distance from current position to goal
        b) Find the robot's distance to the box (shortest path)
        c) Add these distances with appropriate weights
    2. Sum these values for all boxes
    3. Add a small penalty for boxes not being on goal positions to break ties
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building adjacency graph."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal positions for boxes
        self.box_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, box, loc = get_parts(goal)
                self.box_goals[box] = loc
        
        # Build adjacency graph
        self.adjacency = defaultdict(set)
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.adjacency[loc1].add(loc2)
                self.adjacency[loc2].add(loc1)
        
        # Precompute all pairs shortest paths (Floyd-Warshall would be better for large maps)
        self.distance_cache = {}
    
    def _parse_coords(self, loc):
        """Extract coordinates from location name (format loc_X_Y)."""
        parts = loc.split('_')
        return int(parts[1]), int(parts[2])
    
    def _manhattan_distance(self, loc1, loc2):
        """Calculate Manhattan distance between two locations."""
        x1, y1 = self._parse_coords(loc1)
        x2, y2 = self._parse_coords(loc2)
        return abs(x1 - x2) + abs(y1 - y2)
    
    def _shortest_path(self, start, end):
        """BFS to find shortest path between two locations."""
        if start == end:
            return 0
        
        # Check cache first
        if (start, end) in self.distance_cache:
            return self.distance_cache[(start, end)]
        
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == end:
                self.distance_cache[(start, end)] = dist
                return dist
            
            if current in visited:
                continue
                
            visited.add(current)
            
            for neighbor in self.adjacency[current]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        
        # If no path found (shouldn't happen in valid Sokoban problems)
        return float('inf')
    
    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state
        
        # Find current positions of robot and boxes
        robot_pos = None
        box_positions = {}
        
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
                
            if parts[0] == "at-robot":
                robot_pos = parts[1]
            elif parts[0] == "at" and parts[1].startswith("box"):
                box_positions[parts[1]] = parts[2]
        
        if not robot_pos:
            return float('inf')  # Invalid state
        
        total_cost = 0
        
        for box, goal_loc in self.box_goals.items():
            if box not in box_positions:
                return float('inf')  # Box missing
                
            current_loc = box_positions[box]
            
            if current_loc == goal_loc:
                continue  # Box already at goal
                
            # Distance from box to goal (Manhattan as approximation)
            box_to_goal = self._manhattan_distance(current_loc, goal_loc)
            
            # Distance from robot to box
            robot_to_box = self._shortest_path(robot_pos, current_loc)
            
            # Add to total cost (each push counts as 1 action)
            total_cost += robot_to_box + box_to_goal
        
        # Add small penalty for each box not at goal to break ties
        boxes_not_at_goal = sum(1 for box in self.box_goals 
                               if box_positions.get(box) != self.box_goals[box])
        total_cost += boxes_not_at_goal * 0.1
        
        return total_cost
