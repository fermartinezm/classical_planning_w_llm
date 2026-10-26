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
    3. Adding penalties for boxes that are not in goal positions but are blocking other boxes
    4. Considering the need to move around boxes to reach pushing positions

    # Assumptions:
    - Each box has exactly one goal position (standard Sokoban)
    - The grid is rectangular and coordinates can be extracted from location names
    - Pushing a box always moves it one unit in the specified direction
    - The robot can only push one box at a time

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals
    - Build a graph representation of the grid from adjacency facts
    - Precompute shortest paths between all locations for distance calculations

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Calculate Manhattan distance to its goal position
        b) Add the distance from robot to the box (for pushing)
    2. For boxes already at goals, ignore them in calculations
    3. Add a penalty for boxes that are in positions that block other boxes' paths
    4. Consider that the robot might need to move around boxes to reach pushing positions
    5. Sum all these costs to get the total heuristic estimate
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal locations for boxes
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.box_goals[box] = location
        
        # Build adjacency graph
        self.graph = defaultdict(set)
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1)
        
        # Precompute all pairs shortest paths using BFS
        self.distances = {}
        for source in self.graph:
            visited = {source: 0}
            queue = [source]
            while queue:
                current = queue.pop(0)
                for neighbor in self.graph[current]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            self.distances[source] = visited

    def get_location_coords(self, loc):
        """Extract coordinates from location name (e.g., 'loc_3_5' -> (3,5))"""
        parts = loc.split('_')
        return (int(parts[1]), int(parts[2])

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        
        # Track current positions of robot and boxes
        robot_pos = None
        box_positions = {}
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_pos = parts[1]
            elif parts[0] == "at" and parts[1].startswith("box"):
                box_positions[parts[1]] = parts[2]
        
        if not robot_pos:
            return float('inf')  # Invalid state
        
        total_cost = 0
        
        # For each box not at its goal, add distance to goal plus distance from robot to box
        for box, current_pos in box_positions.items():
            if box in self.box_goals and current_pos != self.box_goals[box]:
                goal_pos = self.box_goals[box]
                
                # Distance from box to goal (Manhattan distance)
                x1, y1 = self.get_location_coords(current_pos)
                x2, y2 = self.get_location_coords(goal_pos)
                box_to_goal = abs(x1 - x2) + abs(y1 - y2)
                
                # Distance from robot to box
                if current_pos in self.distances.get(robot_pos, {}):
                    robot_to_box = self.distances[robot_pos][current_pos]
                else:
                    robot_to_box = 10  # Large penalty if no path exists
                
                total_cost += box_to_goal + robot_to_box
        
        # Add penalty for boxes that are blocking paths
        for box, pos in box_positions.items():
            if box in self.box_goals and pos == self.box_goals[box]:
                continue  # Box is at goal, no penalty
                
            # Check if this box is adjacent to another box that needs to move
            for neighbor in self.graph.get(pos, []):
                if neighbor in box_positions.values():
                    total_cost += 2  # Additional cost for moving blocking boxes
        
        return total_cost
