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
    3. Considering whether the robot needs to go around boxes to push them
    4. Summing these distances with appropriate weights

    # Assumptions:
    - Each box has exactly one goal position (no multiple goals for same box)
    - The grid is rectangular and coordinates can be extracted from location names
    - Pushing a box always moves it one step in a cardinal direction
    - The robot can only push one box at a time

    # Heuristic Initialization
    - Extract goal positions for boxes from task.goals
    - Build a graph of adjacent locations from static facts
    - Precompute shortest paths between all locations for distance calculations

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Calculate Manhattan distance to its goal (minimum pushes needed)
        b) Find the robot's distance to the box (minimum moves needed to reach it)
        c) Add these distances with appropriate weights
    2. For boxes already at their goal, no cost is added
    3. The heuristic value is the sum of:
        a) The robot's distance to the nearest box not at goal
        b) The sum of Manhattan distances of all boxes to their goals
        c) An additional penalty for boxes that are not in a straight line pushable position
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building location graph."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal positions for boxes
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
        
        # Precompute shortest paths between all locations
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """Compute shortest paths between all locations using BFS."""
        distances = defaultdict(dict)
        for loc in self.adjacency:
            visited = {loc: 0}
            queue = [loc]
            while queue:
                current = queue.pop(0)
                for neighbor in self.adjacency[current]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for other_loc, dist in visited.items():
                distances[loc][other_loc] = dist
        return distances

    def _get_coords(self, location):
        """Extract coordinates from location name (e.g., 'loc_2_3' -> (2,3))"""
        parts = location.split('_')
        return (int(parts[1]), int(parts[2]))

    def _manhattan_distance(self, loc1, loc2):
        """Calculate Manhattan distance between two locations"""
        x1, y1 = self._get_coords(loc1)
        x2, y2 = self._get_coords(loc2)
        return abs(x1 - x2) + abs(y1 - y2)

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
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
        
        if not robot_pos:
            return float('inf')  # Invalid state
        
        total_cost = 0
        min_robot_to_box = float('inf')
        
        for box, current_loc in box_positions.items():
            goal_loc = self.box_goals.get(box)
            if not goal_loc:
                continue  # Box has no goal (shouldn't happen in valid tasks)
            
            if current_loc == goal_loc:
                continue  # Box is already at goal
                
            # Distance from box to its goal (minimum pushes needed)
            box_to_goal = self._manhattan_distance(current_loc, goal_loc)
            
            # Distance from robot to this box
            robot_to_box = self.distances.get(robot_pos, {}).get(current_loc, float('inf'))
            
            # Update minimum robot-to-box distance
            if robot_to_box < min_robot_to_box:
                min_robot_to_box = robot_to_box
            
            # Add cost for this box
            total_cost += box_to_goal * 2  # Pushes are more expensive than moves
            total_cost += robot_to_box  # Add robot's distance to reach the box
            
            # Additional penalty if box is not in a straight line pushable position
            if not self._is_pushable_position(current_loc, state):
                total_cost += 2
        
        # If all boxes are at goals, return 0
        if total_cost == 0:
            return 0
            
        # Add base cost for robot to reach nearest box
        if min_robot_to_box != float('inf'):
            total_cost += min_robot_to_box
        
        return total_cost

    def _is_pushable_position(self, box_loc, state):
        """Check if box is in a position where it can be pushed in at least one direction"""
        # Get all adjacent locations
        adjacent_locs = self.adjacency.get(box_loc, set())
        
        for neighbor in adjacent_locs:
            # Check if neighbor is clear and opposite side is clear or out of bounds
            if f"(clear {neighbor})" in state:
                # Check if there's space behind the box to push it
                # This is a simplified check - a more thorough one would consider directions
                return True
        return False
