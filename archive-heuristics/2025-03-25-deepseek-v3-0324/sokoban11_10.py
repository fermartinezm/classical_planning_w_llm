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
    3. Adding penalties for boxes that are not in goal positions but are blocking paths

    # Assumptions:
    - Each box has exactly one goal position (no multiple goals for the same box)
    - The grid is rectangular and coordinates follow the pattern loc_X_Y
    - Only one robot exists in the puzzle
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals
    - Build an adjacency graph from the static 'adjacent' facts
    - Precompute shortest paths between all locations using Floyd-Warshall algorithm

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not in its goal position:
        a) Calculate the shortest path distance from its current position to its goal
        b) Calculate the shortest path distance from the robot to the box
        c) Add these distances to the total heuristic value
    2. For boxes already in goal positions, no cost is added
    3. Add a small penalty for each box that's not in a goal position to encourage moving boxes
    4. The final heuristic is the sum of all these components
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the location graph."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal positions for boxes
        self.box_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, box, loc = get_parts(goal)
                self.box_goals[box] = loc
        
        # Build adjacency graph and collect all locations
        self.locations = set()
        self.adjacency = defaultdict(dict)
        
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.adjacency[loc1][loc2] = 1
                self.adjacency[loc2][loc1] = 1
        
        # Precompute all pairs shortest paths using Floyd-Warshall
        self.distances = defaultdict(dict)
        for loc1 in self.locations:
            for loc2 in self.locations:
                if loc1 == loc2:
                    self.distances[loc1][loc2] = 0
                elif loc2 in self.adjacency[loc1]:
                    self.distances[loc1][loc2] = 1
                else:
                    self.distances[loc1][loc2] = float('inf')
        
        for k in self.locations:
            for i in self.locations:
                for j in self.locations:
                    if self.distances[i][j] > self.distances[i][k] + self.distances[k][j]:
                        self.distances[i][j] = self.distances[i][k] + self.distances[k][j]

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        
        # Find current robot position
        robot_pos = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_pos = get_parts(fact)[1]
                break
        
        if not robot_pos:
            return float('inf')
        
        # Find current box positions
        box_positions = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_positions[box] = loc
        
        total_cost = 0
        
        for box, goal_loc in self.box_goals.items():
            current_loc = box_positions.get(box, None)
            
            if not current_loc:
                return float('inf')
            
            if current_loc == goal_loc:
                continue  # Box is already at goal
                
            # Cost to move robot to box
            robot_to_box = self.distances[robot_pos][current_loc]
            
            # Cost to push box to goal (approximate)
            box_to_goal = self.distances[current_loc][goal_loc]
            
            # Add to total cost
            total_cost += robot_to_box + box_to_goal
            
            # Small penalty for each box not at goal to encourage progress
            total_cost += 1
        
        return total_cost
