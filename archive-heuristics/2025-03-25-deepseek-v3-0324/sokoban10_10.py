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
    1. Calculating the Manhattan distance from the robot to each box
    2. Calculating the Manhattan distance from each box to its goal position
    3. Adding these distances with appropriate weights
    4. Considering whether boxes are already at goal positions

    # Assumptions:
    - Each box has exactly one goal position (standard Sokoban)
    - The grid is connected (all locations are reachable)
    - Pushing a box always takes one action (regardless of direction)
    - Moving the robot without pushing takes one action

    # Heuristic Initialization
    - Extract goal positions for boxes from task.goals
    - Build an adjacency graph from static facts to enable pathfinding
    - Store all clear locations that can be used for movement

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Find the robot's Manhattan distance to the box
        b) Find the box's Manhattan distance to its goal
        c) Add these distances (with robot distance weighted less)
    2. If multiple boxes exist, sum the distances for all boxes
    3. Add a small penalty for each box not at goal to prioritize pushing
    4. If robot needs to reposition before pushing, add that distance
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
        
        # Build adjacency graph for pathfinding
        self.adjacency = defaultdict(set)
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.adjacency[loc1].add(loc2)
                self.adjacency[loc2].add(loc1)
        
        # Precompute location coordinates for Manhattan distance
        self.loc_coords = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                for loc in get_parts(fact)[1:3]:
                    if loc not in self.loc_coords:
                        try:
                            # Parse coordinates from location names like "loc_1_2"
                            _, x, y = loc.split('_')
                            self.loc_coords[loc] = (int(x), int(y))
                        except:
                            # Fallback for non-standard location names
                            self.loc_coords[loc] = (0, 0)

    def manhattan_distance(self, loc1, loc2):
        """Calculate Manhattan distance between two locations."""
        if loc1 in self.loc_coords and loc2 in self.loc_coords:
            x1, y1 = self.loc_coords[loc1]
            x2, y2 = self.loc_coords[loc2]
            return abs(x1 - x2) + abs(y1 - y2)
        return 0  # Fallback if coordinates aren't available

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state
        
        # Track current positions of robot and boxes
        robot_pos = None
        box_positions = {}
        
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_pos = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_positions[box] = loc
        
        # If no boxes left (shouldn't happen in standard Sokoban)
        if not box_positions:
            return 0
        
        total_cost = 0
        
        for box, current_loc in box_positions.items():
            goal_loc = self.box_goals.get(box)
            
            # Skip if box is already at goal
            if current_loc == goal_loc:
                continue
                
            # Distance from robot to box
            robot_to_box = self.manhattan_distance(robot_pos, current_loc) if robot_pos else 0
            
            # Distance from box to goal
            box_to_goal = self.manhattan_distance(current_loc, goal_loc) if goal_loc else 0
            
            # Weight robot distance less since it might be shared between boxes
            total_cost += 0.7 * robot_to_box + box_to_goal + 1  # +1 for the push action
            
            # Update robot position to behind the box (for next box calculation)
            robot_pos = current_loc
        
        return total_cost
