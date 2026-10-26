from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

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
    3. Summing these distances with appropriate weights

    # Assumptions:
    - Each box has exactly one goal position (no multiple goals per box)
    - The grid is connected (all locations are reachable)
    - Pushing a box always moves it closer to its goal (no deadlock detection)

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals
    - Build a graph representation of the grid from static adjacency facts
    - Precompute shortest paths between all locations using BFS

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Calculate the Manhattan distance from its current position to its goal
        b) Add this distance to the total (weighted by 2 to account for push+move)
    2. Find the robot's position
    3. For each box not at its goal:
        a) Calculate the shortest path distance from the robot to the box
        b) Add this distance to the total
    4. Return the sum of all distances as the heuristic value
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal positions for boxes
        self.box_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, box, loc = get_parts(goal)
                self.box_goals[box] = loc
        
        # Build adjacency graph from static facts
        self.graph = defaultdict(list)
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)  # undirected graph
        
        # Precompute all pairs shortest paths using BFS
        self.distances = {}
        locations = set(self.graph.keys())
        for source in locations:
            self.distances[source] = self._bfs(source)
    
    def _bfs(self, start):
        """Compute shortest paths from start location to all others using BFS."""
        distances = {start: 0}
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            for neighbor in self.graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        
        # Extract current positions of boxes and robot
        box_positions = {}
        robot_pos = None
        
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "at":
                _, box, loc = parts
                box_positions[box] = loc
            elif len(parts) == 2 and parts[0] == "at-robot":
                _, loc = parts
                robot_pos = loc
        
        # If all boxes are at their goals, return 0
        all_goals_reached = True
        for box, goal_loc in self.box_goals.items():
            if box_positions.get(box) != goal_loc:
                all_goals_reached = False
                break
        if all_goals_reached:
            return 0
        
        total_cost = 0
        
        # For each box not at its goal
        for box, goal_loc in self.box_goals.items():
            current_loc = box_positions.get(box)
            if current_loc != goal_loc:
                # Add distance from box to goal (weighted by 2 for push+move)
                try:
                    box_to_goal = self.distances[current_loc][goal_loc]
                    total_cost += 2 * box_to_goal
                except KeyError:
                    # If path doesn't exist (shouldn't happen in valid states)
                    return float('inf')
        
        # Add distance from robot to nearest box not at goal
        if robot_pos:
            min_robot_to_box = float('inf')
            for box, current_loc in box_positions.items():
                if current_loc != self.box_goals[box]:
                    try:
                        dist = self.distances[robot_pos][current_loc]
                        if dist < min_robot_to_box:
                            min_robot_to_box = dist
                    except KeyError:
                        pass
            
            if min_robot_to_box != float('inf'):
                total_cost += min_robot_to_box
        
        return total_cost
