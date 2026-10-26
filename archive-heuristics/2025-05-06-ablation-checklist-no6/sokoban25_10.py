from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Utility functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    Estimates the cost by summing the shortest path distance for each box
    to its goal location and the shortest path distance from the robot
    to the closest misplaced box.

    # Heuristic Initialization
    - Build a graph of locations based on `adjacent` facts for shortest path calculations.
    - Store goal locations for each box.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each box not at its goal:
       - Calculate the shortest path distance from the box's current location
         to its goal location using BFS on the location graph. Add this to the total.
    2. Find the robot's current location.
    3. Find the locations of all boxes that are not at their goals.
    4. If there are misplaced boxes, calculate the shortest path distance from the
       robot's location to each misplaced box location using BFS.
    5. Add the minimum of these robot-to-box distances to the total.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - The location graph from `adjacent` facts.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals
        self.static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the location graph from adjacent facts.
        # graph[location] = [neighbor1, neighbor2, ...]
        self.location_graph = {}
        # Collect all locations first to ensure every location mentioned has an entry
        all_locations = set()
        for fact in self.static_facts:
             if match(fact, "adjacent", "*", "*", "*"):
                 _, loc1, loc2, _ = get_parts(fact)
                 all_locations.add(loc1)
                 all_locations.add(loc2)

        for loc in all_locations:
             self.location_graph[loc] = []

        for fact in self.static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                # Add directed edge
                self.location_graph[loc1].append(loc2)


    def bfs_distance(self, start, end):
        """
        Calculate the shortest path distance between two locations using BFS.
        Returns float('inf') if the end is unreachable from the start.
        """
        if start == end:
            return 0

        # Ensure start and end nodes exist in the graph
        if start not in self.location_graph or end not in self.location_graph:
             return float('inf')

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            # Neighbors are guaranteed to be in the graph keys if graph was built correctly
            # However, the neighbor itself might not have outgoing edges, but it must be a key
            # if it was mentioned in any adjacent fact.
            for neighbor in self.location_graph.get(current_loc, []): # Use .get for safety
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # End is unreachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        robot_location = None
        box_locations = {}
        misplaced_boxes = []
        misplaced_box_locations = []

        # Parse state to find robot and box locations
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at-robot":
                robot_location = args[0]
            elif predicate == "at":
                obj, loc = args
                # Check if the object is one of the boxes we care about (i.e., has a goal)
                if obj in self.goal_locations:
                     box_locations[obj] = loc
                     if loc != self.goal_locations[obj]:
                         misplaced_boxes.append(obj)
                         misplaced_box_locations.append(loc)

        # If there are no misplaced boxes, the goal is reached.
        if not misplaced_boxes:
            return 0

        total_cost = 0

        # Calculate box-goal distances for all misplaced boxes
        for box in misplaced_boxes:
            current_loc = box_locations[box]
            goal_loc = self.goal_locations[box]
            dist = self.bfs_distance(current_loc, goal_loc)
            # If any box cannot reach its goal, the state is likely unsolvable
            if dist == float('inf'):
                 return float('inf')
            total_cost += dist

        # Calculate robot-to-closest-misplaced-box distance
        min_robot_dist = float('inf')
        # Ensure robot_location is a valid location in the graph
        if robot_location in self.location_graph:
            for box_loc in misplaced_box_locations:
                 # We already know box_loc is in the graph if box_goal_dist was finite
                 dist = self.bfs_distance(robot_location, box_loc)
                 min_robot_dist = min(min_robot_dist, dist)

        # If robot cannot reach any misplaced box, the state is likely unsolvable
        if min_robot_dist == float('inf'):
             return float('inf')

        total_cost += min_robot_dist

        # The total cost is the sum of box-goal distances and the minimum robot-box distance.
        # If any component was infinite, total_cost will be infinite.
        return total_cost
