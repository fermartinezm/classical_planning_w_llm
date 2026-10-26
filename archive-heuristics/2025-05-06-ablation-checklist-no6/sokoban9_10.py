from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available from a library path like 'heuristics.heuristic_base'
# from heuristics.heuristic_base import Heuristic

# Note: The Heuristic base class is assumed to be provided by the environment.
# If running this code standalone, you might need a dummy definition:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to compare parts and args up to the length of the shorter sequence.
    # fnmatch handles '*' wildcards.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost by summing the shortest path distances for each box
    from its current location to its goal location. This heuristic ignores
    the robot's position and potential blockages by other boxes or walls,
    making it non-admissible but efficiently computable.

    The heuristic value is the sum of the minimum number of pushes required
    for each box independently to reach its goal location, assuming no
    obstacles other than the grid structure itself.

    # Heuristic Initialization
    - Builds a graph representation of the grid from adjacent facts.
    - Stores goal locations for each box.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the current location of each box that needs to reach a goal.
    2. For each such box:
       a. If the box is already at its goal location, its contribution is 0.
       b. If the box is not at its goal, compute the shortest path distance
          between the box's current location and its goal location using BFS
          on the grid graph (ignoring current state's clear/occupied status).
          This distance represents the minimum number of pushes needed for this box.
       c. If no path exists, the state is likely unsolvable; return infinity.
    3. Sum the distances computed for all boxes.
    4. The total sum is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - Grid structure from adjacent facts.
        """
        # self.goals = task.goals # Goal facts are processed to get goal_locations
        static_facts = task.static

        # Build the grid graph from adjacent facts.
        # The graph is represented as an adjacency dictionary: {location: [neighbor1, neighbor2, ...]}
        self.grid_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Check if it's an 'adjacent' predicate with 4 parts (predicate, loc1, loc2, dir)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Add bidirectional edges for distance calculation, ignoring direction
                if loc1 not in self.grid_graph:
                    self.grid_graph[loc1] = []
                if loc2 not in self.grid_graph:
                    self.grid_graph[loc2] = []
                # Avoid adding duplicate neighbors if adjacent facts are listed in both directions
                if loc2 not in self.grid_graph[loc1]:
                    self.grid_graph[loc1].append(loc2)
                if loc1 not in self.grid_graph[loc2]:
                    self.grid_graph[loc2].append(loc1) # Assuming adjacency is symmetric for distance

        # Store goal locations for each box.
        # self.goal_locations is a dictionary: {box_name: goal_location_name}
        self.goal_locations = {}
        # task.goals is a frozenset of goal facts, e.g., {'(at box1 loc_2_4)', '(at box2 loc_3_5)'}
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            # Check if it's an 'at' predicate with 3 parts (predicate, obj, loc)
            # We assume goal facts are primarily about box locations.
            if parts[0] == "at" and len(parts) == 3:
                 box, location = parts[1], parts[2]
                 self.goal_locations[box] = location


    def bfs(self, start_loc, end_loc):
        """
        Compute the shortest path distance between two locations using BFS on the grid graph.
        Returns float('inf') if no path exists.
        """
        # If start and end are the same, distance is 0.
        if start_loc == end_loc:
            return 0
        # If either location is not in the graph, they are disconnected or invalid.
        if start_loc not in self.grid_graph or end_loc not in self.grid_graph:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, current_dist = queue.popleft()

            # Found the target location
            if current_loc == end_loc:
                return current_dist

            # Explore neighbors
            # Check if current_loc has neighbors in the graph (should be true if it's in grid_graph keys)
            if current_loc in self.grid_graph:
                for neighbor_loc in self.grid_graph[current_loc]:
                    if neighbor_loc not in visited:
                        visited.add(neighbor_loc)
                        queue.append((neighbor_loc, current_dist + 1))

        # If the queue is empty and the end_loc was not reached, no path exists.
        return float('inf')

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        This is the sum of shortest path distances for each goal box
        from its current location to its goal location.
        """
        state = node.state

        # Find current location of each box that is a goal object.
        # box_locations is a dictionary: {box_name: current_location_name}
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if it's an 'at' predicate with 3 parts (predicate, obj, loc)
            # and if the object is one of the boxes we care about (i.e., listed in goal_locations)
            if parts[0] == "at" and len(parts) == 3 and parts[1] in self.goal_locations:
                 box, location = parts[1], parts[2]
                 box_locations[box] = location

        total_heuristic = 0

        # Sum shortest path distances for each box to its goal.
        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box)

            # If a box specified in the goal is not found in the current state,
            # this state is likely invalid or unreachable. Return infinity.
            if current_location is None:
                 return float('inf')

            # If the box is not yet at its goal location
            if current_location != goal_location:
                dist = self.bfs(current_location, goal_location)
                if dist == float('inf'):
                    # If a box cannot reach its goal location from its current position
                    # on the grid graph, the state is unsolvable. Return infinity.
                    return float('inf')
                total_heuristic += dist

        # The heuristic is 0 if and only if all boxes in self.goal_locations
        # are at their respective goal_location (distance 0 for all).
        # If any box is not at its goal, its distance is >= 1, so total_heuristic > 0.
        # The heuristic is finite for solvable states where all goal boxes are reachable.
        # All used modules (deque, fnmatch) are imported. Heuristic base class is assumed.
        # Static info (grid_graph, goal_locations) is extracted in __init__.
        # Parsing logic handles brackets using get_parts.

        return total_heuristic
