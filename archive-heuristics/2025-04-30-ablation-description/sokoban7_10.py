from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    # Simple check for matching number of parts if no wildcards are used in args
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS distance function for graph traversal
def bfs_distance(graph, start, end):
    """
    Computes the shortest path distance between two locations in a graph using BFS.
    Returns infinity if the end is unreachable from the start.
    """
    if start == end:
        return 0
    queue = deque([(start, 0)]) # Queue stores tuples of (location, distance)
    visited = {start} # Set to keep track of visited locations
    while queue:
        current_loc, dist = queue.popleft()
        if current_loc == end:
            return dist
        # Check if current_loc is a valid key in the graph and has neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # Return infinity if goal is unreachable from start

# Define a dummy Heuristic base class if not provided externally
# This is just for ensuring the class structure matches the expectation
# In a real environment, this would be imported from heuristics.heuristic_base.
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass

class sokobanHeuristic(Heuristic): # Assuming Heuristic is imported or defined
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    shortest path distances between each box's current location and its goal location.
    The distance is calculated based on the adjacency graph defined in the PDDL domain.

    # Assumptions
    - The grid structure is implicitly defined by the `adjacent` predicates.
    - The shortest path distance between locations is a reasonable lower bound
      (though not strictly admissible in terms of actions, it's a distance lower bound).
    - The heuristic ignores the robot's position and the cost/constraints of pushing,
      treating box movement as simple traversal for distance calculation.
    - All goal locations are reachable by boxes through the defined adjacency.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an undirected graph (adjacency list) of locations based on the
      `adjacent` predicates from the static facts. This graph is used to compute
      shortest path distances between any two locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box from the `task.goals` (done in __init__).
    2. Build the location adjacency graph from `task.static` (done in __init__).
    3. For a given state (`node.state`):
       - Find the current location of each box by iterating through the state facts
         and finding predicates of the form `(at ?box ?location)`.
       - Initialize a total heuristic cost to 0.
       - For each box that has a goal location defined in the goals:
         - Get the box's current location from the state.
         - Get the box's goal location from the pre-calculated goal mapping.
         - If the box's current location is not its goal location:
           - Calculate the shortest path distance between the current box location
             and its goal location using BFS on the pre-built graph.
           - If the goal is unreachable (distance is infinity), return infinity
             immediately as this state is likely a dead end.
           - Add this finite distance to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the location graph."""
        # Assuming task.goals is a frozenset of goal facts
        self.goals = task.goals
        # Assuming task.static is a frozenset of static facts
        static_facts = task.static

        # 1. Extract goal locations for each box from the goal facts
        self.box_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at ?box ?location)
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                 box, location = parts[1], parts[2]
                 self.box_goals[box] = location

        # 2. Build the location adjacency graph from static facts
        self.graph = {}
        for fact in static_facts:
            # Adjacent facts are (adjacent ?l1 ?l2 ?dir)
            parts = get_parts(fact)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Add adjacency in both directions (graph is undirected for distance)
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                self.graph[loc1].add(loc2)
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc2].add(loc1)

    def __call__(self, node):
        """Estimate the minimum cost to move all boxes to their goal locations."""
        state = node.state  # Current world state (frozenset of facts).

        # Find current locations of all boxes that are relevant (i.e., have a goal)
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Look for (at ?obj ?location) facts
            if parts[0] == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                # Check if the object is one of the boxes we care about (i.e., in our goal list)
                if obj in self.box_goals:
                    current_box_locations[obj] = location

        total_distance = 0  # Initialize heuristic cost

        # Calculate sum of distances for each box to its goal
        for box, goal_location in self.box_goals.items():
            # Ensure the box exists in the current state's 'at' facts
            if box in current_box_locations:
                current_location = current_box_locations[box]
                # Only add distance if the box is not already at its goal
                if current_location != goal_location:
                    # Calculate shortest path distance using BFS on the graph
                    distance = bfs_distance(self.graph, current_location, goal_location)
                    # If distance is infinity, it means the goal is unreachable from
                    # the current box location in the graph. This state is likely
                    # a dead end or part of an unsolvable path. A large heuristic
                    # value is appropriate to prune this branch.
                    if distance == float('inf'):
                         return float('inf') # Return a very large value
                    total_distance += distance
            # Note: If a box listed in the goals is NOT found in the current state's
            # 'at' facts, this heuristic implicitly assumes it's not yet placed
            # or the state is malformed. Assuming valid states from the planner,
            # all goal boxes should have an 'at' fact.

        # The heuristic is the sum of the shortest path distances for each box.
        # If all boxes are at their goals, total_distance will be 0.
        return total_distance
