# Assuming Heuristic base class is available in heuristics.heuristic_base
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this import path


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a meaningful match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    shortest path distances for each box from its current location to its
    goal location. The distance is calculated on the graph defined by the
    'adjacent' predicates.

    # Assumptions
    - The primary cost driver is moving boxes to their goal locations.
    - The shortest path distance on the adjacency graph is a reasonable
      estimate of the minimum number of 'push' actions required for a box
      to reach its goal, ignoring robot movement and clear space constraints.
    - The heuristic is not admissible; it does not guarantee finding the
      optimal solution but aims to guide the search efficiently in a greedy
      best-first search.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an undirected graph representing the connectivity between locations
      based on the 'adjacent' predicates in the static facts. This graph is
      used for shortest path calculations (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic value to 0.
    2. Identify the current location of each box from the current state.
    3. For each box:
       a. Determine its goal location (pre-calculated during initialization).
       b. If the box is already at its goal location, its contribution is 0.
       c. If the box is not at its goal location, calculate the shortest path
          distance between the box's current location and its goal location
          using Breadth-First Search (BFS) on the adjacency graph.
       d. Add this calculated distance to the total heuristic value.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the adjacency graph from static facts.
        """
        # Call the base class constructor if necessary, depending on its definition
        # super().__init__(task) # If Heuristic requires task in its init

        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at <box> <location>)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                 # Assuming the object in the goal 'at' predicate is always a box
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location

        # Build the adjacency graph from static facts.
        # The graph is undirected.
        self.adjacency_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            # Check for 'adjacent' predicate with 3 arguments (loc1, loc2, direction)
            if parts and parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                # Add edge loc1 -> loc2
                if loc1 not in self.adjacency_graph:
                    self.adjacency_graph[loc1] = []
                self.adjacency_graph[loc1].append(loc2)
                # Add edge loc2 -> loc1 (graph is undirected for distance)
                if loc2 not in self.adjacency_graph:
                    self.adjacency_graph[loc2] = []
                self.adjacency_graph[loc2].append(loc1)

        # Remove duplicates from adjacency lists to ensure correct graph structure
        for loc in self.adjacency_graph:
             self.adjacency_graph[loc] = list(set(self.adjacency_graph[loc]))


    def _bfs_distance(self, start_loc, goal_loc):
        """
        Calculates the shortest path distance between two locations
        using BFS on the adjacency graph. Returns float('inf') if no path exists.
        """
        # If start or goal location is not in the graph, it's unreachable
        # This might happen if the problem defines locations not connected
        # by any 'adjacent' predicates, or if start/goal are malformed.
        if start_loc not in self.adjacency_graph or goal_loc not in self.adjacency_graph:
             return float('inf')

        if start_loc == goal_loc:
            return 0

        queue = deque([(start_loc, 0)]) # (location, distance)
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == goal_loc:
                return dist

            # Get neighbors from the graph. Use .get() with default empty list
            # in case a location is in visited/queue but not in graph keys (shouldn't happen if graph is built correctly)
            neighbors = self.adjacency_graph.get(current_loc, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # Goal not reachable from start_loc
        return float('inf')


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            # Look for facts like (at box1 loc_4_4)
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # Check if the object is one of the boxes we care about (i.e., has a goal)
                 if obj in self.goal_locations:
                     current_box_locations[obj] = loc

        total_cost = 0  # Initialize heuristic cost.

        # Calculate sum of distances for each box
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If a box is expected (has a goal) but not found in the state,
            # this state is likely invalid or unsolvable.
            if current_location is None:
                 return float('inf')

            # If the box is already at the goal, cost is 0 for this box
            if current_location == goal_location:
                continue

            # Calculate distance from current location to goal location
            distance = self._bfs_distance(current_location, goal_location)

            # If goal is unreachable for a box, the state is likely unsolvable
            # or requires complex maneuvers not captured by simple distance.
            # Returning infinity for unreachable goals is standard.
            if distance == float('inf'):
                 return float('inf') # State is likely unsolvable

            total_cost += distance

        return total_cost
