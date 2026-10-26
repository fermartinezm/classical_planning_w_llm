from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to handle potentially different lengths, fnmatch handles wildcards
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Inherit from Heuristic if it's provided in the environment
# class sokobanHeuristic(Heuristic):
class sokobanHeuristic:
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost by summing the shortest path distances
    from each box's current location to its goal location. It represents a
    lower bound on the number of push actions required to move the boxes
    to their targets, ignoring robot movement costs and obstacles.

    # Assumptions
    - The goal is to move specific boxes to specific target locations, as
      specified by `(at ?box ?location)` predicates in the goal state.
    - The connectivity of locations is defined solely by the `adjacent` facts,
      forming an undirected graph.
    - The shortest path distance between two locations in this graph is a
      reasonable estimate of the minimum number of push actions needed to
      move a box between them, assuming ideal conditions (robot is always
      in position, path is clear).

    # Heuristic Initialization
    - The goal locations for each box are extracted from the task's goal
      conditions.
    - A graph representing the grid of locations and their adjacencies is
      built from the static `adjacent` facts.
    - Shortest path distances between all pairs of locations in this graph
      are pre-computed using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Access the current state of the planning node.
    2. Check if the current state is the goal state. If it is, the heuristic
       value is 0.
    3. If not the goal state, find the current location of each box by
       iterating through the state facts and looking for `(at ?box ?location)`
       predicates.
    4. Initialize the total heuristic cost to 0.
    5. For each box that has a specified goal location:
       - Retrieve the box's current location and its goal location.
       - If the box is not currently at its goal location:
         - Look up the pre-computed shortest path distance between the box's
           current location and its goal location using the distance map
           generated during initialization.
         - If a path exists (distance is finite), add this distance to the
           total heuristic cost.
         - If no path exists (distance is infinity), the goal is unreachable
           for this box from its current location in this grid configuration;
           return infinity as the heuristic value for this state.
       - If a box expected in the goal is not found in the current state,
         return infinity (unsolvable state).
    6. Return the accumulated total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the location graph."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                box, location = args
                self.goal_locations[box] = location
            # Ignore other potential goal predicates if any

        # Build the location graph from adjacent facts
        self.location_graph = {}
        all_locations = set()
        for fact in static_facts:
             if match(fact, "adjacent", "*", "*", "*"):
                 _, loc1, loc2, _ = get_parts(fact)
                 all_locations.add(loc1)
                 all_locations.add(loc2)

        for loc in all_locations:
            self.location_graph[loc] = set()

        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                # Add bidirectional edges
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1)

        # Pre-compute all-pairs shortest paths using BFS
        self.distances = {}
        # Only compute distances for locations that are part of the graph
        for start_node in self.location_graph:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """Perform BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.location_graph}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # current_node is guaranteed to be in self.location_graph keys
            # because we iterate over self.location_graph.keys() initially.
            for neighbor in self.location_graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state  # Current world state.

        # Check if the state is a goal state. If yes, heuristic is 0.
        # This assumes the goal only involves (at box loc) predicates.
        # If there were other goal predicates (e.g., robot position), this check would need adjustment.
        # Based on the examples, this seems sufficient.
        if self.goals <= state:
             return 0

        # Find current locations of all objects (robot and boxes)
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2: # (at box loc)
                 obj, loc = args
                 current_locations[obj] = loc
            # We don't need robot location for this simple heuristic

        total_heuristic = 0

        # Sum distances for each misplaced box
        for box, goal_location in self.goal_locations.items():
            if box in current_locations:
                current_box_location = current_locations[box]

                # If the box is not at its goal
                if current_box_location != goal_location:
                    # Get the shortest distance from current box location to goal location
                    # Handle cases where locations might not be in the graph (e.g., walls, unreachable areas)
                    # A location might be in the state/goal but not in the adjacent facts if it's a wall or isolated.
                    # Our graph only contains locations from adjacent facts.
                    # If a box is at a location not in the graph, or its goal is not in the graph,
                    # or the goal is unreachable from the current location, it's likely unsolvable.
                    if current_box_location in self.distances and goal_location in self.distances[current_box_location]:
                         dist = self.distances[current_box_location][goal_location]
                         # If distance is infinity, it means the goal is unreachable from this location.
                         # This state is likely unsolvable, return infinity.
                         if dist == float('inf'):
                             return float('inf')
                         total_heuristic += dist
                    else:
                         # If either location is not in the graph or unreachable, assume unsolvable.
                         return float('inf')
            else:
                 # If a box that is supposed to be at a goal location is not found in the state,
                 # it's an unsolvable state. Return infinity.
                 return float('inf')

        # If we reached here, the state is not a goal state, but all boxes
        # are accounted for and paths to their goals are finite.
        # The total_heuristic is the sum of distances for misplaced boxes.
        # This value will be > 0 because we already handled the goal state case.
        return total_heuristic
