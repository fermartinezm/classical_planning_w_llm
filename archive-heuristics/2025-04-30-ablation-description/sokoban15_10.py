from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Basic validation for expected format
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Depending on expected robustness, could log a warning or raise an error
         # For this context, assuming valid PDDL fact strings.
         # Returning split might work for some non-standard inputs but is not guaranteed.
         return fact.split()
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the pattern is longer than the fact parts
    if len(args) > len(parts):
        return False
    # Use zip to compare corresponding parts up to the length of args
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the minimum number of push actions required to move
    each box to its goal location, assuming the robot can always reach a position
    to push the box towards the goal. It is calculated as the sum of the shortest
    path distances (in terms of adjacent locations) between each box's current
    location and its goal location.

    # Assumptions
    - The grid/map is represented by `adjacent` predicates.
    - The distance between two locations is the shortest path length in the graph
      defined by the `adjacent` predicates.
    - The heuristic only considers the box-goal distances and ignores the robot's
      position and the cost of moving the robot. This makes it an admissible
      heuristic (it never overestimates the true cost of moving *only* the box).
      For greedy best-first search, admissibility is not strictly required, but
      this provides a simple, informative estimate.
    - All goal conditions related to boxes are of the form `(at ?b ?l)`.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds a graph representation of the locations and their adjacencies
      based on the static `adjacent` facts.
    - Pre-calculates the shortest path distance between all pairs of locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box specified in the goal:
    2. Find the box's current location in the current state by iterating through
       the state facts to find `(at box current_location)`.
    3. Retrieve the box's goal location (stored during initialization).
    4. If the box is already at its goal location, the contribution to the heuristic is 0.
    5. If the box is not at its goal location, find the pre-calculated shortest
       path distance between the box's current location and its goal location
       using the distance map computed from the graph built from `adjacent` predicates.
       If the goal is unreachable from the current location, the distance is infinite.
    6. Sum the distances calculated in step 5 for all boxes.
    7. The total sum is the heuristic value for the current state. If any box-goal
       distance was infinite, the total heuristic is infinite.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for distance calculations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        # task.goals is a frozenset of goal facts (strings)
        for goal_fact_str in self.goals:
            # Assuming goals are individual facts like (at box location)
            # The PDDL parser should have flattened any 'and' structure.
            predicate, *args = get_parts(goal_fact_str)
            if predicate == "at":
                # args should be [box_name, location_name]
                if len(args) == 2:
                    box, location = args
                    self.goal_locations[box] = location
                # else: Handle unexpected goal fact format if necessary


        # Build the graph of locations from adjacent facts.
        # We treat the graph as undirected for distance calculation, assuming
        # if A is adjacent to B, B is adjacent to A (which PDDL usually provides).
        self.graph = {}
        all_locations = set()

        for fact in static_facts:
            # Match (adjacent loc1 loc2 dir)
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                # Add adjacency in both directions
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1)

        # Ensure all locations mentioned in goals are in the graph, even if isolated
        # This handles cases where a goal location might not be adjacent to anything else
        # in the static facts, but is still a valid location in the problem.
        for loc in self.goal_locations.values():
             if loc not in self.graph:
                 self.graph[loc] = set()
                 all_locations.add(loc)

        # Calculate all-pairs shortest paths using BFS from each location.
        self.distances = {}
        for start_node in all_locations:
            self.distances[start_node] = self._bfs(start_node, all_locations)

    def _bfs(self, start_node, all_nodes):
        """
        Performs Breadth-First Search from a start_node to find distances
        to all other reachable nodes in the graph.
        """
        distances = {node: float('inf') for node in all_nodes}
        # Only proceed if the start_node is actually in the graph (has neighbors or is listed)
        if start_node in self.graph:
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                # current_node should be in self.graph if it was added to the queue
                if current_node in self.graph: # Defensive check
                    for neighbor in self.graph[current_node]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_node] + 1
                            queue.append(neighbor)
        # If start_node was not in self.graph initially, all distances remain inf.

        return distances


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required push actions
        based on box-goal distances.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        total_distance = 0  # Initialize heuristic value.

        # Calculate the sum of distances for each box to its goal.
        for box, goal_location in self.goal_locations.items():
            # Find the current location of this specific box in the state.
            current_box_location = None
            # Iterate through the state facts to find the fact (at box current_location)
            for fact_str in state:
                if match(fact_str, "at", box, "*"):
                    # Found the fact, extract the location
                    _, _, current_box_location = get_parts(fact_str)
                    break # Found the box location, move to the next box

            # If the box is not found in the state (should not happen in valid states),
            # or if the goal location is not a known location in our graph,
            # this state might be problematic or unreachable. Return infinity.
            if current_box_location is None or goal_location not in self.distances[current_box_location]:
                 # This indicates an issue with the state or goal definition relative to the graph.
                 # Treat as infinite cost.
                 return float('inf')

            # If the box is already at its goal, distance is 0.
            if current_box_location == goal_location:
                distance = 0
            else:
                # Look up the pre-calculated distance.
                # If goal_location is unreachable from current_box_location, distance will be inf.
                distance = self.distances[current_box_location][goal_location]

            # If any box-goal distance is infinite, the total heuristic is infinite.
            if distance == float('inf'):
                return float('inf')

            total_distance += distance

        # The heuristic is the sum of box-goal distances.
        # It is 0 if and only if all boxes are at their goals.
        return total_distance
