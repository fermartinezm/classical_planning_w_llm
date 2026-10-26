from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
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

# Assuming heuristics.heuristic_base.Heuristic is available in the environment
# from heuristics.heuristic_base import Heuristic

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the minimum number of moves required to get each box
    to its goal location, calculated as the shortest path distance in the grid
    graph from the box's current location to its goal location. The total
    heuristic value is the sum of these distances for all boxes that are not
    yet at their goal. This heuristic ignores the robot's position and the
    presence of other boxes or obstacles on the path, making it potentially
    inadmissible but fast to compute.

    # Assumptions
    - The problem involves moving boxes to specific goal locations.
    - The grid structure and connectivity are defined by 'adjacent' predicates.
    - The shortest path distance between locations is a reasonable estimate
      of the cost to move a box between those locations (ignoring robot
      constraints and other obstacles).

    # Heuristic Initialization
    - The heuristic extracts the goal location for each box from the task's goals.
    - It builds an undirected graph representing the grid connectivity based on
      the 'adjacent' predicates in the static facts.
    - It pre-computes the shortest path distance from each goal location to
      all other reachable locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each box from the task definition.
    2. Build an adjacency list graph representation of the locations based on
       the 'adjacent' facts provided in the static information. Ensure the graph
       is undirected (add edges in both directions for each 'adjacent' fact).
    3. For each unique goal location identified in step 1, perform a BFS starting
       from that goal location to compute the shortest path distance from every
       reachable location *to* that goal location. Store these distances.
    4. In the `__call__` method, given a state:
       a. Find the current location of each box by examining the '(at ?b ?l)'
          facts in the state.
       b. Initialize a total heuristic cost to 0.
       c. For each box that has a goal location:
          i. Get the box's current location and its goal location.
          ii. If the current location is the same as the goal location, the cost
              for this box is 0.
          iii. If the current location is different from the goal location:
              - Look up the pre-computed shortest path distance from the current
                location to the goal location using the distances calculated in
                step 3.
              - If a path exists (distance is finite), add this distance to the
                total heuristic cost.
              - If no path exists (e.g., box is in an isolated area), the problem
                might be unsolvable. Return a very large value (infinity) to
                discourage exploring this state.
       d. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal is (at box_name location_name)
                if len(args) == 2:
                    box, location = args
                    self.goal_locations[box] = location

        # Build the location graph from adjacent facts.
        self.graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent":
                # Assuming fact is (adjacent loc1 loc2 dir)
                if len(parts) == 4:
                    loc1, loc2, direction = parts[1], parts[2], parts[3]
                    # Add undirected edges
                    self.graph.setdefault(loc1, set()).add(loc2)
                    self.graph.setdefault(loc2, set()).add(loc1)

        # Pre-compute distances from each goal location to all other locations.
        self.distances = {}
        unique_goal_locations = set(self.goal_locations.values())
        for goal_loc in unique_goal_locations:
            self.distances[goal_loc] = self._bfs(goal_loc)

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find shortest distances
        to all reachable nodes in the graph. Returns a dictionary mapping
        node -> distance.
        """
        # Initialize distances for all nodes found in the graph
        distances = {node: float('inf') for node in self.graph}

        # If the start_node isn't even in the graph, it's unreachable from anywhere
        # within the defined adjacency structure.
        if start_node not in self.graph:
             # This might indicate an issue with the PDDL instance or domain,
             # or an unreachable goal location. Distances remain infinity.
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is still a valid key in graph (should be if it came from queue)
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    # Ensure neighbor is also a valid key in distances (should be if it's in graph)
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        based on the sum of box-to-goal distances.
        """
        state = node.state  # Current world state.

        # Find current locations of all boxes that have a goal.
        current_box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at":
                 # Assuming fact is (at obj loc)
                 if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    if obj in self.goal_locations: # Only track boxes we care about
                        current_box_locations[obj] = loc

        total_cost = 0  # Initialize heuristic cost.

        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)

            # If a box is not found in the state (shouldn't happen in valid states),
            # or if its current location is the goal, cost is 0 for this box.
            if current_loc is None or current_loc == goal_loc:
                continue

            # Look up the pre-computed distance from current_loc to goal_loc
            # Note: BFS was run from goal_loc, so distances[goal_loc][current_loc]
            # gives the distance from current_loc to goal_loc.
            if goal_loc in self.distances and current_loc in self.distances[goal_loc]:
                 dist = self.distances[goal_loc][current_loc]
                 if dist == float('inf'):
                     # Box is in a part of the graph not reachable from its goal
                     # This state is likely unsolvable. Return infinity.
                     return float('inf')
                 total_cost += dist
            else:
                 # This case might happen if current_loc or goal_loc were not in the graph
                 # built from adjacent facts. This indicates a potential issue with
                 # graph building or an unsolvable state. Treat as unreachable.
                 return float('inf')


        return total_cost
