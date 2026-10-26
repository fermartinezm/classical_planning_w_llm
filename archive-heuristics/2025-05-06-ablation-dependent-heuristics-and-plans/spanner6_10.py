import sys
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available

# Helper functions to parse PDDL facts
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to compute shortest path distances
def bfs(graph, start_node):
    """
    Computes shortest path distances from a start_node to all other nodes in a graph.
    Graph is represented as an adjacency dictionary {node: [neighbor1, neighbor2, ...]}
    Returns a dictionary {node: distance}.
    """
    distances = {node: float('inf') for node in graph}
    if start_node in graph:
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It sums the estimated costs for:
    1. Tightening each untightened goal nut.
    2. Picking up usable spanners if the man doesn't carry enough.
    3. Walking to the locations of untightened nuts and necessary spanners.

    # Assumptions:
    - Links between locations are bidirectional.
    - The problem is solvable (i.e., there are enough usable spanners available
      somewhere to tighten all goal nuts, and the graph is connected).
    - There is exactly one man object in the domain.

    # Heuristic Initialization
    - Identifies all locations and builds a graph based on `link` facts.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Identifies the set of goal nuts from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal nuts that are currently `loose` (untightened goal nuts). Let K be this count.
    2. If K is 0, the state is a goal state, return 0.
    3. Identify the man object and his current location.
    4. Count the number of `usable` spanners the man is currently `carrying`. Let C be this count.
    5. Determine how many additional usable spanners the man needs to pick up from the ground: `pickups_needed = max(0, K - C)`.
    6. Identify the locations of all `usable` spanners that are currently `at` a location (on the ground).
    7. The base cost includes the minimum number of `tighten_nut` actions (K) and the minimum number of `pickup_spanner` actions (`pickups_needed`). `base_cost = K + pickups_needed`.
    8. Calculate the walk cost. The man needs to visit the location of each untightened goal nut, and the locations of the `pickups_needed` closest usable spanners on the ground (relative to the man's current location).
       - Identify the set of locations of untightened goal nuts.
       - If `pickups_needed > 0`, find the `pickups_needed` usable spanners on the ground that are closest to the man's current location, and add their locations to the set of required locations.
       - The walk cost is estimated as the sum of the shortest path distances from the man's current location to *each* location in the set of required locations. This is an overestimate but provides a simple, computable non-admissible estimate of travel cost.
    9. The total heuristic value is `base_cost + walk_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal nuts.
        - Location graph and distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Use initial state to find all objects

        # Identify all locations and build the graph
        self.locations = set()
        self.graph = {} # Adjacency list {location: [neighbor1, neighbor2, ...]}

        # Collect locations from link facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Assuming links are bidirectional

        # Collect locations from initial 'at' facts to ensure all relevant locations are included
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 # The second argument of 'at' is a location
                 self.locations.add(parts[2])

        # Ensure graph includes all identified locations, even if they have no links (isolated)
        for loc in self.locations:
             self.graph.setdefault(loc, [])

        # Compute all-pairs shortest paths
        self.dist_matrix = {}
        for start_loc in self.locations:
            self.dist_matrix[start_loc] = bfs(self.graph, start_loc)

        # Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify untightened goal nuts (K) and their locations
        untightened_goal_nuts = set()
        nut_locations = {} # {nut_name: location} - all nuts mentioned in state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "loose" and parts[1] in self.goal_nuts:
                 untightened_goal_nuts.add(parts[1])
            # Find location for all nuts that are goals (they must be locatable)
            if parts[0] == "at" and len(parts) == 3 and parts[1] in self.goal_nuts:
                 nut_locations[parts[1]] = parts[2]

        # Filter nut_locations to only include untightened ones
        untightened_nut_locs = {nut: loc for nut, loc in nut_locations.items() if nut in untightened_goal_nuts}

        K = len(untightened_goal_nuts)

        # 2. If all goal nuts are tightened, heuristic is 0
        if K == 0:
            return 0

        # 3. Identify the man object and his current location
        man_name = None
        # Find man from carrying predicate first (most reliable type hint)
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "carrying" and len(parts) == 3:
                  man_name = parts[1]
                  break

        # If not found via carrying, find man from at predicate
        if man_name is None:
             # Collect all known spanners and nuts from the state
             spanner_names_in_state = {get_parts(f)[1] for f in state if match(f, "usable", "*") or match(f, "carrying", "*", "*")}
             nut_names_in_state = set(nut_locations.keys())

             for fact in state:
                  parts = get_parts(fact)
                  if parts[0] == "at" and len(parts) == 3:
                       obj_at_loc = parts[1]
                       # Assume the object at a location that is not a known spanner or nut is the man
                       if obj_at_loc not in spanner_names_in_state and obj_at_loc not in nut_names_in_state:
                            man_name = obj_at_loc
                            break

        if man_name is None:
             # Should not happen in valid instances, but handle defensively
             # Cannot compute heuristic without identifying the man
             return float('inf')

        man_loc = None
        for fact in state:
            if match(fact, "at", man_name, "*"):
                man_loc = get_parts(fact)[2]
                break

        if man_loc is None:
             # Man exists but is not at any location? Problem state.
             return float('inf') # Cannot compute heuristic if man has no location

        # 4. Count usable spanners carried by the man (C)
        spanners_carried = {get_parts(fact)[2] for fact in state if match(fact, "carrying", man_name, "*")}
        usable_spanners_carried = {s for s in spanners_carried if "(usable {})".format(s) in state}
        C = len(usable_spanners_carried)

        # 5. Find usable spanners on the ground and their locations
        usable_spanner_locs = {} # {spanner_name: location}
        for fact in state:
            parts = get_parts(fact)
            # Check if fact is (at ?s ?l) where ?s is usable and not carried
            if parts[0] == "at" and len(parts) == 3 and "(usable {})".format(parts[1]) in state:
                 spanner_name = parts[1]
                 # Check if this spanner is NOT being carried by the man
                 if "(carrying {} {})".format(man_name, spanner_name) not in state:
                      usable_spanner_locs[spanner_name] = parts[2]

        # 6. Calculate pickups needed
        pickups_needed = max(0, K - C)

        # 7. Calculate base cost (tighten + pickup actions)
        total_cost = K + pickups_needed

        # 8. Calculate walk cost
        required_locations = set(untightened_nut_locs.values())

        if pickups_needed > 0:
            # Find the locations of the `pickups_needed` closest usable spanners on the ground from `man_loc`.
            if not usable_spanner_locs:
                 # Problem requires pickups but none available on ground. Unsolvable.
                 return float('inf')

            spanner_locs_with_dist = []
            for spanner_loc in usable_spanner_locs.values():
                 # Ensure man_loc and spanner_loc are in the distance matrix and reachable
                 if man_loc in self.dist_matrix and spanner_loc in self.dist_matrix[man_loc]:
                      dist = self.dist_matrix[man_loc][spanner_loc]
                      if dist == float('inf'):
                           # Spanner location is unreachable from man_loc. Unsolvable.
                           return float('inf')
                      spanner_locs_with_dist.append((dist, spanner_loc))
                 else:
                      # Should not happen if graph is built correctly and locations are valid
                      return float('inf') # Location not in graph or dist matrix

            spanner_locs_with_dist.sort()
            # Take the locations of the closest 'pickups_needed' spanners
            required_spanner_locs_to_visit = {loc for dist, loc in spanner_locs_with_dist[:pickups_needed]}
            required_locations.update(required_spanner_locs_to_visit)

        walk_cost = 0
        if required_locations:
            # Ensure man_loc is in dist_matrix
            if man_loc not in self.dist_matrix:
                 return float('inf') # Man's location not in graph

            # Sum of distances from man_loc to each required location
            for loc in required_locations:
                 if loc in self.dist_matrix[man_loc]:
                      dist = self.dist_matrix[man_loc][loc]
                      if dist == float('inf'):
                           # Required location is unreachable from man_loc. Unsolvable.
                           return float('inf')
                      walk_cost += dist
                 else:
                      return float('inf') # Required location not in dist matrix


        total_cost += walk_cost

        return total_cost

