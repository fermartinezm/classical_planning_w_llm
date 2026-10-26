from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for arity match
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the number of nuts to tighten, the cost to acquire a spanner if needed,
    and the estimated movement cost to visit all locations with loose nuts.

    # Assumptions
    - All spanners relevant to the problem are initially marked as `usable`.
    - The location graph defined by `link` predicates is connected for all relevant locations (shed, gate, spanner locations, nut locations).
    - A spanner, once picked up, remains usable.

    # Heuristic Initialization
    - Builds a graph of locations based on `link` predicates.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of Bob, the locations of all loose nuts,
       the locations of all usable spanners, and whether Bob is currently carrying a spanner.
    2. Count the number of loose nuts. If zero, the heuristic is 0 (goal reached).
    3. Initialize the heuristic cost with the number of loose nuts (representing the `tighten` actions).
    4. Calculate the cost to acquire a spanner if Bob is not carrying one and there are loose nuts:
       - Find the nearest usable spanner location reachable from Bob's current location.
       - If no reachable usable spanner exists but one is needed, the problem is likely unsolvable; return infinity.
       - The cost is the distance from Bob's current location to the nearest spanner location plus 1 for the `pickup` action.
       - Update Bob's hypothetical current location to the spanner's location after pickup for subsequent movement calculations.
    5. Calculate the estimated movement cost to visit all locations containing loose nuts:
       - Start from Bob's current (or hypothetical after spanner pickup) location.
       - Use a greedy approach: repeatedly find the nearest unvisited location with a loose nut.
       - Add the distance to this nearest location to the total movement cost.
       - Update Bob's hypothetical location to the newly visited location.
       - Mark the location as visited and repeat until all unique loose nut locations are visited.
       - If any loose nut location is unreachable from the current hypothetical location, return infinity.
    6. The total heuristic value is the sum of the initial loose nut count (tighten actions), the spanner acquisition cost, and the estimated movement cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and computing distances."""
        # self.goals = task.goals # Not strictly needed for this heuristic's calculation, but available
        static_facts = task.static

        # Build location graph from link facts
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link":
                # Ensure the fact has enough parts for a link predicate
                if len(parts) == 3:
                    loc1, loc2 = parts[1], parts[2]
                    self.location_graph.setdefault(loc1, set()).add(loc2)
                    self.location_graph.setdefault(loc2, set()).add(loc1)
                    locations.add(loc1)
                    locations.add(loc2)
                # else: Ignore malformed link fact

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        all_locations_list = list(locations) # Use a list of unique locations found
        for start_loc in all_locations_list:
            self.distances[start_loc] = self._bfs(start_loc, locations)

    def _bfs(self, start_loc, all_locations):
        """Helper function to perform BFS and find distances from start_loc."""
        dist = {loc: float('inf') for loc in all_locations}
        # Check if start_loc is actually one of the known locations
        if start_loc not in dist:
             # This can happen if the initial state has Bob at a location not linked in static facts
             # or if a nut/spanner is at such a location. Handle gracefully.
             return dist # All distances remain infinity

        dist[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            curr = queue.popleft()
            # Check if curr is in the graph keys before accessing neighbors
            if curr in self.location_graph:
                for neighbor in self.location_graph[curr]:
                    # Ensure neighbor is a known location before updating distance
                    if neighbor in dist and dist[neighbor] == float('inf'):
                        dist[neighbor] = dist[curr] + 1
                        queue.append(neighbor)
        return dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current state facts
        bob_loc = None
        loose_nuts = {} # {nut_name: nut_loc}
        usable_spanners = {} # {spanner_name: spanner_loc}
        bob_carrying = None # spanner_name if carrying

        # Extract relevant facts efficiently
        state_facts_map = {get_parts(fact)[0]: get_parts(fact) for fact in state if get_parts(fact)}
        state_facts_list = list(state) # Keep list for iterating and matching

        # Find Bob's location and what he's carrying
        for fact in state_facts_list:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "at" and len(parts) == 3 and parts[1] == "bob":
                bob_loc = parts[2]
            elif parts[0] == "carrying" and len(parts) == 3 and parts[1] == "bob":
                bob_carrying = parts[2] # Store the spanner name Bob is carrying

        # Find loose nuts and usable spanners
        for fact in state_facts_list:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj.startswith("nut") and match(f"(loose {obj})", "loose", obj): # Check if the corresponding loose fact exists
                      loose_nuts[obj] = loc
                 elif obj.startswith("spanner") and match(f"(usable {obj})", "usable", obj): # Check if usable
                      usable_spanners[obj] = loc


        num_loose_nuts = len(loose_nuts)

        # 2. If no loose nuts, goal is reached
        if num_loose_nuts == 0:
            return 0

        # 3. Base cost: one tighten action per loose nut
        h = num_loose_nuts

        # Locations Bob needs to visit to tighten nuts
        target_nut_locations = set(loose_nuts.values())

        spanner_cost = 0
        current_bob_loc_after_spanner = bob_loc # Where Bob is after potentially getting a spanner

        # 4. Calculate cost to acquire a spanner if needed
        # Need a spanner if Bob isn't carrying one AND there's at least one nut to tighten
        if not bob_carrying:
            # Find nearest usable spanner
            nearest_spanner_dist = float('inf')
            spanner_loc_to_go = None

            # Ensure bob_loc is a known location in our distance map
            if bob_loc not in self.distances:
                 # Bob is at an unknown location, cannot calculate distances
                 return float('inf')

            for spanner_loc in usable_spanners.values():
                # Ensure spanner_loc is a known location
                if spanner_loc in self.distances[bob_loc]:
                    dist = self.distances[bob_loc][spanner_loc]
                    if dist != float('inf') and dist < nearest_spanner_dist:
                        nearest_spanner_dist = dist
                        spanner_loc_to_go = spanner_loc

            if spanner_loc_to_go is not None:
                spanner_cost += nearest_spanner_dist # Move to spanner
                spanner_cost += 1 # Pickup action
                current_bob_loc_after_spanner = spanner_loc_to_go # Bob is now hypothetically at spanner location
            else:
                # Cannot get a spanner, but need one. Problem likely unsolvable.
                # This happens if no usable spanners exist or none are reachable from bob_loc.
                return float('inf')

        # 5. Calculate movement cost to nut locations (Greedy TSP)
        movement_cost = 0
        current_loc = current_bob_loc_after_spanner
        unvisited_nut_locations = set(loose_nuts.values()) # Make a mutable copy

        # Ensure current_loc is in the distance map (should be after spanner step or if bob_carrying)
        if current_loc not in self.distances:
             # This indicates an issue with graph building or location data
             return float('inf') # Cannot calculate distances

        # Check if any loose nut location is reachable from current_loc
        reachable_nut = False
        for nut_loc in unvisited_nut_locations:
            if nut_loc in self.distances[current_loc] and self.distances[current_loc][nut_loc] != float('inf'):
                reachable_nut = True
                break
        if not reachable_nut and unvisited_nut_locations:
             # Loose nuts exist but are unreachable from current location (after getting spanner)
             return float('inf')

        while unvisited_nut_locations:
            nearest_dist = float('inf')
            next_loc = None
            # Find nearest unvisited nut location from current_loc
            for nut_loc in unvisited_nut_locations:
                 # Ensure nut_loc is in the distance map for current_loc
                 if nut_loc in self.distances[current_loc]:
                    dist = self.distances[current_loc][nut_loc]
                    if dist != float('inf') and dist < nearest_dist:
                        nearest_dist = dist
                        next_loc = nut_loc

            if next_loc is None:
                 # This should not happen if reachable_nut was True initially and graph is correct
                 # It implies no path exists from current_loc to any remaining unvisited_nut_locations
                 # which contradicts the initial reachability check unless the set became empty.
                 # If it happens unexpectedly, it suggests unsolvability or graph issue.
                 return float('inf') # Cannot reach remaining nuts

            movement_cost += nearest_dist
            current_loc = next_loc
            unvisited_nut_locations.remove(next_loc)

        # 6. Total heuristic
        h += spanner_cost + movement_cost

        return h
