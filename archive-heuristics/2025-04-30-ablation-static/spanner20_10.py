from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions (as provided in Logistics example)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS implementation
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all reachable nodes."""
    distances = {start_node: 0}
    queue = [start_node]
    visited = {start_node}

    while queue:
        current_node = queue.pop(0)
        if current_node in graph: # Handle nodes with no outgoing links
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the number of nuts, the need to pick up spanners, and the travel
    cost for the man to move between his current location, spanner locations,
    and nut locations. It estimates travel based on shortest paths between locations
    and the sequence of operations (get spanner, go to nut, tighten, repeat).

    # Assumptions
    - The problem instance is solvable (enough usable spanners exist initially and are reachable).
    - The man can only carry one spanner at a time.
    - Spanners become unusable after one use.
    - Travel cost between linked locations is 1.

    # Heuristic Initialization
    - Identify all locations and the man's name from the initial state and static facts.
    - Build a graph representing the locations and links.
    - Compute all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners that are currently at a location (not carried).
    4. Check if the man is currently carrying a usable spanner.
    5. Count the number of loose nuts (`N_loose`). If 0, the heuristic is 0.
    6. Calculate the fixed costs: `N_loose` for tighten actions, plus `N_loose` (or `N_loose - 1` if carrying a usable spanner) for pickup actions.
    7. Calculate the travel cost:
       - This involves the man moving from his current location to the first required object (spanner if not carrying one, nut if carrying one).
       - Then, for each subsequent nut (N_loose - 1 times), the man needs to travel from a nut location to a spanner location, and then from that spanner location to the next nut location.
       - Estimate these travel costs using precomputed shortest path distances:
         - `min_dist_m_n`: min distance from man's location to any loose nut location.
         - `min_dist_m_s`: min distance from man's location to any usable spanner location (at a location).
         - `min_dist_s_n`: min distance from any usable spanner location (at a location) to any loose nut location.
         - `min_dist_n_s`: min distance from any loose nut location to any usable spanner location (at a location).
       - Handle cases where required objects (loose nuts, usable spanners at locations) are unavailable or unreachable, returning infinity if the problem is unsolvable from this state.
    8. Sum the fixed costs and the estimated travel cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances."""
        self.task = task
        self.man_name = None
        self.locations = set()
        self.graph = {}
        self.distances = {}

        # Collect all locations from initial state and static links
        all_locations = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                all_locations.add(parts[2])
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                all_locations.add(parts[1])
                all_locations.add(parts[2])
        self.locations = list(all_locations)

        # Build graph from link facts
        self.graph = {loc: [] for loc in self.locations}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                if l1 in self.graph and l2 in self.graph: # Ensure locations are known
                    self.graph[l1].append(l2)
                    self.graph[l2].append(l1) # Links are bidirectional

        # Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs(self.graph, start_loc)

        # Find the man's name
        # Try finding the object in a 'carrying' fact first (only man can carry)
        for fact in task.initial_state:
            if match(fact, "carrying", "*", "*"):
                self.man_name = get_parts(fact)[1]
                break
        # If not found, try finding an object in an 'at' fact that isn't a spanner or nut by name pattern
        # This is a fallback and might be fragile depending on naming conventions.
        if self.man_name is None:
             spanners_nuts_in_init = set()
             for fact in task.initial_state:
                 if match(fact, "at", "*", "*"):
                     obj = get_parts(fact)[1]
                     if obj.startswith('spanner') or obj.startswith('nut'):
                         spanners_nuts_in_init.add(obj)
             for fact in task.initial_state:
                  if match(fact, "at", "*", "*"):
                      obj = get_parts(fact)[1]
                      if obj not in spanners_nuts_in_init:
                           self.man_name = obj
                           break
        # Final fallback, assume 'bob' if no other object found.
        if self.man_name is None:
             self.man_name = 'bob' # This is a guess based on examples.


    def get_distance(self, loc1, loc2):
        """Safely get distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
            return float('inf') # No path
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Man's Location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man must be somewhere if problem is solvable. Should not happen in valid states.
             return float('inf') # Should indicate an invalid state representation

        # 2. Identify Loose Nuts and their Locations
        loose_nuts = set()
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut_name = get_parts(fact)[1]
                loose_nuts.add(nut_name)
        # Find locations for loose nuts
        for nut_name in loose_nuts:
             for fact in state:
                 if match(fact, "at", nut_name, "*"):
                     nut_locations[nut_name] = get_parts(fact)[2]
                     break
        loose_nut_locs = set(nut_locations.values())

        # 3. Identify Usable Spanners at Locations (not carried)
        usable_spanners_at_loc = set()
        spanner_locations = {}
        carried_spanner = None

        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner = get_parts(fact)[2]

        for fact in state:
            if match(fact, "usable", "*"):
                spanner_name = get_parts(fact)[1]
                # Check if this usable spanner is at a location and not carried
                if spanner_name != carried_spanner:
                    for loc_fact in state:
                        if match(loc_fact, "at", spanner_name, "*"):
                            spanner_locations[spanner_name] = get_parts(loc_fact)[2]
                            usable_spanners_at_loc.add(spanner_name)
                            break # Found location, move to next usable spanner

        usable_spanner_locs = set(spanner_locations[s] for s in usable_spanners_at_loc)

        # 4. Check if Man is Carrying a Usable Spanner
        man_carrying_usable = False
        if carried_spanner:
            if f"(usable {carried_spanner})" in state:
                 man_carrying_usable = True

        # 5. Count Loose Nuts
        n_loose = len(loose_nuts)

        # 6. If no loose nuts, goal is reached
        if n_loose == 0:
            return 0

        # 7. Calculate Fixed Costs (tighten + pickup)
        cost = n_loose # tighten actions
        pickups_needed = n_loose
        if man_carrying_usable:
            pickups_needed -= 1
        cost += pickups_needed # pickup actions

        # 8. Calculate Travel Costs

        # Calculate minimum distances needed for travel estimation
        min_dist_m_n = float('inf')
        if loose_nut_locs:
            for loc_n in loose_nut_locs:
                min_dist_m_n = min(min_dist_m_n, self.get_distance(man_loc, loc_n))

        min_dist_m_s = float('inf')
        if usable_spanner_locs:
            for loc_s in usable_spanner_locs:
                 min_dist_m_s = min(min_dist_m_s, self.get_distance(man_loc, loc_s))

        min_dist_s_n = float('inf')
        if usable_spanner_locs and loose_nut_locs:
            for loc_s in usable_spanner_locs:
                for loc_n in loose_nut_locs:
                    min_dist_s_n = min(min_dist_s_n, self.get_distance(loc_s, loc_n))

        min_dist_n_s = float('inf')
        if loose_nut_locs and usable_spanner_locs:
            for loc_n in loose_nut_locs:
                for loc_s in usable_spanner_locs:
                    min_dist_n_s = min(min_dist_n_s, self.get_distance(loc_n, loc_s))

        # Check for unsolvable states based on connectivity and spanners
        # If we need nuts tightened (n_loose > 0)
        # And man cannot reach any loose nut location
        if min_dist_m_n == float('inf'):
             return float('inf')

        # If man is not carrying a usable spanner, he needs to pick one up.
        # This requires a usable spanner to be available at a location, and he must be able to reach it.
        if not man_carrying_usable:
            if not usable_spanner_locs:
                 # No usable spanners at locations, and not carrying one. Unsolvable.
                 return float('inf')
            if min_dist_m_s == float('inf'):
                 # Usable spanners exist at locations, but man cannot reach any of them. Unsolvable.
                 return float('inf')
            # Also need to be able to get from spanner to nut for the first cycle
            if min_dist_s_n == float('inf'):
                 return float('inf')


        # If there is more than one nut, we need subsequent cycles (nut -> spanner -> nut)
        # This requires paths between nut locations and spanner locations, and spanners at locations.
        if n_loose > 1:
             # Need usable spanners at locations for subsequent pickups
             if not usable_spanner_locs:
                  return float('inf')
             # Need paths between nuts and spanners for cycles
             if min_dist_n_s == float('inf') or min_dist_s_n == float('inf'):
                  return float('inf')


        travel_cost = 0
        # Apply travel formula
        if man_carrying_usable:
            # Travel to first nut
            travel_cost += min_dist_m_n
            # Travel for subsequent cycles (nut -> spanner -> nut)
            if n_loose > 1:
                 travel_cost += (n_loose - 1) * (min_dist_n_s + min_dist_s_n)
        else: # Not carrying usable spanner
            # Travel to first spanner, then to first nut
            travel_cost += min_dist_m_s + min_dist_s_n
            # Travel for subsequent cycles (nut -> spanner -> nut)
            if n_loose > 1:
                 travel_cost += (n_loose - 1) * (min_dist_n_s + min_dist_s_n)

        # Add travel cost to total cost
        cost += travel_cost

        return cost
