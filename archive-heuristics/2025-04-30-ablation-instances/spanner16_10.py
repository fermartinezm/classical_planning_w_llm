from collections import deque
import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It sums the number of tighten actions, the number of spanner pickup actions needed,
    and an estimated travel cost for the man to visit all necessary locations (nut locations
    and locations of spanners that need picking up) using a Nearest Neighbor approach.

    # Assumptions
    - Each loose nut requires one tighten action.
    - Each tighten action consumes the usability of one spanner.
    - The man can carry multiple spanners (although the heuristic simplifies this by just counting usable ones carried).
    - Spanners on the ground need to be picked up (1 action) before being used.
    - The set of locations and links is static.
    - Nuts and spanners on the ground are static unless moved by an action (nuts are static, spanners move when picked up).
    - The problem instances are solvable within the defined actions and available objects.
    - The location graph is connected for all relevant locations (man's start, nut locations, usable spanner locations).
    - There is exactly one man object.

    # Heuristic Initialization
    - Builds a graph of locations based on `link` facts found in the static information.
    - Collects all locations mentioned in static `link` facts and initial state `at` facts to ensure all relevant locations are included in the graph.
    - Computes all-pairs shortest paths between these locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify State Components:** Parse the current state to find:
       - The man object and his current location.
       - All loose nuts and their current locations.
       - The number of usable spanners the man is currently carrying.
       - All usable spanners on the ground and their locations.

    2. **Check for Goal State:** If there are no loose nuts, the goal is reached, and the heuristic value is 0.

    3. **Calculate Base Actions:**
       - The number of `tighten_nut` actions needed is equal to the number of loose nuts (`N_loose`). Add `N_loose` to the total cost.
       - Determine how many additional usable spanners the man needs to pick up from the ground. This is `N_spanners_from_ground = max(0, N_loose - usable_spanners_carried)`. Add `N_spanners_from_ground` to the total cost (each pickup is 1 action).

    4. **Check for Unsolvability:** If the total number of usable spanners available (carried + on ground) is less than the number of loose nuts, the problem is likely unsolvable with the available resources. Return infinity in this case. Also, if `N_spanners_from_ground > 0` but there are fewer than `N_spanners_from_ground` usable spanners on the ground, return infinity (assuming spanners cannot be acquired otherwise). Checks are also made to ensure relevant locations are reachable via the graph.

    5. **Identify Required Locations for Travel:** The man needs to visit:
       - The location of every loose nut.
       - The locations of the `N_spanners_from_ground` nearest usable spanners on the ground (relative to the man's current location). Only reachable locations are considered.

    6. **Estimate Travel Cost (Nearest Neighbor):** Estimate the minimum travel cost for the man to visit all required locations starting from his current position. This is done using a Nearest Neighbor-like greedy approach:
       - Start with the man's current location.
       - While there are required locations not yet "visited" in this estimate:
           - Find the required location that is nearest to the current estimated position using precomputed shortest paths.
           - Add the distance to this location to the total cost.
           - Update the current estimated position to this nearest location.
           - Remove the location from the set of required locations.

    7. **Sum and Return:** The total heuristic value is the sum of the base actions (tighten, pickup) and the estimated travel cost. Return this sum.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and computing
        all-pairs shortest paths.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Collect all locations mentioned in static links and initial state 'at' facts
        all_locations = set()
        # Locations from link facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'link' and len(parts) == 3:
                all_locations.add(parts[1])
                all_locations.add(parts[2])
        # Locations from initial state 'at' facts (assuming facts are strings)
        for fact_str in task.initial_state:
             parts = get_parts(fact_str)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 all_locations.add(parts[2])

        self.all_locations = list(all_locations)

        # 2. Build location graph
        self.graph = {loc: [] for loc in self.all_locations}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                # Only add links between known locations
                if l1 in self.graph and l2 in self.graph:
                    self.graph[l1].append(l2)
                    self.graph[l2].append(l1)

        # 3. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.all_locations:
            self.dist[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.graph}
        if start_node not in self.graph:
             # Start node is not in the graph of linked locations, it's isolated.
             # Distances to other nodes remain infinity.
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Ensure u is still in graph (should be if it was added to queue)
            if u in self.graph:
                for v in self.graph[u]:
                    if distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify state components
        man_obj = None
        man_location = None
        all_nuts = set()
        all_spanners = set()

        # Pre-pass to identify all nuts and spanners
        for fact_str in state:
            parts = get_parts(fact_str)
            if parts and parts[0] in ['loose', 'tightened'] and len(parts) == 2:
                all_nuts.add(parts[1])
            elif parts and parts[0] == 'usable' and len(parts) == 2:
                 all_spanners.add(parts[1]) # spanner is parts[1] in (usable spanner_obj)
            elif parts and parts[0] == 'carrying' and len(parts) == 3:
                 all_spanners.add(parts[2]) # spanner is parts[2] in (carrying man_obj spanner_obj)
            # 'at' facts can refer to man, nut, or spanner - handle below

        # Find the man and his location
        for fact_str in state:
            parts = get_parts(fact_str)
            if parts and parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assume the object at a location that is not a known nut or spanner is the man
                if obj not in all_nuts and obj not in all_spanners:
                    man_obj = obj
                    man_location = loc
                    break # Assuming only one man

        # If man location is not in our graph, he is isolated, likely unsolvable
        if man_location is None or man_location not in self.dist:
             return float('inf')


        # Identify loose nuts and their locations
        loose_nut_locs = {} # {nut_obj: location}
        for fact_str in state:
            parts = get_parts(fact_str)
            if parts and parts[0] == 'loose' and len(parts) == 2:
                nut_obj = parts[1]
                # Find location of this nut
                for loc_fact_str in state:
                    loc_parts = get_parts(loc_fact_str)
                    if loc_parts and loc_parts[0] == 'at' and len(loc_parts) == 3 and loc_parts[1] == nut_obj:
                        nut_loc = loc_parts[2]
                        # Ensure nut location is reachable from man's current location
                        if nut_loc in self.dist[man_location] and self.dist[man_location][nut_loc] != float('inf'):
                             loose_nut_locs[nut_obj] = nut_loc
                        else: # Nut location is unreachable from man - problem unsolvable
                            return float('inf')
                        break # Found location for this nut

        # 2. Check for Goal State
        if not loose_nut_locs:
            return 0

        # 3. Count usable spanners carried by the man
        usable_spanners_carried = 0
        for fact_str in state:
            parts = get_parts(fact_str)
            if parts and parts[0] == 'carrying' and len(parts) == 3 and parts[1] == man_obj:
                 spanner_obj = parts[2]
                 if f'(usable {spanner_obj})' in state:
                     usable_spanners_carried += 1

        # 4. Identify usable spanners on the ground and their locations
        usable_spanners_on_ground_locs = {} # {spanner_obj: location}
        for fact_str in state:
            parts = get_parts(fact_str)
            # Check if it's an 'at' fact for a spanner
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] in all_spanners:
                 spanner_obj, loc = parts[1], parts[2]
                 # Check if the spanner is usable and its location is reachable from man's current location
                 if f'(usable {spanner_obj})' in state and loc in self.dist[man_location] and self.dist[man_location][loc] != float('inf'):
                     usable_spanners_on_ground_locs[spanner_obj] = loc
                 elif f'(usable {spanner_obj})' in state: # Usable spanner at unreachable location
                     return float('inf')


        # 5. Calculate counts
        N_loose = len(loose_nut_locs)
        total_usable_spanners_available = usable_spanners_carried + len(usable_spanners_on_ground_locs)

        # 6. Check for unsolvability (not enough spanners total)
        if N_loose > total_usable_spanners_available:
             return float('inf')

        # 7. Calculate number of spanners needed from ground
        N_spanners_from_ground = max(0, N_loose - usable_spanners_carried)

        # If we need spanners from the ground but there aren't enough *reachable* ones
        if N_spanners_from_ground > len(usable_spanners_on_ground_locs):
             # This case is covered by the total count check if all usable spanners are reachable.
             # If some usable spanners are unreachable, len(usable_spanners_on_ground_locs)
             # only counts reachable ones, so this check is correct.
             return float('inf')


        # 8. Initialize base cost (actions other than travel)
        total_cost = N_loose # tighten actions
        total_cost += N_spanners_from_ground # pickup actions

        # 9. Determine required locations for travel
        required_locations = list(loose_nut_locs.values())

        if N_spanners_from_ground > 0:
            # Find the N_spanners_from_ground nearest usable spanner locations on ground from man_location
            spanner_distances_from_man = []
            for spanner_obj, loc in usable_spanners_on_ground_locs.items():
                 # We already filtered for reachable locations when building usable_spanners_on_ground_locs
                 spanner_distances_from_man.append((self.dist[man_location][loc], loc))

            spanner_distances_from_man.sort()
            # Add the locations of the N_spanners_from_ground nearest reachable spanners
            for i in range(min(N_spanners_from_ground, len(spanner_distances_from_man))):
                required_locations.append(spanner_distances_from_man[i][1])

        # Use a set for efficient removal during NN calculation
        required_locations_set = set(required_locations)

        # 10. Estimate travel cost using Nearest Neighbor
        current_l_m = man_location
        while required_locations_set:
            nearest_loc = None
            min_dist = float('inf')

            # Find the nearest required location from the current man location
            for loc in required_locations_set:
                # Ensure current_l_m is in self.dist (it should be, it's the man's location)
                # Ensure loc is in self.dist[current_l_m] (it should be if added to required_locations_set)
                d = self.dist[current_l_m][loc]
                if d < min_dist:
                    min_dist = d
                    nearest_loc = loc

            # If min_dist is still inf, it means remaining required locations are unreachable from the current NN location
            # This implies unsolvability or a graph issue.
            if min_dist == float('inf'):
                 return float('inf') # Should not happen in standard benchmarks if initial checks pass

            total_cost += min_dist
            current_l_m = nearest_loc
            required_locations_set.remove(nearest_loc)

        # 11. Add estimated travel cost (already added in step 10 loop)

        # 12. Return total cost
        return total_cost
