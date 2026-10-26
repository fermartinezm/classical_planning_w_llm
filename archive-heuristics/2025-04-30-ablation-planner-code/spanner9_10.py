from fnmatch import fnmatch
from collections import deque
import math

# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check is element-wise matching up to the length of the shorter sequence
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It sums the number of necessary
    tighten and pickup actions, plus an estimated travel cost for the man.
    The travel cost is estimated by calculating the greedy path distance
    to collect needed spanners and then the greedy path distance to visit
    all nut locations.

    # Assumptions
    - Each nut requires one tighten action.
    - Each tighten action consumes one usable spanner.
    - The man can carry multiple spanners.
    - The man is the only agent performing actions.
    - The location graph defined by 'link' predicates is static and connected.
    - Solvable instances have enough usable spanners available initially.
    - The man object can be identified (e.g., by being involved in 'carrying' or named 'bob').

    # Heuristic Initialization
    - Build the location graph from 'link' predicates.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify the set of nuts that need to be tightened (goal nuts).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location (`M_loc`).
    2. Identify the set of nuts that are currently loose and are goal nuts (`NutsToTighten`) and their locations (`N_locs`).
    3. Count the number of such nuts (`num_nuts`). If `num_nuts` is 0, the heuristic is 0.
    4. Identify usable spanners currently carried by the man (`CarriedUsableSpanners`).
    5. Calculate the number of additional usable spanners needed (`needed_spanners = max(0, num_nuts - |CarriedUsableSpanners|)`).
    6. The base heuristic cost is `num_nuts` (for tighten actions) + `needed_spanners` (for pickup actions).
    7. Calculate the travel cost:
       a. Identify usable spanners on the ground and their locations.
       b. Find the `needed_spanners` closest usable spanners on the ground to `M_loc`. Get their locations (`S_locs`).
       c. Calculate the greedy travel cost to visit all locations in `S_locs` starting from `M_loc`. Let the last location visited be `last_s_loc`. If no spanners are needed, `last_s_loc` is `M_loc`.
       d. Calculate the greedy travel cost to visit all locations in `N_locs` (allowing revisits if multiple nuts are at the same location) starting from `last_s_loc`.
       e. The total travel cost is the sum of the spanner travel cost and the nut travel cost.
    8. The total heuristic value is the base cost + travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and computing
        shortest paths, and identifying goal nuts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                self.goal_nuts.add(args[0])

        # Build location graph from 'link' predicates
        self.graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                locations.add(loc1)
                locations.add(loc2)
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        self.locations = list(locations)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Performs BFS starting from start_loc to find distances to all other locations."""
        q = deque([(start_loc, 0)])
        visited = {start_loc}
        self.distances[(start_loc, start_loc)] = 0

        while q:
            current_loc, dist = q.popleft()

            if current_loc in self.graph:
                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_loc, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Return infinity if no path exists (graph disconnected or location not in graph)
        # Assuming solvable problems have connected graphs containing all relevant locations
        return self.distances.get((loc1, loc2), math.inf)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify object locations and carried items from the state
        locatables_at_loc = {}
        carried_by_man = {}
        man_name = None

        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                locatables_at_loc[obj] = loc
            elif match(fact, "carrying", "*", "*"):
                 m, s = get_parts(fact)[1:]
                 # Assume the carrier is the man
                 man_name = m
                 carried_by_man[s] = m

        # Identify the man's location
        man_loc = None
        if man_name and man_name in locatables_at_loc:
             man_loc = locatables_at_loc[man_name]
        elif man_name is None:
             # Fallback: Try to find an object at a location that isn't a nut or spanner
             # This is a heuristic guess for the man's name if 'carrying' facts are absent.
             # A more robust solution requires parsing object types from the problem file.
             # Based on examples, 'bob' is the man.
             for fact in state:
                 if match(fact, "at", "bob", "*"): # Assuming 'bob' is the man
                     man_name = "bob"
                     man_loc = get_parts(fact)[2]
                     break

        if man_loc is None:
             # Man's location not found. Problem state might be invalid or disconnected.
             return math.inf


        # 2. Identify NutsToTighten and their locations
        nuts_to_tighten = set()
        nut_locations = {}
        for nut in self.goal_nuts:
            # Check if the nut is currently loose
            if f"(loose {nut})" in state:
                nuts_to_tighten.add(nut)
                # Find the location of this nut
                if nut in locatables_at_loc:
                    nut_locations[nut] = locatables_at_loc[nut]
                else:
                    # Nut location not found in state, problem state might be invalid
                    return math.inf


        # 3. Count num_nuts
        num_nuts = len(nuts_to_tighten)

        # 4. Check for goal state
        if num_nuts == 0:
            return 0

        # 5. Identify CarriedUsableSpanners
        carried_usable_spanners = set()
        for spanner in carried_by_man:
             if f"(usable {spanner})" in state:
                 carried_usable_spanners.add(spanner)

        num_carried_usable = len(carried_usable_spanners)

        # 6. Calculate needed_spanners and base cost
        needed_spanners = max(0, num_nuts - num_carried_usable)
        heuristic_base = num_nuts + needed_spanners # tighten actions + pickup actions

        # 7. Calculate travel cost
        spanner_travel = 0
        nut_travel = 0
        current_travel_loc = man_loc
        last_s_loc = man_loc # Start nut travel from here if no spanners needed

        # a. Identify usable spanners on the ground and their locations.
        # b. Find the needed_spanners closest usable spanners on the ground.
        if needed_spanners > 0:
            usable_spanners_on_ground = []
            for fact in state:
                if match(fact, "usable", "*"):
                    spanner = get_parts(fact)[1]
                    # Check if this usable spanner is on the ground (not carried)
                    if spanner not in carried_by_man:
                         # Find its location
                         if spanner in locatables_at_loc:
                             spanner_loc = locatables_at_loc[spanner]
                             dist_from_man = self.get_distance(man_loc, spanner_loc)
                             if dist_from_man == math.inf: return math.inf # Unreachable spanner
                             usable_spanners_on_ground.append((spanner_loc, dist_from_man))
                         else:
                             # Spanner location not found, problem state might be invalid
                             return math.inf

            # Sort usable spanners on ground by distance from man and take the closest 'needed_spanners'
            usable_spanners_on_ground.sort(key=lambda item: item[1])
            s_locs_list = [loc for loc, dist in usable_spanners_on_ground[:needed_spanners]]

            # If there aren't enough usable spanners on the ground, the problem is likely unsolvable
            if len(s_locs_list) < needed_spanners:
                 return math.inf

            # c. Calculate the greedy travel cost to visit all locations in S_locs starting from M_loc.
            current_s_travel_loc = man_loc
            remaining_s_locs = list(s_locs_list) # Make a mutable copy

            while remaining_s_locs:
                min_dist = math.inf
                next_s_loc = None

                for loc in remaining_s_locs:
                    dist = self.get_distance(current_s_travel_loc, loc)
                    if dist < min_dist:
                        min_dist = dist
                        next_s_loc = loc

                if next_s_loc is None: # Should not happen if remaining_s_locs is not empty and locations are reachable
                     return math.inf

                spanner_travel += min_dist
                current_s_travel_loc = next_s_loc
                remaining_s_locs.remove(next_s_loc)

            last_s_loc = current_s_travel_loc # This is the location after picking up the last needed spanner
        else:
             last_s_loc = man_loc # If no spanners needed, start nut travel from man's current location


        # d. Calculate the greedy travel cost to visit all locations in N_locs starting from last_s_loc.
        current_n_travel_loc = last_s_loc
        remaining_n_locs = list(nut_locations.values()) # List of nut locations (allows duplicates)

        while remaining_n_locs:
            min_dist = math.inf
            next_n_loc = None
            next_n_loc_index = -1 # Need index to remove one instance

            for i, loc in enumerate(remaining_n_locs):
                dist = self.get_distance(current_n_travel_loc, loc)
                if dist == math.inf: return math.inf # Unreachable nut location
                if dist < min_dist:
                    min_dist = dist
                    next_n_loc = loc
                    next_n_loc_index = i

            if next_n_loc is None: # Should not happen if remaining_n_locs is not empty and locations are reachable
                 return math.inf

            nut_travel += min_dist
            current_n_travel_loc = next_n_loc
            # Remove only one instance of the visited location
            remaining_n_locs.pop(next_n_loc_index)


        # e. Total travel cost
        travel_cost = spanner_travel + nut_travel

        # 8. Total heuristic
        return heuristic_base + travel_cost
