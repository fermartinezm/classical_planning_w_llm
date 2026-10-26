from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    if len(parts) != len(args):
        return False # Pattern length mismatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It simulates the process of the man sequentially picking up usable spanners
    and walking to each loose nut's location to tighten it.

    # Assumptions:
    - The man can carry at most one spanner at a time.
    - There is no action to drop a spanner, so a spanner is carried until used.
      After use, it becomes unusable. For the next nut, the man must acquire
      a new usable spanner from the ground.
    - Nuts do not move from their initial locations.
    - Spanners do not move unless picked up by the man.
    - The location graph defined by 'link' predicates is undirected.
    - There are enough usable spanners for all loose nuts in solvable problems.

    # Heuristic Initialization
    - Build the location graph from 'link' facts.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify the man object.
    - Identify the static location of each nut from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the goal is already reached (all target nuts are tightened). If so, the heuristic is 0.
    2.  Extract the current state information: man's location, whether he is carrying a usable spanner, locations of usable spanners on the ground, and the set of loose nuts.
    3.  Check if there are enough usable spanners (carried or on ground) for the remaining loose nuts. If not, return infinity (or a large value) as the state is likely unsolvable.
    4.  Initialize the total estimated cost to 0.
    5.  Initialize the heuristic's simulation state: current man location, whether he starts carrying a usable spanner (only relevant for the first nut), and the available usable spanners on the ground.
    6.  Iterate through the set of loose nuts. For each loose nut:
        a.  Get the nut's static location.
        b.  Estimate the cost to tighten this nut:
            i.  If the man is currently carrying a usable spanner (this only happens for the very first nut if he started that way):
                -   The cost is the distance from the man's current location to the nut's location (walk) plus 1 (tighten action).
            ii. If the man is not carrying a usable spanner (this is the case for all nuts if he started without one, and for all nuts after the first one is tightened):
                -   Find the usable spanner on the ground that is closest to the man's current location.
                -   The cost is the distance from the man's current location to the spanner's location (walk) + 1 (pickup) + distance from the spanner's location to the nut's location (walk) + 1 (tighten).
        c.  Add the estimated cost for this nut to the total cost.
        d.  Update the heuristic's simulation state for the *next* nut:
            -   The man's location becomes the location of the nut just tightened.
            -   The spanner used is consumed; the man is no longer carrying a usable spanner. If a spanner was picked up from the ground, remove it from the available ground spanners.
    7.  After iterating through all loose nuts, return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Build location graph from link facts
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Identify all locations mentioned in initial 'at' facts
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                  _, obj, loc = get_parts(fact)
                  locations.add(loc)

        self.locations = list(locations)

        # Compute all-pairs shortest paths
        self.distances = self._compute_all_pairs_shortest_paths(self.locations, self.location_graph)

        # Identify nuts and their static locations from initial state and goals
        self.nut_locations = {}
        all_nuts = set()

        # Nuts are objects that appear in (loose) in initial state or (tightened) in goal state
        for fact in initial_state:
            if match(fact, "loose", "*"):
                all_nuts.add(get_parts(fact)[1])

        for goal_fact in self.goals:
             if match(goal_fact, "tightened", "*"):
                  all_nuts.add(get_parts(goal_fact)[1])

        # Find locations for all identified nuts from initial state
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in all_nuts:
                    self.nut_locations[obj] = loc


        # Identify the man object
        # Find the object that is 'at' a location and is not a nut and is not a spanner.
        # Spanners are objects that are 'usable' or 'carrying'.
        all_locatables_in_init = set()
        spanners_in_init = set()

        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 all_locatables_in_init.add(get_parts(fact)[1])
             if match(fact, "usable", "*"):
                 spanners_in_init.add(get_parts(fact)[1])
             if match(fact, "carrying", "*", "*"):
                 # The second argument of carrying is the spanner
                 spanners_in_init.add(get_parts(fact)[2])

        man_candidates = all_locatables_in_init - all_nuts - spanners_in_init
        assert len(man_candidates) == 1, f"Could not identify unique man object. Candidates: {man_candidates}"
        self.man_name = list(man_candidates)[0]


    def _compute_all_pairs_shortest_paths(self, locations, graph):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        # Ensure all locations are keys in the graph, even if isolated
        # Create a mutable copy for modification
        mutable_graph = {loc: neighbors[:] for loc, neighbors in graph.items()}
        for loc in locations:
             if loc not in mutable_graph:
                  mutable_graph[loc] = []


        for start_loc in locations:
            distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            distances[start_loc][start_loc] = 0

            while q:
                curr_loc, dist = q.popleft()

                # If curr_loc is in graph, iterate neighbors
                if curr_loc in mutable_graph:
                    for neighbor in mutable_graph[curr_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_loc][neighbor] = dist + 1
                            q.append((neighbor, dist + 1))

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Extract current state information
        current_man_location = None
        man_carrying_spanner_name = None # Name of the spanner being carried, if any
        usable_spanners_on_ground_locs = {} # {spanner_name: location}
        loose_nuts = set() # {nut_name}
        usable_spanners_set = set() # {spanner_name}

        # First pass to identify key facts
        for fact in state:
             if match(fact, "at", self.man_name, "*"):
                 current_man_location = get_parts(fact)[2]
             elif match(fact, "carrying", self.man_name, "*"):
                 man_carrying_spanner_name = get_parts(fact)[2]
             elif match(fact, "loose", "*"):
                 loose_nuts.add(get_parts(fact)[1])
             elif match(fact, "usable", "*"):
                 usable_spanners_set.add(get_parts(fact)[1])

        # Second pass to find locations of usable spanners on the ground
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in usable_spanners_set and obj != man_carrying_spanner_name:
                      usable_spanners_on_ground_locs[obj] = loc


        # Determine if carried spanner is usable
        man_carrying_usable_spanner = (man_carrying_spanner_name is not None and
                                       man_carrying_spanner_name in usable_spanners_set)

        # Heuristic simulation state
        h_man_loc = current_man_location
        h_carrying_usable = man_carrying_usable_spanner # Only relevant for the first nut processed
        h_usable_ground_spanners_locs = usable_spanners_on_ground_locs.copy()

        # Check solvability based on spanners
        num_usable_spanners_available = len(h_usable_ground_spanners_locs) + (1 if h_carrying_usable else 0)
        num_loose_nuts = len(loose_nuts)

        if num_loose_nuts > num_usable_spanners_available:
             return float('inf')

        total_cost = 0

        # Process nuts one by one (order doesn't affect total sum in this model)
        remaining_loose_nuts_list = list(loose_nuts)

        for nut_name in remaining_loose_nuts_list:
            nut_location = self.nut_locations.get(nut_name)
            if nut_location is None:
                 # This indicates a problem with __init__ or instance definition
                 # A loose nut must have a static location defined in the initial state
                 return float('inf') # Safety

            cost_for_this_nut = 0

            if h_carrying_usable:
                # Man starts this nut sequence carrying a usable spanner
                # Cost: walk to nut + tighten
                # Check if man's current location can reach nut's location
                walk_cost = self.distances.get(h_man_loc, {}).get(nut_location, float('inf'))
                if walk_cost == float('inf'): return float('inf') # Cannot reach nut location
                cost_for_this_nut = walk_cost + 1 # walk + tighten

                # Update simulation state for the *next* nut
                h_man_loc = nut_location
                h_carrying_usable = False # Spanner used

            else:
                # Man starts this nut sequence NOT carrying a usable spanner
                # Need to find and pick up a usable spanner first
                best_spanner_name = None
                min_walk_to_spanner_cost = float('inf')

                if h_man_loc not in self.distances: return float('inf') # Safety

                # Find the closest usable spanner on the ground
                for s_name, s_loc in h_usable_ground_spanners_locs.items():
                    # Ensure spanner location is reachable from man's location
                    walk_to_spanner_cost = self.distances.get(h_man_loc, {}).get(s_loc, float('inf'))
                    if walk_to_spanner_cost == float('inf'):
                         continue # Skip this spanner, it's unreachable

                    if walk_to_spanner_cost < min_walk_to_spanner_cost:
                        min_walk_to_spanner_cost = walk_to_spanner_cost
                        best_spanner_name = s_name

                if best_spanner_name is None or min_walk_to_spanner_cost == float('inf'):
                    # No reachable usable ground spanners left
                    return float('inf')

                best_spanner_loc = h_usable_ground_spanners_locs[best_spanner_name]

                # Cost: walk to spanner + pickup + walk to nut + tighten
                # Ensure spanner location can reach nut location
                walk_to_nut_cost = self.distances.get(best_spanner_loc, {}).get(nut_location, float('inf'))
                if walk_to_nut_cost == float('inf'):
                     return float('inf') # Cannot reach nut location from spanner location

                cost_for_this_nut = min_walk_to_spanner_cost + 1 + walk_to_nut_cost + 1 # walk1 + pickup + walk2 + tighten

                # Update simulation state for the *next* nut
                h_man_loc = nut_location
                del h_usable_ground_spanners_locs[best_spanner_name] # Spanner used
                h_carrying_usable = False # Spanner used

            total_cost += cost_for_this_nut

        return total_cost
