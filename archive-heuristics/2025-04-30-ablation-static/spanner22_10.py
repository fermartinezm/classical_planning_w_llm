from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

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

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts
    that are part of the goal. It sums the cost of the tighten actions, the cost
    of picking up necessary spanners, and an estimated travel cost to reach the
    locations where nuts need tightening.

    # Assumptions
    - The man can only carry one spanner at a time.
    - Each tighten action consumes one usable spanner.
    - The graph of locations connected by 'link' predicates is static.
    - Nut locations are static and provided in the initial state.
    - The problem instance is solvable (unless the heuristic returns a large value).

    # Heuristic Initialization
    - Build the graph of locations based on 'link' predicates found in static facts.
    - Identify all locations mentioned in the initial state, static links, and goals
      to ensure the graph includes all relevant places.
    - Compute all-pairs shortest paths (distances) between these locations using BFS.
    - Identify the static locations of all nuts from the initial state.
    - Identify the goal nuts from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of three main components:
    1.  **Tighten Actions:** Each loose nut that is specified in the goal requires one
        `tighten_nut` action. This contributes 1 to the heuristic for each such nut.
    2.  **Pickup Actions:** To tighten `N` loose goal nuts, `N` usable spanners are
        needed throughout the plan. If the man is currently carrying a usable spanner,
        he needs `N-1` more from locations. If he is not carrying one, he needs `N`
        more from locations. Each required spanner from a location must be picked up,
        costing 1 action. This component sums the number of pickup actions needed.
    3.  **Movement Cost:** The man needs to travel to the location of each loose goal
        nut to tighten it. A simple estimate for the travel cost is the sum of the
        shortest distances from the man's current location to each *distinct* location
        where a loose goal nut is located. This additive distance ignores potential
        optimizations from visiting locations in a specific order (like TSP) but
        provides a simple, efficiently computable estimate of the total travel "effort"
        to reach all necessary work sites.

    The total heuristic value is:
    `h = (Number of loose goal nuts)`
      `+ (Number of spanner pickups needed)`
      `+ (Sum of shortest distances from man's current location to each distinct loose goal nut location)`

    If any loose goal nut location is unreachable from the man's current location,
    or if spanners are needed but no usable spanner is reachable at a location,
    the heuristic returns a large value indicating an likely unsolvable state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts, initial state,
        goals, and computing distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.locations = set()
        links = []
        self.nut_locations = {} # Map nut name to its location (static)

        # Collect all locations from initial state, static links, and goals
        all_facts = list(initial_state) + list(static_facts) + list(self.goals)
        for fact in all_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                links.append((loc1, loc2))
                self.locations.add(loc1)
                self.locations.add(loc2)
            elif parts[0] in ["at", "in-city"]: # "at" for objects, "in-city" for locations
                 if len(parts) > 2: # Ensure there's a location argument
                     self.locations.add(parts[2])
            # Also add locations from goals if they involve locations (e.g., (at obj loc))
            elif parts[0] in ["at"] and len(parts) > 2:
                 self.locations.add(parts[2])


        # Get static nut locations from initial state
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at" and parts[1].startswith("nut"):
                 nut, loc = parts[1], parts[2]
                 self.nut_locations[nut] = loc
                 self.locations.add(loc) # Ensure nut location is in graph nodes

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in list(self.locations): # Iterate over a list copy as locations might be added
            self.distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current_loc, dist = q.popleft()
                self.distances[start_loc][current_loc] = dist

                # Find neighbors
                neighbors = set()
                for l1, l2 in links:
                    if l1 == current_loc:
                        neighbors.add(l2)
                    if l2 == current_loc:
                        neighbors.add(l1)

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened":
                self.goal_nuts.add(parts[1])


    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 not in self.locations or loc2 not in self.locations:
             # This indicates a location not included in the graph construction
             # Should not happen if all relevant locations are added in __init__
             return float('inf')
        # Return infinity if no path exists (location not in distances dict for start_loc)
        return self.distances.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find man's current location and whether he carries a usable spanner
        man_location = None
        carried_spanner_is_usable = False

        # Find usable spanners at locations
        usable_spanners_at_locs = {} # {spanner_name: location}

        # Find loose goal nuts
        loose_goal_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj.startswith("bob"): # Assuming 'bob' is the man
                    man_location = loc
                elif obj.startswith("spanner") and "(usable " + obj + ")" in state:
                     usable_spanners_at_locs[obj] = loc
            elif parts[0] == "carrying":
                carrier, spanner = parts[1], parts[2]
                if carrier.startswith("bob") and "(usable " + spanner + ")" in state:
                    carried_spanner_is_usable = True
            elif parts[0] == "loose":
                nut = parts[1]
                if nut in self.goal_nuts: # Only care about loose nuts that are goals
                    loose_goal_nuts.add(nut)

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # Heuristic calculation
        h_cost = 0

        # 1. Cost for tighten actions
        h_cost += len(loose_goal_nuts)

        # 2. Cost for spanner acquisition (pickups)
        spanners_needed_count = len(loose_goal_nuts)
        spanners_carried_count = 1 if carried_spanner_is_usable else 0
        pickups_needed_count = max(0, spanners_needed_count - spanners_carried_count)

        h_cost += pickups_needed_count

        # Check if spanners are needed but none are reachable at locations
        if pickups_needed_count > 0:
             min_dist_to_any_spanner_loc = float('inf')
             for spanner, loc in usable_spanners_at_locs.items():
                  dist = self.get_distance(man_location, loc)
                  min_dist_to_any_spanner_loc = min(min_dist_to_any_spanner_loc, dist)

             if min_dist_to_any_spanner_loc == float('inf'):
                  # Needs spanner from a location but none are reachable
                  return 1000000 # Return a large value indicating unsolvable

        # 3. Cost for movement to distinct loose goal nut locations
        distinct_loose_nut_locations = {self.nut_locations[nut] for nut in loose_goal_nuts}

        for loc in distinct_loose_nut_locations:
             dist = self.get_distance(man_location, loc)
             if dist == float('inf'):
                  # A loose goal nut is at an unreachable location
                  return 1000000 # Return a large value indicating unsolvable
             h_cost += dist

        return h_cost
