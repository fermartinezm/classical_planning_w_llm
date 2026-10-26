from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

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
    # Use zip which stops when the shortest iterable is exhausted.
    # This allows matching patterns like ("at", "*", "*") against "(at obj loc)".
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the number of nuts to tighten, the cost to acquire a usable spanner
    if Bob doesn't have one, and the travel cost for Bob to visit all locations
    where loose nuts are found.

    # Assumptions
    - The goal is to tighten all specified nuts.
    - To tighten a nut, Bob must be at the nut's location and carrying a usable spanner.
    - Bob can carry one or more spanners. The heuristic only cares if Bob has *at least one* usable spanner.
    - Action costs are uniform (implicitly 1).

    # Heuristic Initialization
    - Build the location graph based on `link` predicates found in static facts.
    - Collect all relevant locations mentioned in static facts, initial state, and goals.
    - Compute all-pairs shortest paths between these locations using BFS.
    - Identify all spanners and nuts mentioned in the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all nuts that are currently `loose`. These are the nuts that need tightening. If there are no loose nuts, the goal is achieved, and the heuristic is 0.
    2. Initialize the heuristic value `h` with the count of loose nuts. This represents the minimum number of `tighten` actions required.
    3. Determine Bob's current location by finding the fact `(at bob ?loc)` in the state.
    4. Identify the set of unique locations where loose nuts are currently located (`TargetNutLocations`).
    5. Check if Bob is currently carrying *any* spanner that is marked as `usable`.
    6. If Bob is *not* carrying a usable spanner:
       - Identify the set of unique locations where usable spanners are available on the ground (not carried by Bob) (`UsableSpannerLocations`).
       - If there are no such spanners, the problem is likely unsolvable from this state; return infinity.
       - Calculate the minimum travel distance for Bob to reach any location in `UsableSpannerLocations` from his current location.
       - Add this minimum distance plus 1 (for the `pickup` action) to the heuristic `h`. This is the estimated cost to acquire a spanner.
       - The location where Bob would pick up the spanner (the one minimizing the travel cost) becomes the effective starting point for the next travel calculation step.
    7. If Bob *is* carrying a usable spanner, the effective starting point for the next travel calculation step is his current location.
    8. Calculate the minimum travel distance from the effective starting point (determined in the previous step) to reach any location in `TargetNutLocations`. Add this minimum distance to the heuristic `h`. This is the estimated cost to reach the first nut location.
    9. Add a proxy cost for visiting the *remaining* distinct locations in `TargetNutLocations`. A simple estimate is the number of additional distinct locations (`|TargetNutLocations| - 1`), assuming at least one move is needed to travel between each subsequent location. Add this value (or 0 if there's only one or zero target locations) to the heuristic `h`.
    10. Return the final calculated value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and computing
        all-pairs shortest paths for locations.
        """
        self.goals = task.goals

        # Build location graph and collect all locations
        self.location_graph = {}
        self.all_locations = set()
        self.all_spanners = set()
        self.all_nuts = set()

        # Collect locations and build graph from static facts
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Collect locations, spanners, and nuts from initial state and goals
        # This is a heuristic-specific way to identify objects based on predicates
        # they appear in. A more robust parser would use PDDL types.
        for fact in task.initial_state | task.goals:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                if len(parts) == 3: # (at obj loc)
                    obj, loc = parts[1], parts[2]
                    self.all_locations.add(loc)
                    # Attempt to infer type based on naming convention or common predicates
                    if obj != 'bob': # Assume 'bob' is the man
                        # Check if obj appears in spanner/nut related predicates
                        if any(match(f, "usable", obj) for f in task.initial_state | task.goals) or \
                           any(match(f, "carrying", "bob", obj) for f in task.initial_state | task.goals):
                            self.all_spanners.add(obj)
                        elif any(match(f, "loose", obj) for f in task.initial_state | task.goals) or \
                             any(match(f, "tightened", obj) for f in task.initial_state | task.goals):
                            self.all_nuts.add(obj)

            elif predicate == "carrying":
                 if len(parts) == 3 and parts[1] == 'bob': # (carrying bob spanner)
                     self.all_spanners.add(parts[2])
            elif predicate == "usable":
                 if len(parts) == 2: # (usable spanner)
                     self.all_spanners.add(parts[1])
            elif predicate in ["loose", "tightened"]:
                 if len(parts) == 2: # (loose nut) or (tightened nut)
                     self.all_nuts.add(parts[1])

        # Ensure all collected locations are in the graph keys, even if isolated initially
        for loc in self.all_locations:
             self.location_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths
        self.distances = {}
        for start_node in self.all_locations:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS from start_node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.all_locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # If current_node is not in graph keys (isolated location), it has no neighbors
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose nuts
        loose_nuts = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}

        # If no loose nuts, goal is reached
        if not loose_nuts:
            return 0

        # 2. Initialize heuristic with the number of tighten actions needed
        h = len(loose_nuts)

        # 3. Determine Bob's current location
        bob_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_loc = get_parts(fact)[2]
                break
        if bob_loc is None:
             # Bob's location is unknown - indicates an invalid state representation
             return float('inf')

        # 4. Identify the locations of all loose nuts
        target_nut_locations = set()
        # We don't need a map from nut to location for the heuristic calculation logic
        for nut in loose_nuts:
             # Find the location of this loose nut in the current state
             nut_loc = None
             for fact in state:
                 if match(fact, "at", nut, "*"):
                     nut_loc = get_parts(fact)[2]
                     break
             if nut_loc:
                 target_nut_locations.add(nut_loc)
             # else: A loose nut has no location? Invalid state? Assume valid states.


        # 5. Check if Bob is carrying any usable spanner
        bob_carrying_usable_spanner = False
        carried_spanners = {get_parts(fact)[2] for fact in state if match(fact, "carrying", "bob", "*")}
        for spanner in carried_spanners:
            # Check if the carried spanner is usable in the current state
            if f'(usable {spanner})' in state:
                bob_carrying_usable_spanner = True
                break

        # 6. Cost for spanner acquisition if needed
        cost_spanner_acquisition = 0
        loc_after_spanner_acquisition = bob_loc # Where Bob is after potential spanner acquisition

        if not bob_carrying_usable_spanner:
            # Identify locations of usable spanners not carried by Bob
            available_usable_spanner_locations = set()
            for spanner in self.all_spanners:
                # Check if spanner is usable and not carried by Bob
                if f'(usable {spanner})' in state and f'(carrying bob {spanner})' not in state:
                    # Find the location of this available spanner
                    spanner_loc = None
                    for fact in state:
                        if match(fact, "at", spanner, "*"):
                            spanner_loc = get_parts(fact)[2]
                            break
                    if spanner_loc:
                        available_usable_spanner_locations.add(spanner_loc)
                    # else: Usable spanner exists but has no location? Invalid state?

            if not available_usable_spanner_locations:
                # No usable spanners available to pick up - unsolvable
                return float('inf')

            # Find the spanner location that minimizes (dist from BobLoc + 1)
            min_dist_to_spanner_loc = float('inf')
            best_spanner_loc = None

            # Ensure bob_loc is a valid starting point in the distances map
            if bob_loc not in self.distances:
                 # Bob is in an isolated location not in the graph - unsolvable?
                 return float('inf')

            for loc in available_usable_spanner_locations:
                 # Ensure the spanner location is reachable from Bob's location
                 if loc in self.distances[bob_loc] and self.distances[bob_loc][loc] != float('inf'):
                      dist_to_loc = self.distances[bob_loc][loc]
                      if dist_to_loc < min_dist_to_spanner_loc:
                          min_dist_to_spanner_loc = dist_to_loc
                          best_spanner_loc = loc

            if best_spanner_loc is None:
                 # All usable spanner locations are unreachable from Bob's location
                 return float('inf')

            cost_spanner_acquisition = min_dist_to_spanner_loc + 1 # Travel + Pickup
            loc_after_spanner_acquisition = best_spanner_loc # Bob is now conceptually at this location
            h += cost_spanner_acquisition

        # 8. Cost for visiting nut locations, starting from loc_after_spanner_acquisition
        if not target_nut_locations:
             # This case is already handled at the beginning (h=0)
             pass
        else:
            # Ensure the starting location for visiting nuts is valid
            if loc_after_spanner_acquisition not in self.distances:
                 # Location after acquiring spanner is isolated - unsolvable?
                 return float('inf')

            min_dist_to_first_nut_loc = float('inf')
            for loc in target_nut_locations:
                 # Ensure the nut location is reachable from the starting location
                 if loc in self.distances[loc_after_spanner_acquisition] and self.distances[loc_after_spanner_acquisition][loc] != float('inf'):
                      min_dist_to_first_nut_loc = min(min_dist_to_first_nut_loc, self.distances[loc_after_spanner_acquisition][loc])

            if min_dist_to_first_nut_loc == float('inf'):
                 # All nut locations are unreachable from where Bob is
                 return float('inf')

            h += min_dist_to_first_nut_loc # Travel to first nut location

            # 9. Proxy for travel to remaining distinct nut locations
            # Add 1 action for each subsequent distinct location Bob needs to visit.
            h += max(0, len(target_nut_locations) - 1)

        return h

