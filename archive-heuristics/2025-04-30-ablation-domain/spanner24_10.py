from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It calculates the cost based on the number of loose nuts, the cost for Bob
    to acquire a spanner if he doesn't have one, and the estimated movement cost
    for Bob to visit the location of each loose nut. The movement cost is estimated
    greedily by always moving to the nearest remaining loose nut location.

    # Assumptions
    - There is only one man, 'bob'.
    - Nut locations are static (do not change during planning).
    - Spanner locations can change, but usable spanners are identified from the initial state/static facts.
    - All links between locations are bidirectional.
    - All usable spanners are interchangeable for tightening any nut.
    - The location graph formed by links is connected for all relevant locations (shed, gate, locations with nuts/spanners).
    - The cost of each action (move, pickup, drop, tighten) is 1.

    # Heuristic Initialization
    - Build the graph of locations based on `link` facts.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify all nuts and their fixed locations from the initial state.
    - Identify all usable spanners from the initial state/static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify Bob's current location and whether he is carrying a spanner.
    2. Identify all nuts that are currently loose (not tightened).
    3. Identify the current locations of all usable spanners.
    4. If there are no loose nuts, the heuristic is 0 (goal state).
    5. Initialize the heuristic cost `h` to 0.
    6. If Bob is not currently carrying a spanner and there are loose nuts remaining:
       - Calculate the minimum distance from Bob's current location to any location containing a usable spanner.
       - Add this minimum distance plus 1 (for the pickup action) to `h`.
       - Update Bob's current location to the location of the picked-up spanner. Bob now has a spanner.
    7. While there are still loose nuts remaining:
       - Find the loose nut whose location is nearest to Bob's current location using the precomputed distances.
       - Add the distance to this nearest nut's location to `h`.
       - Update Bob's current location to the nearest nut's location.
       - Add 1 to `h` for the `tighten` action performed at this location.
       - Mark the nut as no longer loose (remove it from the set of remaining loose nuts).
    8. Return the total calculated cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        location graph, distances, nut locations, and spanner list.
        """
        self.goals = task.goals

        # 1. Identify all locations, nuts, and spanners from initial state, goals, and static facts
        all_locations = set()
        self.all_nuts = set()
        self.spanners = set()

        facts_to_process = task.initial_state | task.static | task.goals

        for fact in facts_to_process:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             args = parts[1:]

             if predicate == 'link' and len(args) == 2:
                 all_locations.add(args[0])
                 all_locations.add(args[1])
             elif predicate == 'at' and len(args) == 2:
                 # The second argument of 'at' is a location
                 all_locations.add(args[1])
             elif predicate == 'loose' and len(args) == 1:
                 self.all_nuts.add(args[0])
             elif predicate == 'tightened' and len(args) == 1:
                 self.all_nuts.add(args[0])
             elif predicate == 'usable' and len(args) == 1:
                 self.spanners.add(args[0])
             elif predicate == 'carrying' and len(args) == 2 and args[0] == 'bob':
                 # If bob is carrying something, it must be a spanner in this domain
                 self.spanners.add(args[1])


        self.all_locations = list(all_locations) # Keep a list if needed, set is fine too

        # 2. Build location graph
        self.location_graph = {loc: [] for loc in self.all_locations}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                if loc1 in self.location_graph and loc2 in self.location_graph:
                    self.location_graph[loc1].append(loc2)
                    self.location_graph[loc2].append(loc1) # Assuming links are bidirectional

        # 3. Compute all-pairs shortest path distances using BFS
        self.location_distances = {}
        for start_loc in self.all_locations:
            distances = {loc: float('inf') for loc in self.all_locations}
            distances[start_loc] = 0
            queue = deque([start_loc])

            while queue:
                curr = queue.popleft()
                # Ensure curr is a valid key before accessing graph
                if curr in self.location_graph:
                    for neighbor in self.location_graph[curr]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[curr] + 1
                            queue.append(neighbor)
            self.location_distances[start_loc] = distances

        # 4. Store static locations of nuts from the initial state
        self.nut_locations = {}
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 if obj in self.all_nuts:
                     self.nut_locations[obj] = loc


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Extract Bob's current state and spanner locations from the current state
        bob_loc = None
        bob_has_spanner = False
        current_spanner_locs = set()
        tightened_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2 and args[0] == 'bob':
                bob_loc = args[1]
            elif predicate == 'carrying' and len(args) == 2 and args[0] == 'bob':
                # Check if the carried object is indeed a spanner we identified
                if args[1] in self.spanners:
                    bob_has_spanner = True # We only care if he has *a* spanner
            elif predicate == 'at' and len(args) == 2 and args[0] in self.spanners:
                 current_spanner_locs.add(args[1])
            elif predicate == 'tightened' and len(args) == 1 and args[0] in self.all_nuts:
                 tightened_nuts.add(args[0])

        # 2. Identify loose nuts in the current state
        loose_nuts = set(self.all_nuts) - tightened_nuts

        # 3. Goal check
        if not loose_nuts:
            return 0 # All nuts tightened

        # Heuristic calculation starts
        h = 0
        current_bob_loc = bob_loc
        current_bob_has_spanner = bob_has_spanner
        remaining_loose_nuts = set(loose_nuts) # Copy for modification

        # Ensure Bob has a spanner if needed for the remaining nuts
        # This step is performed once if Bob doesn't have a spanner initially
        if not current_bob_has_spanner:
            min_spanner_dist = float('inf')
            nearest_spanner_loc = None

            # Find the nearest usable spanner location on the ground
            for sloc in current_spanner_locs:
                 # Check if locations are valid and reachable
                 if current_bob_loc in self.location_distances and sloc in self.location_distances[current_bob_loc]:
                    dist = self.location_distances[current_bob_loc][sloc]
                    if dist < min_spanner_dist:
                        min_spanner_dist = dist
                        nearest_spanner_loc = sloc

            # If a spanner location was found on the ground
            if nearest_spanner_loc is not None and min_spanner_dist != float('inf'):
                # Cost to move to spanner + pickup
                h += min_spanner_dist + 1
                current_bob_loc = nearest_spanner_loc
                current_bob_has_spanner = True
            # Note: If Bob doesn't have a spanner and none are on the ground,
            # this heuristic might underestimate if a spanner needs to be dropped first by Bob
            # (which is not possible if he doesn't have one) or moved by another agent (not in domain).
            # Assuming solvable problems don't get into such dead ends based on spanners.


        # Now, Bob has a spanner (or the heuristic assumes he can get one) and is at current_bob_loc.
        # Greedily visit remaining loose nuts.
        while remaining_loose_nuts:
            min_dist_to_nut = float('inf')
            nearest_nut = None
            nearest_nut_loc = None

            # Find the nearest loose nut location
            for nut in remaining_loose_nuts:
                loc_n = self.nut_locations.get(nut) # Get fixed location of the nut
                # Check if nut location is known and reachable from current Bob location
                if loc_n and current_bob_loc in self.location_distances and loc_n in self.location_distances[current_bob_loc]:
                    dist = self.location_distances[current_bob_loc][loc_n]
                    if dist < min_dist_to_nut:
                        min_dist_to_nut = dist
                        nearest_nut = nut
                        nearest_nut_loc = loc_n

            # If a reachable loose nut was found
            if nearest_nut_loc is not None and min_dist_to_nut != float('inf'):
                # Cost to move to the nut's location
                h += min_dist_to_nut
                current_bob_loc = nearest_nut_loc

                # Cost to tighten the nut
                h += 1
                remaining_loose_nuts.remove(nearest_nut)
            else:
                 # This case implies a remaining loose nut location is not reachable from current_bob_loc.
                 # In a solvable problem with a connected graph covering all relevant locations,
                 # this should not happen if Bob started in a reachable part of the graph.
                 # If it happens, the problem might be unsolvable from this state, or the graph is disconnected.
                 # For a non-admissible heuristic, we can break or return a large value.
                 # Breaking might underestimate if other nuts were reachable.
                 # Returning infinity would be more accurate for unsolvability but might affect search.
                 # Let's break, assuming graph connectivity for solvable problems.
                 break

        return h
