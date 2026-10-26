import math
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It sums the cost of tightening each nut,
    the cost to acquire a spanner (if not already carrying one), and the
    estimated travel cost to visit all locations with loose nuts. The travel
    cost is estimated using a greedy approach (Nearest Neighbor).

    # Assumptions
    - The man ('bob') is the only agent.
    - The man needs to carry a spanner to tighten a nut.
    - One spanner is sufficient to tighten all nuts.
    - Spanners do not break or become unusable during the plan.
    - The location graph defined by 'link' facts is static.
    - All locations, nuts, and spanners relevant to the problem are mentioned
      in the initial state or goals.
    - Travel between linked locations costs 1 action.
    - Pickup and Tighten actions cost 1 action.

    # Heuristic Initialization
    - Identify all locations from static 'link' facts and initial 'at' facts.
    - Build the location graph based on 'link' facts.
    - Precompute all-pairs shortest path distances between all identified locations
      using Breadth-First Search (BFS).
    - Identify all nuts that are part of the goal condition.
    - Identify all spanners present in the initial state (based on 'usable' facts).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Determine if the man is currently carrying a spanner.
    3. Identify all nuts that are loose in the current state AND are part of the goal.
    4. Identify the locations of these loose nuts.
    5. Identify all usable spanners in the current state and their locations.
    6. If there are no loose nuts that are goals, the heuristic is 0.
    7. Initialize the heuristic value with the number of loose nuts (representing the 'tighten' action cost for each).
    8. Determine the man's effective starting location for subsequent tasks. This is his current location.
    9. If the man is not carrying a spanner:
       - Find the minimum travel distance from the man's current location to any location containing a usable spanner.
       - If no usable spanner is reachable, the state is likely unsolvable; return a large value (infinity).
       - Add this minimum distance plus 1 (for the 'pickup' action) to the heuristic.
       - Update the man's effective starting location to the location of the closest usable spanner found.
    10. The man (conceptually) now has a spanner and is at the effective starting location. He needs to visit all locations with loose nuts.
    11. Estimate the travel cost to visit all loose nut locations using a greedy Nearest Neighbor approach:
        - Start from the current effective location.
        - While there are unvisited loose nut locations:
            - Find the closest unvisited loose nut location.
            - Add the distance to this location to the heuristic.
            - Update the current effective location to this newly visited nut location.
            - Mark the location as visited.
        - If any required nut location is unreachable, return a large value (infinity).
    12. Return the total accumulated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        self.location_graph = {}
        self.all_locations = set()
        self.all_nuts_in_goals = set()
        self.all_spanners_in_init = set()

        # Collect locations and build graph from static 'link' facts
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Collect locations from initial 'at' facts (may include locations not in 'link')
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 self.all_locations.add(loc)

        # Collect nuts from goal conditions
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.all_nuts_in_goals.add(nut)

        # Collect spanners from initial state (based on 'usable' facts)
        for fact in initial_state:
            if match(fact, "usable", "*"):
                 _, spanner = get_parts(fact)
                 self.all_spanners_in_init.add(spanner)
            # Also check for spanners being carried or at locations initially
            # This helps identify spanners even if not marked usable initially,
            # assuming anything carried or at a location that isn't bob/nut/location is a spanner.
            # A more robust parser would use domain type info.
            # For this domain, 'usable' seems the primary way to identify relevant spanners.
            # Let's stick to 'usable' for spanner identification in init for simplicity.


        # Precompute all-pairs shortest paths using BFS
        self.distances = {loc: {} for loc in self.all_locations}
        large_value = float('inf') # Use infinity for unreachable locations

        for start_loc in self.all_locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0

            while q:
                current_loc, d = q.popleft()

                # Get neighbors from the graph, handle locations not in graph keys
                neighbors = self.location_graph.get(current_loc, [])

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_loc][neighbor] = d + 1
                        q.append((neighbor, d + 1))

            # Mark unreachable locations with large_value
            for loc in self.all_locations:
                if loc not in self.distances[start_loc]:
                    self.distances[start_loc][loc] = large_value

    def get_distance(self, loc1, loc2):
        """Lookup precomputed distance between two locations."""
        if loc1 not in self.all_locations or loc2 not in self.all_locations:
            # Should not happen if all locations are collected correctly, but safety check
            return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        large_value = float('inf')

        # 1. Extract state information
        man_location = None
        carrying_spanner = False
        loose_nuts_in_state = set()
        nut_locations = {} # Map nut -> location in current state
        spanner_locations = {} # Map spanner -> location in current state
        usable_spanners_in_state = set() # Usable spanners in current state

        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_location = get_parts(fact)[2]
            elif match(fact, "carrying", "bob", "*"):
                carrying_spanner = True # Only need to know if carrying *a* spanner
            elif match(fact, "loose", "*"):
                _, nut = get_parts(fact)
                if nut in self.all_nuts_in_goals: # Only care about nuts that are goals
                    loose_nuts_in_state.add(nut)
            elif match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.all_nuts_in_goals:
                    nut_locations[obj] = loc
                # Identify spanners based on the spanners found in init
                if obj in self.all_spanners_in_init:
                     spanner_locations[obj] = loc
            elif match(fact, "usable", "*"):
                _, spanner = get_parts(fact)
                if spanner in self.all_spanners_in_init: # Only care about spanners we know about
                    usable_spanners_in_state.add(spanner)

        # Ensure man_location was found (should always be the case in a valid state)
        if man_location is None:
             # This indicates an invalid state representation or domain assumption failure
             return large_value # Cannot proceed

        # 2. Count loose nuts that are goals
        num_loose_nuts = len(loose_nuts_in_state)

        # 6. If no loose nuts, goal is reached
        if num_loose_nuts == 0:
            return 0

        # 7. Initialize heuristic with tighten action cost
        h = num_loose_nuts

        # 8. Determine effective starting location
        current_pos = man_location

        # 9. Cost to get a spanner if needed
        if not carrying_spanner:
            min_dist_to_spanner = large_value
            closest_spanner_loc = None

            usable_spanner_locations = {spanner_locations[s] for s in usable_spanners_in_state if s in spanner_locations}

            if not usable_spanner_locations:
                 # No usable spanners available in the current state
                 return large_value # Unsolvable from this state

            for loc_s in usable_spanner_locations:
                dist = self.get_distance(current_pos, loc_s)
                if dist == large_value:
                    # This spanner location is unreachable
                    continue
                if dist < min_dist_to_spanner:
                    min_dist_to_spanner = dist
                    closest_spanner_loc = loc_s

            if closest_spanner_loc is None:
                 # All usable spanner locations are unreachable
                 return large_value # Unsolvable from this state

            h += min_dist_to_spanner + 1 # Travel to spanner + pickup
            current_pos = closest_spanner_loc # Man is now conceptually at spanner location

        # 10. Man has spanner, needs to visit loose nut locations.
        loose_nut_locations = {nut_locations[nut] for nut in loose_nuts_in_state if nut in nut_locations}

        if not loose_nut_locations and num_loose_nuts > 0:
             # Loose nuts exist but their locations are unknown in the state?
             # Or perhaps they are carried by something? Domain doesn't suggest this.
             # Assuming loose nuts must have an 'at' fact if not carried (which is not possible for nuts).
             # This case indicates an inconsistency or unsolvable state.
             return large_value


        # 11. Estimate travel cost to visit loose nut locations (Greedy Nearest Neighbor)
        unvisited_nut_locations = set(loose_nut_locations)
        travel_cost = 0

        while unvisited_nut_locations:
            min_dist_to_nut = large_value
            next_nut_loc = None

            for nut_loc in unvisited_nut_locations:
                dist = self.get_distance(current_pos, nut_loc)
                if dist == large_value:
                    # This nut location is unreachable from the current position
                    return large_value # Unsolvable from this state

                if dist < min_dist_to_nut:
                    min_dist_to_nut = dist
                    next_nut_loc = nut_loc

            # Should always find a next_nut_loc if unvisited_nut_locations is not empty
            # and no location is unreachable (handled above).
            if next_nut_loc is None:
                 # This indicates an unexpected error or state inconsistency
                 return large_value # Should not happen if reachable check passed

            travel_cost += min_dist_to_nut
            current_pos = next_nut_loc # Man is now conceptually at this nut location
            unvisited_nut_locations.remove(next_nut_loc)

        h += travel_cost

        # 12. Return total heuristic value
        return h

