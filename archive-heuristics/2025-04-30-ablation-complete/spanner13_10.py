# Assuming Heuristic base class is available in a module named heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque
import math # Import math for float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or invalid format defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the total number of actions (walk, pickup_spanner, tighten_nut)
    required to tighten all loose nuts. It does this by greedily processing each loose nut,
    calculating the cost to get the man to the nut's location, acquire a usable spanner
    (either carried or by picking up the closest available one), and perform the tighten action.

    # Assumptions
    - The man can only carry one spanner at a time (implied by the domain predicates).
    - A spanner becomes unusable after one tighten action.
    - The location graph defined by 'link' predicates is connected for all relevant locations
      in solvable instances.
    - There are enough usable spanners available (either carried or at locations) to tighten
      all loose nuts in solvable instances.
    - The man object can be identified (currently infers from 'carrying' or 'at' predicates).
    - Nuts are objects appearing in 'tightened' goal predicates or 'loose' state predicates.
    - Spanners are objects appearing in 'usable' or 'carrying' predicates.

    # Heuristic Initialization
    - Builds a graph of locations based on 'link' predicates.
    - Pre-calculates shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners currently carried by the man.
    4. Identify all usable spanners available at locations.
    5. If there are no loose nuts, the heuristic is 0.
    6. Initialize the total heuristic cost to 0.
    7. Keep track of the man's current location (starts at the initial man location).
    8. Keep track of the number of usable spanners the man is currently carrying.
    9. Keep track of the usable spanners available at locations (as a mutable list).
    10. Sort the loose nuts by their distance from the man's initial location (this is a greedy choice to process closer nuts first).
    11. Iterate through the sorted list of loose nuts:
        a. Calculate the distance from the man's current location to the current nut's location. Add this distance to the total heuristic cost. Update the man's current location to the nut's location. If the nut location is unreachable, return infinity.
        b. Check if the man is carrying a usable spanner.
           - If yes, decrement the count of carried spanners. No extra cost for spanner acquisition for this nut.
           - If no, a spanner must be picked up.
             - Find the closest usable spanner available at a location, relative to the man's *current* location (which is the nut's location).
             - If no usable spanners are available or reachable, return infinity (unsolvable state).
             - Calculate the distance to this spanner location. Add this distance to the total heuristic cost. Update the man's current location to the spanner's location.
             - Add 1 to the total heuristic cost for the 'pickup_spanner' action.
             - Calculate the distance from the spanner location back to the nut's location. Add this distance to the total heuristic cost. Update the man's current location back to the nut's location. If the nut location is unreachable from the spanner location, return infinity.
             - Remove the picked-up spanner from the list of available spanners at locations.
        c. Add 1 to the total heuristic cost for the 'tighten_nut' action.
    12. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the location graph, and pre-calculating distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build location graph from 'link' facts
        self.locations = set()
        self.graph = {} # Adjacency list: {loc: [neighbor1, neighbor2, ...]}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.graph.setdefault(l1, []).append(l2)
                self.graph.setdefault(l2, []).append(l1) # Links are bidirectional

        # Ensure all locations mentioned in links are in the graph keys
        for loc in self.locations:
             self.graph.setdefault(loc, [])

        # Pre-calculate all-pairs shortest paths using BFS
        self.distances = {} # {(loc1, loc2): distance}
        for start_node in self.locations:
            queue = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while queue:
                (current_loc, dist) = queue.popleft()

                for neighbor in self.graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

            # Handle unreachable locations within the graph
            for loc in self.locations:
                if (start_node, loc) not in self.distances:
                    self.distances[(start_node, loc)] = math.inf


    def get_distance(self, loc1, loc2):
        """Looks up the pre-calculated distance between two locations."""
        # Return infinity if either location is not in our graph of linked locations
        # This handles cases where objects might be at locations not part of the movement graph
        if loc1 not in self.locations or loc2 not in self.locations:
             return math.inf
        return self.distances.get((loc1, loc2), math.inf)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Identify the man object name (brittle inference)
        man_name = None
        # Try to find the object that is 'carrying' something
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'carrying' and len(parts) == 3:
                man_name = parts[1]
                break
        # If not found by 'carrying', assume the object at a location that isn't a nut or spanner
        # This requires knowing all nuts and spanners first.
        if man_name is None:
             # Identify nuts (objects that appear in 'tightened' goals or 'loose' state facts)
            all_nuts = set()
            for goal in self.goals:
                 parts = get_parts(goal)
                 if parts and parts[0] == 'tightened' and len(parts) == 2:
                     all_nuts.add(parts[1])
            for fact in state:
                 parts = get_parts(fact)
                 if parts and parts[0] in ['loose', 'tightened'] and len(parts) == 2:
                     all_nuts.add(parts[1])

            # Identify spanners (objects that appear in 'usable' or 'carrying' state facts)
            all_spanners = set()
            for fact in state:
                 parts = get_parts(fact)
                 if parts and parts[0] in ['usable'] and len(parts) == 2:
                     all_spanners.add(parts[1])
                 elif parts and parts[0] == 'carrying' and len(parts) == 3:
                      all_spanners.add(parts[2]) # The carried object is a spanner

            # Find the object at a location that is not a nut or spanner
            for fact in state:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'at' and len(parts) == 3:
                     obj_name = parts[1]
                     if obj_name not in all_nuts and obj_name not in all_spanners:
                         man_name = obj_name
                         break

        # Final fallback if inference failed (e.g., empty state or unusual problem)
        if man_name is None:
             # print("Error: Could not identify the man object.")
             return math.inf # Cannot proceed without identifying the man


        # 1. Identify the man's current location.
        man_loc = None
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] == man_name:
                 man_loc = parts[2]
                 break

        if man_loc is None:
             # Man's location not found, state is likely invalid or goal reached
             # If goal is reached, h should be 0, handled below.
             # If state is truly invalid (e.g., man not at any location), returning inf is reasonable.
             # print(f"Warning: Man location not found for '{man_name}' in state.")
             return math.inf # Cannot proceed without man location


        # Identify nuts (objects that appear in 'tightened' goals or 'loose' state facts)
        all_nuts = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == 'tightened' and len(parts) == 2:
                 all_nuts.add(parts[1])
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] in ['loose', 'tightened'] and len(parts) == 2:
                 all_nuts.add(parts[1])


        # 2. Identify all loose nuts and their locations.
        loose_nuts = [] # List of (nut_name, nut_location)
        nut_locations_map = {} # {nut_name: nut_location}
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] in all_nuts:
                 nut_locations_map[parts[1]] = parts[2]

        for nut_name in all_nuts:
             if f'(loose {nut_name})' in state:
                 if nut_name in nut_locations_map:
                     loose_nuts.append((nut_name, nut_locations_map[nut_name]))
                 else:
                     # Loose nut location not found, invalid state?
                     # print(f"Warning: Location not found for loose nut {nut_name} in state.")
                     return math.inf # Cannot proceed


        # If no loose nuts, goal is reached.
        if not loose_nuts:
            return 0

        # Identify spanners (objects that appear in 'usable' or 'carrying' state facts)
        all_spanners = set()
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] in ['usable'] and len(parts) == 2:
                 all_spanners.add(parts[1])
             elif parts and parts[0] == 'carrying' and len(parts) == 3:
                  all_spanners.add(parts[2]) # The carried object is a spanner


        # 3. Identify all usable spanners currently carried by the man.
        carried_usable_spanners = [] # List of spanner_name
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'carrying' and len(parts) == 3 and parts[1] == man_name:
                spanner_name = parts[2]
                if f'(usable {spanner_name})' in state:
                    carried_usable_spanners.append(spanner_name)

        # 4. Identify all usable spanners available at locations.
        available_spanners_at_locs = [] # List of (spanner_name, spanner_location)
        spanner_locations_map = {} # {spanner_name: spanner_location}
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] in all_spanners:
                 spanner_locations_map[parts[1]] = parts[2]

        for spanner_name in all_spanners:
             if f'(usable {spanner_name})' in state and spanner_name not in carried_usable_spanners:
                 if spanner_name in spanner_locations_map:
                     available_spanners_at_locs.append((spanner_name, spanner_locations_map[spanner_name]))
                 # else: Usable spanner not carried and not at a location.
                 # This might happen if it was just used in the previous step and is now unusable.
                 # We only care about *usable* spanners at locations here.


        # --- Heuristic Calculation ---
        heuristic = 0
        current_man_loc = man_loc
        num_spanners_carried = len(carried_usable_spanners)
        # available_spanners_at_locs is already a list of (s, s_loc)

        # Sort loose nuts by distance from initial man location (heuristic choice)
        # Use math.inf for unreachable locations in sorting
        loose_nuts.sort(key=lambda item: self.get_distance(man_loc, item[1]))

        # Create a mutable list of available spanners at locations
        current_available_spanners_at_locs = list(available_spanners_at_locs)


        for (nut_name, nut_loc) in loose_nuts:
            # 1. Get man to nut_loc
            dist_to_nut = self.get_distance(current_man_loc, nut_loc)
            if dist_to_nut == math.inf:
                # Cannot reach this nut location from current position
                return math.inf
            heuristic += dist_to_nut
            current_man_loc = nut_loc # Man is now at the nut location

            # 2. Get spanner if needed
            if num_spanners_carried > 0:
                num_spanners_carried -= 1
                # Spanner is now carried and used for this nut. No extra travel/pickup cost.
            else:
                # Need to pick up a spanner
                if not current_available_spanners_at_locs:
                    # No usable spanners left anywhere
                    return math.inf

                # Find closest available spanner from current_man_loc (which is nut_loc)
                closest_spanner_info = None
                min_dist_to_spanner = math.inf

                # Filter for reachable spanners first and find the min distance
                reachable_spanners_with_dist = []
                for (s, s_loc) in current_available_spanners_at_locs:
                    dist = self.get_distance(current_man_loc, s_loc)
                    if dist != math.inf:
                        reachable_spanners_with_dist.append(((s, s_loc), dist))

                if not reachable_spanners_with_dist:
                     # No reachable usable spanner from current location
                     return math.inf

                # Find the spanner info corresponding to the minimum distance
                closest_spanner_info_with_dist = min(reachable_spanners_with_dist, key=lambda item: item[1])
                closest_spanner_info = closest_spanner_info_with_dist[0]
                min_dist_to_spanner = closest_spanner_info_with_dist[1]


                (s, s_loc) = closest_spanner_info
                current_available_spanners_at_locs.remove(closest_spanner_info) # Mark as used

                heuristic += min_dist_to_spanner # Travel to spanner
                current_man_loc = s_loc # Man is now at spanner location
                heuristic += 1 # Pickup action

                # Man is now at s_loc carrying a spanner. He needs to go back to nut_loc to use it.
                dist_back_to_nut = self.get_distance(current_man_loc, nut_loc)
                if dist_back_to_nut == math.inf:
                     # Cannot get back to nut location after picking up spanner
                     return math.inf

                heuristic += dist_back_to_nut
                current_man_loc = nut_loc # Man is back at nut location

            # 3. Tighten nut
            heuristic += 1 # tighten_nut action

        return heuristic
