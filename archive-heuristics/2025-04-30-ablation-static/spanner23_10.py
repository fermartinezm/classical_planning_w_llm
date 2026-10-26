from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections # Used for BFS queue

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
    # Ensure the number of parts matches the number of args for a stricter match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose goal nuts. It does this by calculating the cost for each loose
    goal nut sequentially, assuming the man processes them one by one.
    For each nut, it estimates the travel cost for the man to reach the nut's
    location and the cost to acquire a usable spanner if he doesn't currently
    carry one.

    # Assumptions:
    - There is only one man object, assumed to be the object whose type is 'man'.
    - Nut objects have type 'nut', spanner objects have type 'spanner', locations have type 'location'.
      (Inference based on common PDDL structure, falls back to naming convention if types aren't easily accessible).
    - Links between locations are bidirectional.
    - The graph of locations is connected (or solvable instances are within a connected component).
    - A spanner becomes unusable after one 'tighten_nut' action.
    - The man can only carry one spanner at a time (implied by the domain predicates).

    # Heuristic Initialization
    - Infers object types (man, nuts, spanners, locations) from the initial state, goals, and static facts.
    - Builds the location graph based on 'link' predicates.
    - Precomputes all-pairs shortest path distances between locations using BFS.
    - Identifies the set of nuts that need to be tightened (goal nuts).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the cost by considering the loose goal nuts one by one.
    The nuts are processed in increasing order of their distance from the man's
    initial location in the current state. This provides a fixed, simple order
    for the sequential calculation.

    For each loose goal nut `N` at location `L_N` in the sorted sequence:
    1.  **Travel to Nut & Spanner Acquisition:** The man needs to reach `L_N`
        while carrying a usable spanner.
        *   If the man is currently carrying a usable spanner (`spanner_is_usable_and_carried` is True):
            The cost added for this step is the travel distance from the man's
            current location (`current_man_loc`) to `L_N`. The carried spanner
            is then considered used for this nut, so `spanner_is_usable_and_carried`
            becomes False for the next nut.
        *   If the man is NOT currently carrying a usable spanner: He must first
            acquire one. Find the closest available usable spanner `S` at location `L_S`
            (among those not yet 'used' by the heuristic calculation for previous nuts).
            The cost added for this step is estimated as:
            `distance(current_man_loc, L_S)` (travel to spanner) + 1 (pickup action) +
            `distance(L_S, L_N)` (travel from spanner location to nut location).
            The spanner `S` is then considered used and removed from the available list.
            The man is now carrying an unusable spanner, so `spanner_is_usable_and_carried`
            remains False for the next nut.
    2.  **Tighten Action:** Add 1 to the cost for the `tighten_nut` action.
    3.  **Update Man's Location:** The man is assumed to end up at location `L_N`
        after tightening the nut. Update `current_man_loc` to `L_N` for the next nut.
    4.  **Sum Costs:** The total heuristic value is the sum of the costs calculated
        for each loose goal nut in the sorted sequence.

    If there are no loose goal nuts, the heuristic is 0. If at any point a usable
    spanner is needed but none are available, the heuristic returns infinity,
    indicating an unsolvable state (within the context of this heuristic's assumptions).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting domain information and precomputing distances.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Infer object types (basic inference based on predicates and naming conventions)
        self.man_name = None
        self.nut_names = set()
        self.spanner_names = set()
        self.location_names = set()

        # A more robust way to get object types would be parsing the :objects section
        # or using a PDDL parser that provides this structure.
        # Falling back to inference from predicates/names found in facts.
        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at':
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # Infer types based on common predicates they appear in
                    # This is not foolproof but works for typical domains like spanner
                    is_locatable = any(match(f, p, obj, '*') for f in all_facts for p in ['at', 'carrying'])
                    is_location = any(match(f, p, '*', obj) for f in all_facts for p in ['at', 'link']) or \
                                  any(match(f, p, obj, '*') for f in all_facts for p in ['link'])

                    if match(fact, 'at', obj, loc):
                        if any(match(f, 'carrying', obj, '*') for f in all_facts): self.man_name = obj
                        elif any(match(f, 'usable', obj) for f in all_facts) or any(match(f, 'carrying', '*', obj) for f in all_facts): self.spanner_names.add(obj)
                        elif any(match(f, 'tightened', obj) for f in all_facts) or any(match(f, 'loose', obj) for f in all_facts): self.nut_names.add(obj)
                        elif is_location: self.location_names.add(obj) # If it's not locatable but is a location
                        else: self.location_names.add(obj) # Assume it's a location if not identified as locatable type

                    if is_location: self.location_names.add(loc) # The second argument of 'at' is always a location

            elif predicate == 'carrying':
                 if len(parts) == 3:
                    carrier, spanner = parts[1], parts[2]
                    self.man_name = carrier # Assume the carrier is the man
                    self.spanner_names.add(spanner)
            elif predicate == 'usable':
                 if len(parts) == 2:
                    self.spanner_names.add(parts[1])
            elif predicate == 'tightened' or predicate == 'loose':
                 if len(parts) == 2:
                    self.nut_names.add(parts[1])
            elif predicate == 'link':
                 if len(parts) == 3:
                    self.location_names.add(parts[1])
                    self.location_names.add(parts[2])

        # Ensure man_name is set, fallback to 'bob' if not found by inference
        if self.man_name is None:
             # Check if 'bob' exists in any fact parts
             for fact in all_facts:
                 if 'bob' in get_parts(fact):
                     self.man_name = 'bob'
                     break
        # If still None, the domain might not have a man or naming is different.
        # This heuristic assumes a single man object exists.

        # Build location graph
        self.location_graph = {loc: set() for loc in self.location_names}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                if l1 in self.location_graph and l2 in self.location_graph:
                  self.location_graph[l1].add(l2)
                  self.location_graph[l2].add(l1)

        # Precompute all-pairs shortest path distances
        self.distances = {}
        for start_loc in self.location_names:
            self.distances[start_loc] = self._calculate_all_distances_from(start_loc)

        # Store goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'tightened' and len(parts) == 2 and parts[1] in self.nut_names:
                self.goal_nuts.add(parts[1])

    def _calculate_all_distances_from(self, start_loc):
        """
        Calculates shortest path distances from a start location to all other locations
        using BFS.
        """
        distances = {loc: float('inf') for loc in self.location_names}
        if start_loc not in self.location_names:
             # Start location is not in the known graph
             return distances

        distances[start_loc] = 0
        queue = collections.deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to tighten
        all loose goal nuts.
        """
        state = node.state

        # Extract relevant state information
        man_loc = None
        carried_spanner = None
        usable_carried = False
        nut_locations = {}
        spanner_locations = {}
        usable_spanners_in_state = set() # Set of usable spanner names currently in state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at':
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    if obj == self.man_name: man_loc = loc
                    elif obj in self.nut_names: nut_locations[obj] = loc
                    elif obj in self.spanner_names: spanner_locations[obj] = loc
            elif predicate == 'carrying':
                if len(parts) == 3:
                    carrier, spanner = parts[1], parts[2]
                    if carrier == self.man_name and spanner in self.spanner_names:
                        carried_spanner = spanner
            elif predicate == 'usable':
                if len(parts) == 2:
                    spanner = parts[1]
                    if spanner in self.spanner_names:
                        usable_spanners_in_state.add(spanner)

        if carried_spanner and carried_spanner in usable_spanners_in_state:
            usable_carried = True

        # Identify loose goal nuts in the current state
        loose_goal_nuts_in_state = {n for n in self.goal_nuts if '(loose ' + n + ')' in state}

        if not loose_goal_nuts_in_state:
            return 0 # Goal reached

        # --- Heuristic Calculation ---
        h = 0
        current_man_loc = man_loc
        spanner_is_usable_and_carried = usable_carried

        # Create a copy of usable spanners available at locations (not carried)
        available_usable_spanners_at_loc_copy = {}
        for spanner in usable_spanners_in_state:
            if spanner != carried_spanner:
                loc = spanner_locations.get(spanner)
                if loc and loc in self.location_names: # Ensure location is valid
                    available_usable_spanners_at_loc_copy.setdefault(loc, []).append(spanner)

        # Sort loose goal nuts by distance from the man's initial location in this state
        # This provides a fixed, simple order for the sequential calculation.
        loose_nuts_list = list(loose_goal_nuts_in_state)

        # Ensure man_loc is valid and distances from man_loc are precomputed
        if man_loc is None or man_loc not in self.distances:
             # This state is likely unreachable or invalid according to the graph
             return float('inf')

        # Sort nuts based on distance from the man's location in the current state
        loose_nuts_list.sort(key=lambda n: self.distances[man_loc].get(nut_locations.get(n), float('inf')))


        for nut in loose_nuts_list:
            nut_loc = nut_locations.get(nut)
            # If nut_loc is None or not in distances, it's an invalid state for this heuristic
            if nut_loc is None or nut_loc not in self.distances[current_man_loc]:
                 return float('inf')

            cost_for_this_nut = 0

            if spanner_is_usable_and_carried:
                # Man has a usable spanner. Travel from current_man_loc to nut_loc, tighten.
                travel_cost = self.distances[current_man_loc][nut_loc]
                cost_for_this_nut = travel_cost + 1 # Travel + Tighten
                spanner_is_usable_and_carried = False # Spanner used, becomes unusable

            else:
                # Man needs to get a spanner first.
                closest_spanner_loc = None
                min_dist_to_spanner = float('inf')
                spanner_to_use = None

                # Find the closest location with an available usable spanner from current_man_loc
                locations_with_spanners = [loc for loc, spanners in available_usable_spanners_at_loc_copy.items() if spanners]

                if not locations_with_spanners:
                    # No usable spanners left to pick up for this nut
                    return float('inf') # Unsolvable from this state

                # Sort locations with spanners by distance from current_man_loc
                locations_with_spanners.sort(key=lambda loc: self.distances[current_man_loc].get(loc, float('inf')))

                # The closest location with an available spanner is the first one
                closest_spanner_loc = locations_with_spanners[0]
                min_dist_to_spanner = self.distances[current_man_loc].get(closest_spanner_loc, float('inf'))

                # If the closest spanner location is unreachable, this path is blocked
                if min_dist_to_spanner == float('inf'):
                     return float('inf')

                spanner_to_use = available_usable_spanners_at_loc_copy[closest_spanner_loc][0] # Take any spanner at this location


                # Cost = Travel to spanner + Pickup + Travel from spanner to nut + Tighten
                travel_to_spanner_cost = min_dist_to_spanner
                pickup_cost = 1
                travel_spanner_to_nut_cost = self.distances[closest_spanner_loc].get(nut_loc, float('inf'))

                # If the nut location is unreachable from the spanner location, this path is blocked
                if travel_spanner_to_nut_cost == float('inf'):
                     return float('inf')

                tighten_cost = 1

                cost_for_this_nut = travel_to_spanner_cost + pickup_cost + travel_spanner_to_nut_cost + tighten_cost

                # Remove the used spanner from the available list
                available_usable_spanners_at_loc_copy[closest_spanner_loc].remove(spanner_to_use)
                # Man is now carrying the used spanner (unusable)
                spanner_is_usable_and_carried = False # Ensure this is false after using the spanner

            h += cost_for_this_nut
            current_man_loc = nut_loc # Man ends up at the nut location after tightening

        return h
