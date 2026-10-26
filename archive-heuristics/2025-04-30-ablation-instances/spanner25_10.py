from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the travel cost for the man to reach nut locations and spanner locations,
    the cost of picking up spanners, and the cost of tightening nuts. It uses a greedy
    approach for assigning spanners to nuts and ordering tasks.

    # Assumptions
    - Nuts are static objects at fixed locations.
    - Spanners are consumed after one use (become unusable).
    - The man can carry at most one spanner at a time. (Inferred from domain structure).
    - The problem is solvable (enough usable spanners exist initially).
    - The location graph defined by 'link' predicates is connected.

    # Heuristic Initialization
    - Builds the location graph from `link` predicates in static facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the static location of each nut that is a goal from the initial state.
    - Identifies the name of the man object by looking for objects involved in 'at' or 'carrying' predicates that are not identified as nuts or spanners.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all loose nuts in the current state that are specified as goals (`LooseGoalNuts`). If none, the goal is met for these nuts, and the heuristic is 0.
    2. Identify the man's current location (`ManLocation`).
    3. Identify usable spanners currently available in the state (`CurrentUsableSpanners`).
    4. Determine if the man is currently carrying a usable spanner (`CarriedSpanner`).
    5. Create a set of usable spanners available for pickup at locations (`AvailablePickupSpanners`).
    6. Initialize the heuristic cost `h = 0`.
    7. Initialize the man's current location for the greedy tour simulation (`current_man_loc = ManLocation`).
    8. Initialize the spanner in hand status (`spanner_in_hand = CarriedSpanner`). If the carried spanner is not usable, it's not considered 'in hand' for tightening purposes.
    9. Create a working set of usable spanners that can be picked up (`remaining_pickup_spanners`). If the man starts with a usable spanner, it's removed from this set as it's already accounted for.
    10. Sort the `LooseGoalNuts` by their distance from the man's *initial* location. This defines a greedy order for tackling the nuts.
    11. Iterate through the sorted `LooseGoalNuts`:
        a. Get the nut's static location `l_n`.
        b. If the man does not currently have a usable spanner in hand (`spanner_in_hand` is None):
            i. Find the closest usable spanner `s` among `remaining_pickup_spanners` (at location `l_s`) relative to the `current_man_loc`.
            ii. If no such spanner exists, the problem is likely unsolvable from this state with the available spanners; return infinity.
            iii. Add `dist(current_man_loc, l_s)` to `h` (cost to travel to the spanner).
            iv. Update `current_man_loc = l_s`.
            v. Add 1 to `h` (cost of the pickup action).
            vi. Set `spanner_in_hand = s`.
            vii. Remove `s` from `remaining_pickup_spanners` as it's now "used" (picked up).
        c. Add `dist(current_man_loc, l_n)` to `h` (cost to travel from the current location to the nut's location).
        d. Update `current_man_loc = l_n`.
        e. Add 1 to `h` (cost of the tighten action).
        f. Set `spanner_in_hand = None` (the spanner is used and becomes unusable).
    12. Return the total estimated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # Build location graph and compute distances
        self.location_graph = {}
        locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1) # Assuming links are bidirectional

        self.locations = list(locations)
        self.distances = self._compute_all_pairs_shortest_paths()

        # Identify initial spanners and nuts first
        initial_spanners_set = set()
        initial_nuts_set = set()

        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == "usable" and len(parts) == 2:
                initial_spanners_set.add(parts[1])
            elif parts[0] == "carrying" and len(parts) == 3:
                 initial_spanners_set.add(parts[2]) # The object being carried is a spanner

        # Refined nut identification (based on goal)
        initial_nuts_set = {get_parts(g)[1] for g in self.goals if match(g, "tightened", "*")}

        # Store nut locations based on initial state (assuming nuts are at fixed locations)
        self.NutLocations = {}
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in initial_nuts_set:
                     self.NutLocations[obj] = loc

        # Find the man object name
        self.man_name = None
        candidate_men = set()
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 candidate_men.add(get_parts(fact)[1])
             elif match(fact, "carrying", "*", "*"):
                 candidate_men.add(get_parts(fact)[1])

        for obj in candidate_men:
             if obj not in initial_nuts_set and obj not in initial_spanners_set:
                 self.man_name = obj
                 break # Assume only one man

        if self.man_name is None:
             # Fallback: If man wasn't found by predicate analysis, try common name 'bob'
             # This is a last resort and makes the heuristic fragile to problem variations.
             # A proper parser providing object types would be ideal.
             print(f"Warning: Could not definitively identify man object from initial state predicates. Candidates: {candidate_men}, Nuts: {initial_nuts_set}, Spanners: {initial_spanners_set}. Assuming 'bob'.")
             self.man_name = 'bob' # Based on example instance


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        for start_node in self.locations:
            distances[(start_node, start_node)] = 0
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_loc, dist = queue.popleft()

                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[(start_node, neighbor)] = dist + 1
                            queue.append((neighbor, dist + 1))
        return distances

    def dist(self, l1, l2):
        """Returns the shortest distance between two locations."""
        if (l1, l2) in self.distances:
            return self.distances[(l1, l2)]
        # If no path exists between locations in the graph, return infinity.
        # This indicates an unreachable location, making the goal impossible.
        return float('inf')

    def get_object_location(self, obj, state):
        """Finds the location of an object in the current state if it's 'at' a location."""
        for fact in state:
            if match(fact, "at", obj, "*"):
                return get_parts(fact)[2]
        return None # Object is not at any location (e.g., carried)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Identify loose nuts that are goals
        loose_goal_nuts = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "loose", "*") and f"(tightened {get_parts(fact)[1]})" in self.goals
        }

        if not loose_goal_nuts:
            return 0 # Goal reached for all relevant nuts

        # 2. Identify man's current location
        man_location = self.get_object_location(self.man_name, state)
        if man_location is None:
             # The man must always be at a location according to the domain.
             # If not found, the state representation might be incomplete or incorrect.
             # Return infinity as the state is likely invalid or leads to an unsolvable path.
             print(f"Error: Man {self.man_name} location not found in state.")
             return float('inf')


        # 3. Identify usable spanners currently available in the state
        current_usable_spanners = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "usable", "*")
        }

        # 4. Check if man is carrying a usable spanner
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                s = get_parts(fact)[2]
                if s in current_usable_spanners:
                    carried_spanner = s
                break # Assume man carries at most one spanner

        # 5. Usable spanners available for pickup at locations
        # Need to know their locations
        available_pickup_spanners_with_locs = {}
        for s in current_usable_spanners:
            if s != carried_spanner: # Don't include the one being carried
                 loc = self.get_object_location(s, state)
                 if loc is not None: # Spanner is at a location
                     available_pickup_spanners_with_locs[s] = loc

        # 6. Initialize heuristic cost
        h = 0

        # 7. Initialize current man location for the greedy tour
        current_man_loc = man_location

        # 8. Initialize spanner in hand status
        spanner_in_hand = carried_spanner # This spanner is available for the first task if usable

        # Keep track of usable spanners that can still be picked up
        # We use the keys from available_pickup_spanners_with_locs
        remaining_pickup_spanners = set(available_pickup_spanners_with_locs.keys())

        # 9. Sort loose goal nuts by distance from the man's initial location
        # Need initial man location for sorting key
        initial_man_location = None
        for fact in self.initial_state:
             if match(fact, "at", self.man_name, "*"):
                 initial_man_location = get_parts(fact)[2]
                 break
        # Fallback if initial location not found (shouldn't happen in valid problems)
        sort_location = initial_man_location if initial_man_location is not None else man_location

        # Ensure all loose goal nuts have a known location
        for nut in loose_goal_nuts:
            if nut not in self.NutLocations:
                 # This nut needs tightening but its location is unknown/not static.
                 # This contradicts the assumption nuts are static.
                 # Return infinity as the goal might be unreachable or the problem malformed.
                 print(f"Error: Location of goal nut {nut} not found in initial state.")
                 return float('inf')


        loose_goal_nuts_list = sorted(
            loose_goal_nuts,
            key=lambda n: self.dist(sort_location, self.NutLocations[n])
        )

        # 10. Iterate through the sorted loose goal nuts
        for nut in loose_goal_nuts_list:
            nut_location = self.NutLocations[nut]

            # Cost to get spanner if needed
            if spanner_in_hand is None:
                # Need to pick up a spanner. Find closest available usable spanner.
                closest_spanner_loc = None
                min_dist_spanner = float('inf')
                spanner_to_pickup = None

                # Find closest spanner among those not yet used/carried
                for s in remaining_pickup_spanners:
                    l_s = available_pickup_spanners_with_locs[s] # Get pre-calculated location
                    d = self.dist(current_man_loc, l_s)
                    if d < min_dist_spanner:
                        min_dist_spanner = d
                        closest_spanner_loc = l_s
                        spanner_to_pickup = s

                if closest_spanner_loc is None:
                     # We need to tighten this nut, but no more usable spanners
                     # are available for pickup. This state is likely unsolvable.
                     return float('inf')

                h += min_dist_spanner # Travel to spanner
                current_man_loc = closest_spanner_loc
                h += 1 # Pickup action
                spanner_in_hand = spanner_to_pickup
                remaining_pickup_spanners.discard(spanner_to_pickup) # This spanner is now "used" for pickup

            # Cost to travel from current location (where spanner was picked up, or man started with one) to nut location
            travel_to_nut_cost = self.dist(current_man_loc, nut_location)
            if travel_to_nut_cost == float('inf'):
                 # Nut location is unreachable from current location
                 return float('inf')
            h += travel_to_nut_cost
            current_man_loc = nut_location

            # Cost to tighten action
            h += 1
            spanner_in_hand = None # Spanner used for this nut

        # 11. Return the total estimated cost
        return h

