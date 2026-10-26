import collections
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    This heuristic estimates the cost to reach the goal state (all goal nuts
    tightened) by simulating a greedy plan. The plan prioritizes using spanners
    the man is already carrying for the closest loose nuts. Then, for the
    remaining loose nuts, it estimates the cost of picking up available usable
    spanners (starting with the closest ones) and taking them to the nut
    locations. The total heuristic value is the sum of estimated walk, pickup,
    and tighten actions along this simulated greedy path.

    Assumptions:
    - Links between locations are bidirectional.
    - Nut locations are fixed throughout the plan (their initial location is used).
    - There is exactly one man object, and its name is 'bob' (based on example instances).
    - Spanner objects can be identified by names starting with 'spanner'.
    - The location graph formed by links and initial object positions is connected,
      or unreachable locations/objects imply infinite cost.

    Heuristic Initialization:
    In the constructor (__init__), the heuristic precomputes static information:
    1.  It builds a graph of locations based on the 'link' facts.
    2.  It computes all-pairs shortest path distances between all relevant locations
        (locations from links and initial object positions) using BFS.
    3.  It identifies the set of goal nuts from the task goals.
    4.  It stores the initial locations of the goal nuts from the task's initial state.
    5.  It identifies all spanner objects present in the initial state.
    6.  It identifies the man object ('bob').

    Step-By-Step Thinking for Computing Heuristic:
    For a given state (in the __call__ method), the heuristic is computed as follows:
    1.  Find the man's current location in the state. If the man's location is unknown,
        return infinity.
    2.  Identify which of the goal nuts are still in the 'loose' state.
    3.  If there are no loose goal nuts, the goal is reached, return 0.
    4.  Identify which usable spanners the man is currently carrying.
    5.  Identify which usable spanners are available at locations (not carried by the man).
    6.  Check if the total number of usable spanners (carried + available) is less
        than the number of loose nuts. If so, the problem is unsolvable from this
        state (as each tighten action consumes a spanner), return infinity.
    7.  Initialize the total heuristic value `h = 0` and set the `current_location`
        for the greedy simulation to the man's current location.
    8.  **Phase 1: Use carried spanners.**
        Sort the loose nuts by their distance from the man's current location.
        Iterate through the first `N_usable_carried` nuts in this sorted list
        (where `N_usable_carried` is the number of usable spanners the man is carrying).
        For each of these nuts:
        -   Add the shortest distance from the `current_location` to the nut's location
            plus 1 (for the 'tighten_nut' action) to `h`.
        -   Update `current_location` to the location of the nut just tightened.
        -   Remove this nut from the list of nuts still needing to be tightened.
    9.  **Phase 2: Use available spanners.**
        Sort the remaining available usable spanners by their distance from the
        man's *current* location (updated in Phase 1).
        Sort the remaining loose nuts by their distance from the man's *current*
        location.
        Iterate through the remaining loose nuts, pairing the i-th remaining nut
        with the i-th closest available usable spanner. For each pair (nut, spanner):
        -   Get the spanner's location.
        -   Add the shortest distance from the `current_location` to the spanner's
            location plus 1 (for the 'pickup_spanner' action) plus the shortest
            distance from the spanner's location to the nut's location plus 1
            (for the 'tighten_nut' action) to `h`.
        -   Update `current_location` to the location of the nut just tightened.
    10. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state # Store initial state to get initial nut locations
        self.static_facts = task.static

        # --- Heuristic Initialization ---
        # 1. Build location graph
        self.location_graph = {}
        locations_from_links = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                locations_from_links.add(l1)
                locations_from_links.add(l2)
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Links are bidirectional

        # Include any locations mentioned in initial state that might not be in links
        all_possible_locations = set(locations_from_links)
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                  obj, loc = parts[1], parts[2]
                  all_possible_locations.add(loc)

        self.all_locations = list(all_possible_locations)

        # 2. Compute all-pairs shortest paths
        self.distances = self._compute_all_pairs_shortest_paths()

        # 3. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])

        # 4. Store initial locations of goal nuts
        self.initial_nut_locations = {}
        for fact in self.initial_state:
            parts = get_parts(fact)
            # Assuming goal nuts are locatable and their initial position is fixed
            if parts[0] == 'at' and parts[1] in self.goal_nuts:
                 self.initial_nut_locations[parts[1]] = parts[2]

        # 5. Identify all spanner objects
        self.all_spanners = set()
        for fact in self.initial_state:
            parts = get_parts(fact)
            # Assuming spanners are locatable and their names start with 'spanner'
            if parts[0] == 'at' and parts[1].startswith('spanner'):
                self.all_spanners.add(parts[1])
            # Also check if spanners are carried in the initial state
            elif parts[0] == 'carrying' and parts[2].startswith('spanner'):
                 self.all_spanners.add(parts[2])

        # 6. Identify the man object (assuming it's 'bob' based on examples)
        self.man_object = 'bob' # Hardcoded based on example instances


    def _bfs(self, start_location):
        """Performs BFS to find shortest distances from a start location."""
        distances = {loc: float('inf') for loc in self.all_locations}
        if start_location in distances: # Check if start_location is one of the known locations
            distances[start_location] = 0
            queue = collections.deque([start_location])
            while queue:
                current_loc = queue.popleft()
                current_dist = distances[current_loc]
                if current_loc in self.location_graph: # Check if current_loc has neighbors defined by links
                    for neighbor in self.location_graph[current_loc]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = current_dist + 1
                            queue.append(neighbor)
        return distances

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest distances between all pairs of locations."""
        all_distances = {}
        for start_loc in self.all_locations:
            all_distances[start_loc] = self._bfs(start_loc)
        return all_distances

    def get_distance(self, loc1, loc2):
        """Returns the precomputed shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # This can happen if a location is unreachable from another.
             # Treat unreachable locations as infinite distance.
             return float('inf')
        return self.distances[loc1][loc2]

    def get_spanner_location(self, spanner, state, man_location):
        """Finds the current location of a spanner (carried or at a location)."""
        # Check if carried by the man
        if '(carrying ' + self.man_object + ' ' + spanner + ')' in state:
            return man_location
        # Check if at a location
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == spanner:
                return parts[2]
        return None # Should not happen for spanners that exist and are locatable/carried

    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            An estimate of the remaining cost to reach the goal.
        """
        state = node.state

        # --- Step-By-Step Thinking for Computing Heuristic ---

        # 1. Find man's current location
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.man_object:
                man_location = parts[2]
                break
        if man_location is None:
             # Man is not at any location? Should not happen in valid states.
             return float('inf') # Return infinity if man's location is unknown.


        # 2. Identify loose goal nuts in the current state
        loose_goal_nuts = {nut for nut in self.goal_nuts if '(loose ' + nut + ')' in state}
        N_loose = len(loose_goal_nuts)

        # 3. Goal reached if no loose goal nuts
        if N_loose == 0:
            return 0

        # 4. Identify usable spanners carried by the man
        carried_spanners = {s for s in self.all_spanners if '(carrying ' + self.man_object + ' ' + s + ')' in state}
        usable_carried_spanners = {s for s in carried_spanners if '(usable ' + s + ')' in state}
        N_usable_carried = len(usable_carried_spanners)

        # 5. Identify usable spanners available at locations (not carried)
        available_usable_spanners = [
            s for s in self.all_spanners
            if s not in carried_spanners and '(usable ' + s + ')' in state
        ]
        N_usable_available = len(available_usable_spanners)

        # 6. If the number of loose nuts exceeds the total number of usable spanners, return infinity
        if N_loose > N_usable_carried + N_usable_available:
            return float('inf')

        # 7. Initialize heuristic value and current location for greedy simulation
        h = 0
        current_location = man_location

        # 8. Get list of loose nuts
        nuts_to_tighten = list(loose_goal_nuts)

        # 9. Phase 1: Use carried spanners for the closest loose nuts
        # Sort nuts by distance from the man's *initial* location in this step
        nuts_for_carried = sorted(nuts_to_tighten, key=lambda nut: self.get_distance(current_location, self.initial_nut_locations[nut]))[:N_usable_carried]

        for nut in nuts_for_carried:
            nut_loc = self.initial_nut_locations[nut]
            dist_to_nut = self.get_distance(current_location, nut_loc)
            if dist_to_nut == float('inf'): return float('inf') # Unreachable nut
            h += dist_to_nut + 1 # Walk to nut + tighten
            current_location = nut_loc # Update location after tightening
            nuts_to_tighten.remove(nut) # Remove from the list for phase 2

        # 10. Phase 2: Use available spanners for the remaining loose nuts
        # Sort remaining available usable spanners by distance from the *current* location
        available_usable_spanners.sort(key=lambda spanner: self.get_distance(current_location, self.get_spanner_location(spanner, state, man_location)))

        # Sort remaining loose nuts by distance from the *current* location
        nuts_to_tighten.sort(key=lambda nut: self.get_distance(current_location, self.initial_nut_locations[nut]))

        # Pair remaining nuts with available spanners greedily
        num_remaining_nuts = len(nuts_to_tighten)
        # num_available_spanners = len(available_usable_spanners) # Already checked total count earlier

        for i in range(num_remaining_nuts):
            nut = nuts_to_tighten[i]
            spanner = available_usable_spanners[i] # Get the i-th closest available spanner
            spanner_loc = self.get_spanner_location(spanner, state, man_location)
            nut_loc = self.initial_nut_locations[nut]

            dist_to_spanner = self.get_distance(current_location, spanner_loc)
            if dist_to_spanner == float('inf'): return float('inf') # Unreachable spanner

            dist_spanner_to_nut = self.get_distance(spanner_loc, nut_loc)
            if dist_spanner_to_nut == float('inf'): return float('inf') # Unreachable nut from spanner loc

            # Cost to walk to spanner + pickup + walk to nut + tighten
            h += dist_to_spanner + 1 + dist_spanner_to_nut + 1
            current_location = nut_loc # Update location after tightening

        # 11. Return the total heuristic value
        return h
