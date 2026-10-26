from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available from heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If the base class is not automatically available, define a dummy one like this:
class Heuristic:
    def __init__(self, task):
        self.task = task # Store task
        self.goals = task.goals
        self.static = task.static
        self.adj = {}
        self.distances = {}
        # Call the overridden method to precompute distances
        # Pass initial_state explicitly if the base class doesn't make task available
        # or if the base class signature for _precompute_distances is different.
        # Assuming self.task is available after super().__init__ and _precompute_distances
        # is called with task.static.
        self._precompute_distances(task.static)

    def __call__(self, node):
        raise NotImplementedError

    def _precompute_distances(self, static_facts):
        # This method will be overridden in the domain-specific heuristic
        pass


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
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the travel cost for the man to reach spanners and nuts,
    the cost of picking up spanners, and the cost of tightening nuts.
    It greedily assigns available usable spanners to loose goal nuts,
    prioritizing nuts based on their initial distance from the man.

    # Assumptions
    - There is exactly one man, named 'bob'.
    - Spanners are objects whose names start with 'spanner'.
    - Nuts are objects whose names start with 'nut'.
    - Locations are other objects mentioned in 'at' or 'link' facts.
    - Each tighten action consumes one usable spanner.
    - The graph of locations connected by 'link' predicates is undirected.
    - Solvable problems have enough usable spanners.

    # Heuristic Initialization
    - Parse static facts and initial state to identify all locations and build the graph of locations connected by 'link' predicates.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify goal nuts from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the man's current location, loose nuts, usable spanners and their locations,
       and check if the man is carrying a usable spanner from the current state.
    2. Filter loose nuts to keep only those that are goal nuts.
    3. If there are no loose goal nuts, the heuristic is 0.
    4. Check if the total number of available usable spanners (including the one the man might be carrying)
       is less than the number of loose goal nuts. If so, return a large value (unsolvable).
    5. Sort the loose goal nuts based on the shortest path distance from the man's initial location
       to the nut's location. This determines the order in which nuts are processed by the greedy strategy.
    6. Initialize total heuristic cost to 0.
    7. Initialize the man's current location for the greedy simulation to his actual current location.
    8. Initialize the list of available usable spanners for the greedy simulation.
    9. Track whether the man currently "possesses" a usable spanner in the greedy simulation
        (initially true if he was carrying one in the state).
    10. Iterate through the sorted loose goal nuts:
        a. For the current nut at its location, calculate the cost to tighten it.
        b. If the man currently possesses a usable spanner:
           - The cost is the travel distance from his current location to the nut's location plus 1 (for the tighten action).
           - The man uses the spanner; he no longer possesses one for the next nut.
        c. If the man does not possess a usable spanner:
           - He must go get one. Find the available usable spanner (from the remaining set)
             whose location minimizes the total travel distance: distance from the man's current location
             to the spanner's location, plus distance from the spanner's location to the nut's location.
           - The cost is this minimum travel distance plus 1 (pickup) plus 1 (tighten).
           - The chosen spanner is removed from the list of available spanners.
           - The man still does not possess a spanner after tightening (it's consumed).
        d. Add the calculated cost for this nut to the total heuristic cost.
        e. Update the man's current location for the greedy simulation to the location of the nut just tightened.
    11. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and precomputing distances.
        """
        # Call base class __init__ first. It stores task and calls self._precompute_distances
        super().__init__(task)

        # Identify goal nuts from the task goals
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened":
                self.goal_nuts.add(parts[1])


    def _precompute_distances(self, static_facts):
        """
        Build adjacency list and compute all-pairs shortest paths.
        Uses static_facts and initial_state (from self.task).
        """
        self.adj = {}
        locations = set()

        # Build adjacency list from link facts
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.adj: self.adj[loc1] = []
                if loc2 not in self.adj: self.adj[loc2] = []
                self.adj[loc1].append(loc2)
                self.adj[loc2].append(loc1) # Links are bidirectional

        # Add locations from initial state 'at' facts
        # Assuming self.task is available here after super().__init__
        if hasattr(self, 'task') and hasattr(self.task, 'initial_state'):
             for fact in self.task.initial_state:
                  if match(fact, "at", "*", "*"):
                      _, obj, loc = get_parts(fact)
                      locations.add(loc)
                      if loc not in self.adj: self.adj[loc] = [] # Ensure all locations are keys in adj

        self.locations = list(locations) # Store list of all locations
        self.distances = {}

        # Compute distances using BFS from each location
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Perform BFS to find distances from start_loc to all other locations."""
        distances = {loc: float('inf') for loc in self.locations}
        if start_loc not in distances:
             # Start location might not be in the collected locations
             return distances # Return empty/inf distances if start is unknown

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            curr_loc = queue.popleft()

            if curr_loc in self.adj: # Handle locations with no links
                for neighbor in self.adj[curr_loc]:
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[curr_loc] + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract relevant information from the current state ---
        man_loc = None
        man_obj = None
        loose_nuts_in_state = set()
        usable_spanners_in_state = set()
        spanner_loc_map = {} # spanner -> location
        nut_loc_map = {} # nut -> location
        man_carrying_spanner_obj = None # spanner object if carrying

        # First pass to find man_obj and locations
        for fact in state:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == "at":
                 obj, loc = parts[1], parts[2]
                 if obj.startswith("bob"): # Assuming 'bob' is the man
                     man_obj, man_loc = obj, loc
                 elif obj.startswith("spanner"):
                     spanner_loc_map[obj] = loc
                 elif obj.startswith("nut"):
                     nut_loc_map[obj] = loc

        # Second pass for other predicates after identifying man_obj
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "loose":
                loose_nuts_in_state.add(parts[1])
            elif pred == "usable":
                usable_spanners_in_state.add(parts[1])
            elif man_obj is not None and pred == "carrying" and parts[1] == man_obj:
                 carried_spanner = parts[2]
                 man_carrying_spanner_obj = carried_spanner # Store the object name

        # Check if the carried spanner is usable
        man_initially_carrying_usable = man_carrying_spanner_obj is not None and man_carrying_spanner_obj in usable_spanners_in_state


        # Identify loose goal nuts and usable spanners with locations
        loose_goal_nuts_list = [] # list of (nut, nut_loc)
        for nut in self.goal_nuts:
            if nut in loose_nuts_in_state:
                 # Nut must have a location in the state
                 if nut in nut_loc_map:
                    loose_goal_nuts_list.append((nut, nut_loc_map[nut]))
                 # else: nut is loose and a goal but has no location? Should not happen in valid states.

        available_spanners_list = [] # list of (spanner, spanner_loc)
        for spanner in usable_spanners_in_state:
            if spanner in spanner_loc_map:
                available_spanners_list.append((spanner, spanner_loc_map[spanner]))

        # --- Heuristic Calculation ---

        num_loose_goal_nuts = len(loose_goal_nuts_list)

        # Base case: Goal reached for all nuts
        if num_loose_goal_nuts == 0:
            return 0

        # Check solvability assumption (enough spanners)
        num_available_spanners = len(available_spanners_list) + (1 if man_initially_carrying_usable else 0)
        if num_available_spanners < num_loose_goal_nuts:
             # Problem is likely unsolvable with current usable spanners
             return float('inf') # Or a sufficiently large integer

        # Handle case where man_loc is not in our distance map (e.g., isolated location)
        if man_loc not in self.distances:
             return float('inf')

        # Sort loose goal nuts by initial distance from man's location
        # This is a heuristic ordering for the greedy assignment
        try:
            loose_goal_nuts_list.sort(key=lambda nut_info: self.distances[man_loc].get(nut_info[1], float('inf')))
        except KeyError:
             # man_loc was not in self.distances, already handled above.
             # This catch might be redundant but safe.
             return float('inf')


        total_cost = 0
        current_man_loc = man_loc
        spanners_available_for_greedy = list(available_spanners_list) # Copy for greedy simulation
        man_has_spanner_now = man_initially_carrying_usable

        for nut, nut_loc in loose_goal_nuts_list:
            # Cost to tighten this nut
            cost_for_this_nut = 0

            if man_has_spanner_now:
                # Man already has a spanner, just need to travel to nut and tighten
                travel_cost = self.distances[current_man_loc].get(nut_loc, float('inf'))
                if travel_cost == float('inf'): return float('inf') # Path not found

                cost_for_this_nut = travel_cost + 1 # travel + tighten
                man_has_spanner_now = False # Spanner is used up

            else:
                # Man needs to get a spanner first
                min_spanner_travel_cost = float('inf')
                best_spanner_info = None # (S, L_spanner)

                # Find the best available spanner to minimize travel (man -> spanner -> nut)
                # Need to handle case where spanners_available_for_greedy is empty
                if not spanners_available_for_greedy:
                     # This should have been caught by the num_available_spanners check,
                     # but as a safeguard:
                     return float('inf')

                for spanner_info in spanners_available_for_greedy:
                    S, L_spanner = spanner_info
                    # Ensure locations are valid start nodes for BFS and path exists
                    if L_spanner not in self.distances or nut_loc not in self.distances:
                         continue # Cannot calculate path

                    travel_to_spanner = self.distances[current_man_loc].get(L_spanner, float('inf'))
                    travel_spanner_to_nut = self.distances[L_spanner].get(nut_loc, float('inf'))

                    if travel_to_spanner == float('inf') or travel_spanner_to_nut == float('inf'):
                         continue # Path through this spanner location not found

                    travel_cost = travel_to_spanner + travel_spanner_to_nut

                    if travel_cost < min_spanner_travel_cost:
                        min_spanner_travel_cost = travel_cost
                        best_spanner_info = spanner_info

                if best_spanner_info is None or min_spanner_travel_cost == float('inf'):
                     # No reachable usable spanner available for this nut
                     return float('inf')

                # Cost = travel to spanner + pickup + travel to nut + tighten
                cost_for_this_nut = min_spanner_travel_cost + 1 + 1

                # Remove the used spanner from the available list for subsequent nuts
                spanners_available_for_greedy.remove(best_spanner_info)

                man_has_spanner_now = False # Just used a spanner

            total_cost += cost_for_this_nut
            # Update man's location for the next nut in the greedy sequence
            current_man_loc = nut_loc

        return total_cost
