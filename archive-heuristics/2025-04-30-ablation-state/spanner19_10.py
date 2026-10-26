from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose
    goal nuts. It sums the estimated cost for each loose goal nut independently.
    The estimated cost for a single nut includes the tighten action itself,
    the travel cost for the man to reach the nut's location, and the cost
    to acquire a usable spanner if the man is not already carrying one.
    The spanner acquisition cost includes travel to the nearest usable spanner
    and the pickup action.

    # Assumptions:
    - The goal is to achieve the `(tightened ?nut)` predicate for a set of nuts.
    - Each `tighten_nut` action consumes the `usable` predicate of the spanner used.
    - A man can carry at most one spanner at a time.
    - There is no action to drop a spanner or make an unusable spanner usable again.
    - If a man is carrying an unusable spanner, he cannot pick up another.
    - The locations form a connected graph (within reachable areas).

    # Heuristic Initialization
    - Identify all locations and build a graph based on `link` facts.
    - Precompute all-pairs shortest path distances between locations using BFS.
    - Identify the man, all spanners, all nuts, and the set of goal nuts by inspecting initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all goal nuts that are currently loose and their locations.
    3. Identify all usable spanners that are currently at locations and their locations.
    4. Check if the man is currently carrying a usable spanner.
    5. Check for unsolvability:
       - If the number of loose goal nuts exceeds the total number of available usable spanners (carried or at locations), the problem is unsolvable from this state. Return a large value (1000000).
       - If the man is carrying an unusable spanner and there are loose goal nuts remaining, the problem is unsolvable from this state (as he cannot pick up another spanner with the standard single-spanner assumption). Return a large value (1000000).
    6. Initialize the total heuristic cost to 0.
    7. For each loose goal nut `N` at location `L_N`:
       a. Add 1 to the cost for the `tighten_nut` action itself.
       b. Calculate the minimum cost to get the man *with a usable spanner* to location `L_N`.
          - Option 1: Use the spanner the man is currently carrying (if usable). The cost is the shortest distance from the man's current location (`L_M`) to `L_N`.
          - Option 2: Pick up a usable spanner from a location. Find the minimum cost over all locations `L_S` where a usable spanner is currently available. The cost via `L_S` is the shortest distance from `L_M` to `L_S`, plus 1 for the `pickup_spanner` action, plus the shortest distance from `L_S` to `L_N`.
          - The minimum of Option 1 (if available) and Option 2 is the estimated travel/pickup cost for this nut. Add this minimum cost to the total cost for this nut.
       c. Add the total cost for this nut to the overall heuristic value.
    8. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal information."""
        self.goals = task.goals

        # Extract objects and locations by inspecting initial state and goals
        all_locations = set()
        all_spanners = set()
        all_nuts = set()
        the_man = None

        # Helper to process facts and identify objects/locations/types
        def identify_objects_from_fact(fact_str):
            nonlocal the_man # Allow modification of outer scope variable
            parts = get_parts(fact_str)
            pred = parts[0]
            args = parts[1:]

            if pred == 'link':
                all_locations.update(args)
            elif pred == 'at':
                if len(args) == 2:
                    obj, loc = args
                    all_locations.add(loc)
                    # Try to infer type based on common predicates in initial state/goals
                    # This is a heuristic for object typing based on common predicates
                    if any(match(f, 'carrying', obj, '*') for f in task.initial_state):
                        the_man = obj
                    # Check if obj is a spanner by looking for usable/carrying predicates involving it
                    if any(match(f, 'usable', obj) for f in task.initial_state) or any(match(f, 'carrying', '*', obj) for f in task.initial_state):
                         all_spanners.add(obj)
                    # Check if obj is a nut by looking for loose/tightened predicates involving it
                    if any(match(f, 'loose', obj) for f in task.initial_state) or any(match(f, 'tightened', obj) for f in task.goals):
                         all_nuts.add(obj)
            elif pred == 'carrying':
                 if len(args) == 2:
                      man_obj, spanner_obj = args
                      the_man = man_obj
                      all_spanners.add(spanner_obj)
            elif pred == 'usable':
                 if len(args) == 1:
                      all_spanners.add(args[0])
            elif pred == 'loose' or pred == 'tightened':
                 if len(args) == 1:
                      all_nuts.add(args[0])


        # Process initial state and static facts to find objects and locations
        for fact in task.initial_state | task.static:
            identify_objects_from_fact(fact)

        # Process goal facts to find goal nuts and goal locations (if any 'at' goals)
        for goal in self.goals:
            identify_objects_from_fact(goal) # Process potential objects/locations in goals

        self.the_man = the_man
        self.all_spanners = list(all_spanners)
        self.all_nuts = list(all_nuts)
        self.all_locations = list(all_locations)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])

        # Compute all-pairs shortest path distances
        self.distances = self.compute_distances(self.all_locations, task.static)

    def compute_distances(self, locations, static_facts):
        """
        Computes shortest path distances between all pairs of locations
        using BFS based on 'link' facts.
        """
        graph = {loc: [] for loc in locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                # Ensure locations are in our list before adding to graph
                if l1 in graph and l2 in graph:
                    graph[l1].append(l2)
                    graph[l2].append(l1)

        distances = {}
        # Handle empty locations list
        if not locations:
            return distances

        for start_loc in locations:
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            while queue:
                current_loc, d = queue.popleft()
                distances[(start_loc, current_loc)] = d

                # Ensure current_loc is a valid key in the graph
                if current_loc in graph:
                    for neighbor in graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))
        return distances

    def dist(self, l1, l2):
        """Returns the precomputed shortest distance between two locations."""
        # Return a large number if locations are not in our graph or unreachable
        # or if l1 or l2 are None (e.g., man_location not found)
        if l1 is None or l2 is None:
             return 1000000
        return self.distances.get((l1, l2), 1000000)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_location = None
        # Ensure self.the_man was identified in __init__
        if self.the_man:
            for fact in state:
                if match(fact, "at", self.the_man, "*"):
                    man_location = get_parts(fact)[2]
                    break
        if man_location is None:
             # Man is not at any location? Should not happen in valid states.
             # Or man object wasn't identified correctly in __init__
             return 1000000 # Indicate unsolvability

        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts = set()
        nut_locations = {}
        for nut in self.goal_nuts:
            if f"(loose {nut})" in state:
                loose_goal_nuts.add(nut)
                # Find location of this nut
                found_loc = False
                for fact in state:
                    if match(fact, "at", nut, "*"):
                        nut_locations[nut] = get_parts(fact)[2]
                        found_loc = True
                        break
                if not found_loc:
                     # Loose goal nut is not at any location? Should not happen.
                     return 1000000 # Indicate unsolvability


        num_loose_goal_nuts = len(loose_goal_nuts)

        if num_loose_goal_nuts == 0:
            return 0 # Goal reached

        # 3. Identify usable spanners at locations
        usable_spanner_locs = set() # Set of locations with usable spanners
        usable_spanners_in_state = set() # Set of usable spanner objects
        unusable_spanners_in_state = set(self.all_spanners) # Start with all, remove usable

        for spanner in self.all_spanners:
             if f"(usable {spanner})" in state:
                  usable_spanners_in_state.add(spanner)
                  unusable_spanners_in_state.discard(spanner) # Remove from unusable set
                  # Check if it's at a location
                  for fact in state:
                       if match(fact, "at", spanner, "*"):
                            loc = get_parts(fact)[2]
                            usable_spanner_locs.add(loc)
                            break # Assuming spanner is only at one location

        # 4. Check if man is carrying a usable spanner
        man_carrying_usable = False
        carried_spanner = None
        for spanner in self.all_spanners:
             if f"(carrying {self.the_man} {spanner})" in state:
                  carried_spanner = spanner
                  if spanner in usable_spanners_in_state:
                       man_carrying_usable = True
                  break # Assuming man carries at most one spanner

        # 5. Check for unsolvability
        num_available_usable_spanners = len(usable_spanner_locs) + (1 if man_carrying_usable else 0)

        if num_loose_goal_nuts > num_available_usable_spanners:
             return 1000000 # Not enough usable spanners in total

        man_carrying_unusable = (carried_spanner is not None) and (not man_carrying_usable)
        if man_carrying_unusable and num_loose_goal_nuts > 0:
             # If he carries an unusable spanner, he cannot pick up another (assuming single capacity)
             # and cannot use the one he has. If there are loose nuts, he's stuck.
             return 1000000 # Stuck with unusable spanner


        # 6. Calculate heuristic cost
        h = 0

        # Cost for each loose nut independently
        for nut in loose_goal_nuts:
            nut_location = nut_locations[nut]

            # Cost for this nut: tighten (1) + travel + pickup (if needed)
            cost_this_nut = 1 # tighten action

            # Option 1: Use carried usable spanner (if available)
            cost_from_carried = 1000000 # Infinity
            if man_carrying_usable:
                 # Travel from man's current location to nut
                 cost_from_carried = self.dist(man_location, nut_location)

            # Option 2: Pick up a usable spanner from a location
            min_cost_from_pickup = 1000000 # Infinity
            if usable_spanner_locs: # Only consider if there are usable spanners at locations
                for L_S in usable_spanner_locs:
                     # Travel from man to spanner, pickup, travel from spanner to nut
                     cost_via_pickup = self.dist(man_location, L_S) + 1 + self.dist(L_S, nut_location)
                     min_cost_from_pickup = min(min_cost_from_pickup, cost_via_pickup)

            # The cost to get man with usable spanner to nut location is the minimum of these options
            # If both options are impossible (e.g., no usable spanners anywhere), min will be infinity.
            # This should be caught by the unsolvability check if there are loose nuts,
            # but this provides a fallback for robustness.
            cost_to_reach_nut_with_spanner = min(cost_from_carried, min_cost_from_pickup)

            # If cost_to_reach_nut_with_spanner is still infinity, something is wrong or unsolvable
            if cost_to_reach_nut_with_spanner >= 1000000:
                 return 1000000 # Should have been caught earlier, but double check

            # Add this cost to the nut's total cost
            cost_this_nut += cost_to_reach_nut_with_spanner

            # Add cost for this nut to overall heuristic
            h += cost_this_nut

        return h
