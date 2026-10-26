import heapq
from collections import deque, defaultdict

from heuristics.heuristic_base import Heuristic
# Assuming Task class is available in the environment from task.py

def parse_fact(fact_string):
    """Parses a fact string into a tuple (predicate, [args])."""
    # Remove parentheses and split by spaces
    parts = fact_string[1:-1].split()
    if not parts: # Handle empty fact string if necessary, though unlikely
        return (None, [])
    predicate = parts[0]
    args = parts[1:]
    return (predicate, args)

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    The heuristic estimates the cost to tighten all loose nuts in the goal.
    It sums the costs for necessary actions (tighten, pickup) and an estimated
    travel cost. The travel cost is estimated using a greedy approach:
    if the man needs a spanner, he travels to the closest available usable spanner,
    picks it up, and then travels to the closest loose nut location. If he
    already has a spanner, he travels directly to the closest loose nut location.
    This process repeats until all loose nuts are considered tightened.
    Shortest path distances between locations are precomputed using BFS.

    Assumptions:
    - The task object provides initial_state, goals, operators, and static facts.
    - Facts are represented as strings like '(predicate arg1 arg2)'.
    - Object types (man, spanner, nut, location) can be inferred from predicates
      in the initial state and goal facts.
    - The graph of locations connected by 'link' predicates is connected for
      all relevant locations (man's initial location, spanner locations, nut locations).
    - The cost of walk, pickup_spanner, and tighten_nut actions is 1.

    Heuristic Initialization:
    1. Identify object types (man, spanner, nut, location) by examining predicates
       in the initial state, goal facts, and static facts.
    2. Build the graph of locations based on 'link' predicates from static facts.
    3. Compute all-pairs shortest paths between locations using BFS. Store distances.
    4. Store initial locations of nuts (they are static) from the initial state.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state facts.
    2. Parse the man's current location from the state.
    3. Identify the set of loose nuts that are part of the goal. If this set is empty,
       the heuristic is 0 (goal reached for these nuts).
    4. Identify the set of usable spanners the man is currently carrying.
    5. Identify the list of usable spanners available at locations (spanner object, location object)
       in the current state.
    6. Check if the total number of usable spanners (carried + at locations) is less
       than the number of loose goal nuts. If so, the problem is unsolvable from this
       state, return infinity.
    7. Initialize total estimated cost (heuristic value) to 0.
    8. Initialize the number of usable spanners the man is considered to be carrying
       for the purpose of this heuristic calculation.
    9. Initialize the list of loose nuts that still need tightening.
    10. Initialize the list of usable spanners available at locations.
    11. Initialize the man's current location for the purpose of travel calculation.
    12. While there are still loose nuts to tighten:
        a. If the man is considered to be carrying a usable spanner:
           i. Find the closest loose nut location among the remaining nuts (using static locations).
           ii. Add the distance from the current location to this nut location to the total cost.
           iii. Update the current location to the nut location.
           iv. Mark the nut as tightened (remove from the list of nuts to tighten).
           v. Decrement the count of carried usable spanners (one is used).
           vi. Add 1 to the total cost for the 'tighten_nut' action.
        b. If the man is not carrying a usable spanner:
           i. Find the closest usable spanner location among the available spanners at locations.
           ii. Add the distance from the current location to this spanner location to the total cost.
           iii. Update the current location to the spanner location.
           iv. Mark the spanner as picked up (remove from the list of available spanners at locations).
           v. Increment the count of carried usable spanners.
           vi. Add 1 to the total cost for the 'pickup_spanner' action.
    13. Return the total estimated cost.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static = task.static

        self._locations = set()
        self._nuts = set()
        self._spanners = set()
        self._man = None

        self._adj = defaultdict(list)
        self._distances = {} # Dict of dicts: distances[loc1][loc2] = dist

        self._nut_initial_locs = {} # nut_name -> location_name

        # 1. Identify object types and build location graph
        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        man_names = set()
        spanner_names = set()
        nut_names = set()
        location_names = set()

        for fact_str in all_facts:
            pred, args = parse_fact(fact_str)
            if pred == 'link' and len(args) == 2:
                l1, l2 = args
                location_names.add(l1)
                location_names.add(l2)
                self._adj[l1].append(l2)
                self._adj[l2].append(l1) # Assuming links are bidirectional
            elif pred == 'at' and len(args) == 2:
                obj, loc = args
                location_names.add(loc)
                # Infer types based on usage in predicates
                # This is a simplification based on the spanner domain structure
                if any(f.startswith(f'(carrying {obj} ') for f in all_facts):
                     man_names.add(obj)
                elif any(f.startswith(f'(usable {obj})') for f in all_facts):
                     spanner_names.add(obj)
                elif any(f.startswith(f'(loose {obj})') for f in all_facts) or any(f.startswith(f'(tightened {obj})') for f in all_facts):
                     nut_names.add(obj)

        self._man = list(man_names)[0] if man_names else None # Assuming exactly one man
        self._spanners = spanner_names
        self._nuts = nut_names
        self._locations = location_names

        # Store initial nut locations (they are static)
        for fact_str in task.initial_state:
            pred, args = parse_fact(fact_str)
            if pred == 'at' and len(args) == 2:
                obj, loc = args
                if obj in self._nuts:
                    self._nut_initial_locs[obj] = loc

        # 2. & 3. Compute all-pairs shortest paths
        self._distances = {loc: {other: float('inf') for other in self._locations} for loc in self._locations}
        for loc in self._locations:
            self._distances[loc][loc] = 0
            q = deque([(loc, 0)])
            visited = {loc}
            while q:
                current_loc, dist = q.popleft()
                for neighbor in self._adj.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self._distances[loc][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, handles unknown locations."""
        if loc1 not in self._locations or loc2 not in self._locations:
            # This implies an unreachable location or a location not defined in links/initial state.
            # Treat as infinite distance.
            return float('inf')
        return self._distances[loc1].get(loc2, float('inf')) # Use .get for safety

    def __call__(self, node):
        state = node.state

        # 2. Parse man's current location
        man_location = None
        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'at' and len(args) == 2 and args[0] == self._man:
                man_location = args[1]
                break
        if man_location is None:
             # Man must always be at a location in a valid state
             # If not found, something is wrong with the state representation or problem.
             # Return infinity as it's likely an invalid or unreachable state.
             return float('inf')


        # 3. Identify loose nuts in goal
        loose_goal_nuts = set()
        # Find all nuts that are goals (i.e., need to be tightened)
        goal_nuts_to_tighten = {parse_fact(g)[1][0] for g in self.goals if parse_fact(g)[0] == 'tightened'}

        # Check which of these goal nuts are currently loose
        for nut in goal_nuts_to_tighten:
             if f'(loose {nut})' in state:
                 loose_goal_nuts.add(nut)

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # 4. Identify usable spanners carried by man
        usable_spanners_carried = set()
        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'carrying' and len(args) == 2 and args[0] == self._man:
                spanner = args[1]
                if f'(usable {spanner})' in state:
                    usable_spanners_carried.add(spanner)

        # 5. Identify usable spanners at locations
        usable_spanners_at_locs = [] # List of (spanner, location)
        for spanner in self._spanners:
             # Check if spanner is usable and at a location (not carried)
             if f'(usable {spanner})' in state:
                 is_carried = any(parse_fact(f)[0] == 'carrying' and parse_fact(f)[1][1] == spanner for f in state)
                 if not is_carried:
                     # Find its location in the current state
                     spanner_loc = None
                     for fact_str in state:
                         pred, args = parse_fact(fact_str)
                         if pred == 'at' and len(args) == 2 and args[0] == spanner:
                             spanner_loc = args[1]
                             break
                     if spanner_loc:
                         usable_spanners_at_locs.append((spanner, spanner_loc))

        # 6. Solvability check
        num_nuts_to_tighten = len(loose_goal_nuts)
        num_usable_spanners_available = len(usable_spanners_carried) + len(usable_spanners_at_locs)
        if num_nuts_to_tighten > num_usable_spanners_available:
            return float('inf') # Not enough spanners in the entire problem instance

        # 7. Initialize heuristic calculation variables
        h = 0
        current_location = man_location
        nuts_remaining = list(loose_goal_nuts) # Use a list to remove elements
        spanners_at_locs_remaining = list(usable_spanners_at_locs) # Use a list
        carried_usable_count = len(usable_spanners_carried)

        # 12. Greedy travel and action cost estimation
        while nuts_remaining:
            if carried_usable_count > 0:
                # Man has a spanner, go to the closest nut
                closest_nut = None
                min_dist = float('inf')
                loc_of_closest_nut = None

                for nut in nuts_remaining:
                    # Nut location is static
                    nut_loc = self._nut_initial_locs.get(nut)
                    if nut_loc is None:
                         # This nut doesn't have an initial location? Problematic instance.
                         return float('inf') # Safeguard for malformed instances

                    dist = self.get_distance(current_location, nut_loc)
                    if dist < min_dist:
                        min_dist = dist
                        closest_nut = nut
                        loc_of_closest_nut = nut_loc

                # If min_dist is inf, it means the nut location is unreachable
                if min_dist == float('inf'):
                    return float('inf')

                # Add travel cost to closest nut
                h += min_dist
                current_location = loc_of_closest_nut

                # Use spanner and tighten nut
                carried_usable_count -= 1
                nuts_remaining.remove(closest_nut)
                h += 1 # Cost of tighten_nut action

            else: # Man needs a spanner
                # Go to the closest available usable spanner at a location
                closest_spanner_info = None # (spanner, location)
                min_dist = float('inf')
                loc_of_closest_spanner = None

                for (spanner, loc) in spanners_at_locs_remaining:
                    dist = self.get_distance(current_location, loc)
                    if dist < min_dist:
                        min_dist = dist
                        closest_spanner_info = (spanner, loc)
                        loc_of_closest_spanner = loc

                # If no usable spanners left at locations, and we need one, it's unsolvable.
                # This check is redundant if the total count check passed, but harmless.
                if closest_spanner_info is None or min_dist == float('inf'):
                     return float('inf')

                # Add travel cost to closest spanner
                h += min_dist
                current_location = loc_of_closest_spanner

                # Pick up spanner
                spanners_at_locs_remaining.remove(closest_spanner_info)
                carried_usable_count += 1
                h += 1 # Cost of pickup_spanner action

        # 13. Return total estimated cost
        return h
