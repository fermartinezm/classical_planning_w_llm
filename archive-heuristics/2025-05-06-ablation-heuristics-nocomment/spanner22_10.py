from collections import deque
import math # For infinity

# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper function to parse a PDDL fact string into its components
def get_parts(fact):
    """Helper function to parse a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

# Define the heuristic class, potentially inheriting from a base class
# class spannerHeuristic(Heuristic):
class spannerHeuristic: # Use this if no base class is provided
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing up
    the estimated costs for tightening each loose nut that is part of the goal.
    The estimation for each nut involves the cost of travel to the nut's location,
    the cost of acquiring a usable spanner (if one is not currently carried),
    and the cost of the tighten_nut action. The heuristic uses a greedy approach
    to estimate the sequence of actions: repeatedly acquire a spanner (if needed)
    and travel to the nearest loose nut, then tighten it.

    Assumptions:
    - The problem is solvable (i.e., there are enough usable spanners available
      at locations or carried by the man to tighten all goal nuts).
    - Nuts do not change locations.
    - Spanners become unusable after one tighten_nut action and cannot be repaired
      or dropped (based on the provided domain definition).
    - The man can carry multiple spanners simultaneously (inferred from action effects).
    - Locations are connected by an undirected graph defined by 'link' facts.
    - The costs of walk, pickup_spanner, and tighten_nut actions are all 1.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes static information from the task:
    - It identifies all locations from the 'link' facts.
    - It builds an adjacency list representation of the location graph based on 'link' facts.
    - It computes the shortest path distances between all pairs of locations using BFS.
    - It identifies the set of nuts that need to be tightened to reach the goal state.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the set of loose nuts that are also goal nuts. If this set is empty,
       the goal is reached, and the heuristic value is 0.
    2. Parse the current state to determine:
       - The man's current location.
       - The current location of all other locatable objects (spanners, nuts).
       - The usability status of all spanners.
       - Which spanners the man is currently carrying.
    3. Initialize the total estimated cost (travel_cost) to 0.
    4. Initialize the count of remaining loose nuts that need tightening.
    5. Initialize the count of usable spanners the man is currently carrying.
    6. Identify the set of locations where usable spanners are currently available (not carried).
    7. Enter a loop that continues as long as there are loose nuts remaining to tighten:
       a. Check if the man is currently carrying at least one usable spanner.
       b. If not carrying a usable spanner:
          i. Find the nearest location among the available usable spanner locations from the man's current location using the precomputed distances.
          ii. Add the distance to this spanner location to the total travel_cost.
          iii. Add 1 to the total travel_cost for the 'pickup_spanner' action.
          iv. Update the man's current location to the spanner location.
          v. Mark the spanner at this location as 'used' for the purpose of this heuristic calculation (e.g., remove the location from the set of available spanner locations). This is an approximation; if multiple spanners are at one location, this step is not perfectly accurate but simplifies the heuristic.
          vi. Increment the count of usable spanners the man is carrying (conceptually, for the heuristic).
          vii. If no usable spanner locations were available but nuts remain, return a large value indicating a likely unsolvable state.
       c. Now that the man is conceptually carrying a usable spanner:
          i. Find the nearest loose nut location among the remaining loose nuts from the man's current location using the precomputed distances.
          ii. Add the distance to this nut location to the total travel_cost.
          iii. Add 1 to the total travel_cost for the 'tighten_nut' action.
          iv. Update the man's current location to the nut location.
          v. Decrement the count of remaining loose nuts.
          vi. Decrement the count of usable spanners the man is carrying (conceptually, as one is used).
          vii. If no loose nut locations were found but nuts remain (should not happen in valid states), return a large value.
    8. Return the final total travel_cost as the heuristic value.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.adj_list = {}
        self.locations = set()
        self.goal_nuts = set()

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.adj_list.setdefault(loc1, []).append(loc2)
                self.adj_list.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Parse goal facts to find goal nuts
        # Goal is a frozenset of facts, e.g., frozenset({'(tightened nut1)', '(tightened nut2)'})
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts[0] == 'tightened':
                 self.goal_nuts.add(parts[1])

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all other nodes."""
        distances = {node: math.inf for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            for v in self.adj_list.get(u, []):
                if distances[v] == math.inf:
                    distances[v] = distances[u] + 1
                    queue.append(v)
        return distances

    def __call__(self, node):
        state = node.state

        # 1. Identify loose nuts that are goal nuts.
        loose_nuts_to_tighten = {n for n in self.goal_nuts if f'(loose {n})' in state}

        if not loose_nuts_to_tighten:
            return 0 # Goal reached for all relevant nuts

        # 2. Parse current state
        man_loc = None
        object_locations = {} # obj -> loc
        spanner_usable_status = {} # spanner -> bool
        man_obj = None
        all_spanners = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                object_locations[obj] = loc
                # Assuming man object name contains 'man' or 'bob' based on examples
                if 'man' in obj or 'bob' in obj: man_obj = obj
                elif 'spanner' in obj: all_spanners.add(obj)
            elif parts[0] == 'carrying':
                 m, s = parts[1], parts[2]
                 if 'man' in m or 'bob' in m: man_obj = m
                 if 'spanner' in s: all_spanners.add(s)
            elif parts[0] == 'usable':
                s = parts[1]
                if 'spanner' in s:
                    all_spanners.add(s)
                    spanner_usable_status[s] = True
            # We don't need nut status explicitly here, just check '(loose n)' fact

        # Ensure all found spanners have a usability status (default False)
        for s in all_spanners:
            spanner_usable_status.setdefault(s, False)

        # Get man's current location
        if man_obj:
            man_loc = object_locations.get(man_obj)
        if man_loc is None:
             # Man's location not found? Should not happen in valid states.
             # Return a high value to indicate a problematic state.
             return 1000000 # Use a large integer instead of inf if required

        # Identify usable spanners the man is carrying
        carried_usable_spanners = {s for s in all_spanners if man_obj and f'(carrying {man_obj} {s})' in state and spanner_usable_status.get(s, False)}

        # Identify usable spanners at locations
        usable_spanners_at_loc = {s for s in all_spanners if s in object_locations and spanner_usable_status.get(s, False) and f'(at {s} {object_locations[s]})' in state}

        # 3. Initialize variables for greedy calculation
        curr_loc = man_loc
        travel_cost = 0
        loose_nuts_remaining_count = len(loose_nuts_to_tighten)
        usable_spanners_carried_count = len(carried_usable_spanners)
        # Use a set of locations where usable spanners are available for pickup
        usable_spanner_locs_available = {object_locations[s] for s in usable_spanners_at_loc}

        # Create a mutable copy of loose nuts objects to track which ones are tightened
        loose_nuts_remaining_objects = set(loose_nuts_to_tighten)

        # 7. Loop while loose nuts remain
        while loose_nuts_remaining_count > 0:
            # a. Check if the man is currently carrying at least one usable spanner.
            if usable_spanners_carried_count == 0:
                # b. If not carrying a usable spanner: Need to pick one up.
                if not usable_spanner_locs_available:
                    # No usable spanners left anywhere? Unsolvable state for remaining nuts.
                    return 1000000 # Return a high value

                # i. Find the nearest location with an available usable spanner
                nearest_spanner_loc = None
                min_dist_spanner = math.inf
                for s_loc in usable_spanner_locs_available:
                    # Ensure the location exists in precomputed distances
                    if curr_loc in self.distances and s_loc in self.distances[curr_loc]:
                         d = self.distances[curr_loc][s_loc]
                         if d < min_dist_spanner:
                             min_dist_spanner = d
                             nearest_spanner_loc = s_loc

                if nearest_spanner_loc is None:
                     # Should not happen if usable_spanner_locs_available is not empty
                     return 1000000 # Error state

                # ii. Add travel cost
                travel_cost += min_dist_spanner
                # iii. Add pickup cost
                travel_cost += 1 # pickup action
                # iv. Update man's current location
                curr_loc = nearest_spanner_loc
                # v. Mark spanner location as used (approximation)
                usable_spanner_locs_available.discard(nearest_spanner_loc) # Use discard just in case
                # vi. Increment carried spanner count
                usable_spanners_carried_count += 1

            # c. Now carrying at least one usable spanner, need to go to a nut
            # i. Find the nearest loose nut location
            nearest_nut_loc = None
            min_dist_nut = math.inf
            nut_to_tighten_now = None

            # Find the location for each remaining loose nut object
            loose_nut_locations_remaining = {}
            for nut in loose_nuts_remaining_objects:
                 nut_loc = object_locations.get(nut)
                 if nut_loc:
                      loose_nut_locations_remaining[nut] = nut_loc
                 else:
                      # Nut location not found? Problematic state.
                      return 1000000 # Error state

            if not loose_nut_locations_remaining:
                 # Should not happen if loose_nuts_remaining_count > 0
                 return 1000000 # Error state

            for nut, nut_loc in loose_nut_locations_remaining.items():
                 # Ensure the location exists in precomputed distances
                 if curr_loc in self.distances and nut_loc in self.distances[curr_loc]:
                      d = self.distances[curr_loc][nut_loc]
                      if d < min_dist_nut:
                          min_dist_nut = d
                          nearest_nut_loc = nut_loc
                          nut_to_tighten_now = nut # Keep track of the nut object

            if nearest_nut_loc is None:
                 # Should not happen if loose_nut_locations_remaining is not empty
                 return 1000000 # Error state

            # ii. Add travel cost
            travel_cost += min_dist_nut
            # iii. Add tighten cost
            travel_cost += 1 # tighten action
            # iv. Update man's current location
            curr_loc = nearest_nut_loc
            # v. Decrement remaining loose nuts count and remove the object
            loose_nuts_remaining_count -= 1
            loose_nuts_remaining_objects.discard(nut_to_tighten_now) # Use discard just in case
            # vi. Decrement carried spanner count
            usable_spanners_carried_count -= 1

        # 8. Return total travel cost
        return travel_cost

