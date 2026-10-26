import logging
from collections import defaultdict

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    Estimates the cost to reach the goal state (all nuts tightened) by simulating
    a greedy plan. The plan involves the man sequentially tightening each loose nut.
    To tighten a nut, the man must be at the nut's location and be carrying a
    usable spanner. If the man is not carrying a spanner when needed, the plan
    includes walking to the closest location with an available usable spanner
    and picking one up. The heuristic sums the estimated costs (walk actions +
    pickup actions + tighten actions) along this greedy path until all loose
    nuts are conceptually tightened.

    Assumptions:
    - The problem is solvable from the initial state (i.e., there are enough
      usable spanners in the initial state to tighten all goal nuts, and all
      locations are connected). The heuristic will return infinity if it
      encounters a situation where a needed item (spanner or nut) is unreachable
      or if no more usable spanners are available at locations when needed.
    - The man object is assumed to be the object involved in 'carrying' facts
      and the first argument of 'at' facts that is not a spanner or nut. Based
      on examples, this is typically 'bob'.
    - Spanners are objects that appear in 'usable' or 'carrying' facts.
    - Nuts are objects that appear in 'loose' facts or the goal predicate 'tightened'.
    - Links between locations are bidirectional.

    Heuristic Initialization:
    The constructor precomputes the shortest path distances between all pairs
    of locations based on the 'link' facts in the static information. This
    is done using a Breadth-First Search (BFS) starting from each location.
    It also identifies all locations present in the domain.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to extract:
        - The man's current location.
        - The set of loose nuts and their current locations.
        - The set of usable spanners currently carried by the man.
        - The set of usable spanners currently located at specific locations.
    2.  Count the total number of loose nuts (N). If N is 0, the state is a goal state, return 0.
    3.  Count the number of usable spanners currently carried (K_carried).
    4.  Count the number of usable spanners currently at locations (K_loc).
    5.  If the total number of usable spanners available (K_carried + K_loc) is less than the number of loose nuts (N), the state is likely unsolvable from this point with currently available spanners. Return infinity.
    6.  Initialize the heuristic value `h` to 0.
    7.  Initialize the man's current location `curr_l` to the actual man's location from the state.
    8.  Initialize the number of spanners carried conceptually by the man `k_carried` to K_carried.
    9.  Initialize the set of remaining loose nuts `remaining_nuts` and the available usable spanners at locations `available_spanners_at_loc_dict` (grouped by location).
    10. Enter a loop that continues as long as there are `remaining_nuts`.
    11. Inside the loop:
        a.  Check if the man needs a spanner (`k_carried == 0`).
        b.  If a spanner is needed:
            i.  Find the closest location `l_s` that currently has at least one usable spanner available (from `available_spanners_at_loc_dict`).
            ii. If no such location exists, return infinity (unsolvable).
            iii. Add the walk distance from `curr_l` to `l_s` (`self.distances[(curr_l, l_s)]`) to `h`.
            iv. Add 1 to `h` for the `pickup_spanner` action.
            v.  Increment `k_carried` by 1.
            vi. Remove one spanner from the conceptual pool at location `l_s` in `available_spanners_at_loc_dict`.
            vii. Update `curr_l` to `l_s`.
        c.  If a spanner is not needed (`k_carried > 0`):
            i.  Find the closest remaining loose nut `n` at location `l_n` (from `remaining_nuts` and `Nut_locations`).
            ii. If no remaining nut is found (should not happen if loop condition is correct) or the nut location is unreachable, return infinity (unsolvable).
            iii. Add the walk distance from `curr_l` to `l_n` (`self.distances[(curr_l, l_n)]`) to `h`.
            iv. Add 1 to `h` for the `tighten_nut` action.
            v.  Decrement `k_carried` by 1 (the spanner is conceptually used).
            vi. Remove nut `n` from `remaining_nuts`.
            vii. Update `curr_l` to `l_n`.
    12. Once the loop finishes (all nuts are tightened), return the accumulated heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.locations = set()
        graph = defaultdict(list)

        # Parse static facts to build the location graph
        for fact_str in task.static:
            parsed = self._parse_fact(fact_str)
            if parsed[0] == 'link':
                l1, l2 = parsed[1], parsed[2]
                self.locations.add(l1)
                self.locations.add(l2)
                graph[l1].append(l2)
                graph[l2].append(l1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            q = [(start_loc, 0)]
            visited = {start_loc: 0}
            self.distances[(start_loc, start_loc)] = 0 # Distance to self is 0

            head = 0
            while head < len(q):
                curr_loc, dist = q[head]
                head += 1

                # Store the distance when first visited (shortest path in unweighted graph)
                if (start_loc, curr_loc) not in self.distances:
                     self.distances[(start_loc, curr_loc)] = dist

                if curr_loc in graph:
                    for neighbor in graph[curr_loc]:
                        if neighbor not in visited:
                            visited[neighbor] = dist + 1
                            q.append((neighbor, dist + 1))
                            # Store distance here as well, it's the shortest path
                            self.distances[(start_loc, neighbor)] = dist + 1
                        # No need to check shorter path for unweighted BFS, first visit is shortest.


    def _parse_fact(self, fact_str):
        """Helper to parse a PDDL fact string into a tuple."""
        # Remove surrounding brackets and split by space
        parts = fact_str[1:-1].split()
        return tuple(parts)

    def __call__(self, node):
        state = node.state

        # 1. Parse the current state
        l_m = None
        Loose_nuts = set()
        Nut_locations = {} # nut_name -> location_name
        Usable_spanners_at_loc = [] # list of (spanner_name, location_name)
        Usable_spanners_carried = [] # list of spanner_name

        item_locations = {} # item -> location
        item_is_usable_spanner = set() # spanner_name
        item_is_loose_nut = set() # nut_name
        item_is_carried = set() # item_name (only man carries in this domain)

        # Collect all objects mentioned in the state
        all_objects_in_state = set()
        for fact_str in state:
             parsed = self._parse_fact(fact_str)
             for arg in parsed[1:]:
                 # Assume arguments that are not locations are objects
                 if arg not in self.locations:
                      all_objects_in_state.add(arg)


        # First pass to identify predicates and locations
        for fact_str in state:
            parsed = self._parse_fact(fact_str)
            pred = parsed[0]
            if pred == 'at':
                item, loc = parsed[1], parsed[2]
                item_locations[item] = loc
            elif pred == 'usable':
                spanner = parsed[1]
                item_is_usable_spanner.add(spanner)
            elif pred == 'loose':
                nut = parsed[1]
                item_is_loose_nut.add(nut)
            elif pred == 'carrying':
                # Assuming carrier is the man
                carried_item = parsed[2]
                item_is_carried.add(carried_item)

        # Identify Man, Spanners, Nuts based on predicates and state contents
        Spanners_in_state = {obj for obj in all_objects_in_state if obj in item_is_usable_spanner or obj in item_is_carried}
        # Nuts are those that are loose or are goal nuts (check goal facts)
        goal_nuts = set()
        for g in self.task.goals:
             parsed_goal = self._parse_fact(g)
             if parsed_goal[0] == 'tightened' and len(parsed_goal) > 1:
                  goal_nuts.add(parsed_goal[1])
        Nuts_in_state = {obj for obj in all_objects_in_state if obj in item_is_loose_nut or obj in goal_nuts}

        # The man is the locatable object that is not a spanner or nut
        Man_in_state = {obj for obj in all_objects_in_state if obj not in Spanners_in_state and obj not in Nuts_in_state and obj in item_locations}

        if len(Man_in_state) != 1:
             # Handle case where man is not uniquely identified or missing
             logging.error(f"Heuristic error: Could not uniquely identify the man in state. Found: {Man_in_state}")
             return float('inf') # Cannot compute heuristic without man

        man_name = list(Man_in_state)[0]
        l_m = item_locations.get(man_name)

        if l_m is None:
             # Man's location not found? Problematic state.
             logging.error(f"Heuristic error: Man '{man_name}' location not found in state.")
             return float('inf')


        Loose_nuts = {nut for nut in Nuts_in_state if nut in item_is_loose_nut}

        for nut in Loose_nuts:
             l_n = item_locations.get(nut)
             if l_n is None:
                  # Loose nut found but its location is unknown? Problematic state.
                  logging.error(f"Heuristic error: Location for loose nut {nut} not found.")
                  return float('inf')
             Nut_locations[nut] = l_n


        for spanner in Spanners_in_state:
             if spanner in item_is_usable_spanner:
                  if spanner in item_is_carried:
                       Usable_spanners_carried.append(spanner)
                  elif spanner in item_locations:
                       Usable_spanners_at_loc.append((spanner, item_locations[spanner]))
                  # else: usable spanner not carried and not at location? Assume valid.


        # 2. Count loose nuts
        N = len(Loose_nuts)
        if N == 0:
            return 0 # Goal state

        # 3. Count carried usable spanners
        k_carried = len(Usable_spanners_carried)

        # 4. Count usable spanners at locations and group by location
        available_spanners_at_loc_dict = defaultdict(list)
        for s, l in Usable_spanners_at_loc:
             available_spanners_at_loc_dict[l].append(s)

        # 5. Check solvability based on available spanners in current state
        K_loc = sum(len(spanners) for spanners in available_spanners_at_loc_dict.values())
        K_total = k_carried + K_loc

        if N > K_total:
             # Not enough usable spanners currently exist to tighten all loose nuts
             return float('inf')


        # 6. Initialize heuristic value
        h = 0

        # 7. Initialize current location (already done)
        curr_l = l_m

        # 8. k_carried is already initialized

        # 9. remaining_nuts is already initialized. available_spanners_at_loc_dict is initialized.

        # 10. Greedy loop
        remaining_nuts = set(Loose_nuts) # Create a mutable copy

        while remaining_nuts:
            # 11. Decide whether to get spanner or tighten nut
            if k_carried == 0:
                # Need spanner. Find closest location with available spanners.
                min_dist = float('inf')
                spanner_loc_to_go = None
                locations_with_usable_spanners = [l for l, spanners in available_spanners_at_loc_dict.items() if spanners]

                if not locations_with_usable_spanners:
                     # Need spanner but none available at locations. Unsolvable from here.
                     return float('inf')

                for l_s in locations_with_usable_spanners:
                    dist_to_spanner_loc = self.distances.get((curr_l, l_s), float('inf'))
                    if dist_to_spanner_loc == float('inf'):
                         # Spanner location is unreachable
                         continue # Check other spanner locations

                    if dist_to_spanner_loc < min_dist:
                        min_dist = dist_to_spanner_loc
                        spanner_loc_to_go = l_s

                if spanner_loc_to_go is None or min_dist == float('inf'):
                    # No reachable spanner location found
                    return float('inf') # Unreachable spanner location

                h += min_dist # walk to spanner
                h += 1 # pickup action
                k_carried += 1
                # Remove one spanner from the conceptual pool at this location
                available_spanners_at_loc_dict[spanner_loc_to_go].pop()
                if not available_spanners_at_loc_dict[spanner_loc_to_go]:
                     del available_spanners_at_loc_dict[spanner_loc_to_go] # Clean up empty entries

                curr_l = spanner_loc_to_go

            else: # k_carried > 0
                # Have spanner, go tighten closest nut
                min_dist = float('inf')
                best_nut = None
                best_nut_loc = None
                for nut in remaining_nuts:
                    l_n = Nut_locations.get(nut) # Location already checked during Nut_locations population

                    dist_to_nut_loc = self.distances.get((curr_l, l_n), float('inf'))
                    if dist_to_nut_loc == float('inf'):
                         # Nut location is unreachable
                         continue # Check other nuts

                    if dist_to_nut_loc < min_dist:
                        min_dist = dist_to_nut_loc
                        best_nut = nut
                        best_nut_loc = l_n

                if best_nut is None or min_dist == float('inf'):
                    # No reachable nut location found among remaining nuts
                    return float('inf') # Unreachable nut location

                h += min_dist # walk to nut
                h += 1 # tighten action
                k_carried -= 1 # Spanner is conceptually used
                remaining_nuts.remove(best_nut)
                curr_l = best_nut_loc

        # 12. All nuts tightened, return total cost
        return h
