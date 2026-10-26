import heapq
from collections import deque, defaultdict
import math
import logging

from heuristics.heuristic_base import Heuristic
# Assuming Task and Operator classes are available in the environment

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove parentheses and split by spaces
    parts = fact_str.strip('()').split()
    return tuple(parts)

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        Estimates the cost to reach the goal (tighten all required nuts)
        by summing the number of tighten actions, the number of spanner
        pickup actions needed, and an estimate of the travel cost.
        The travel cost is estimated as the maximum shortest path distance
        from the man's current location to any required location (nut locations
        and locations of needed spanners).

    Assumptions:
        - There is exactly one man object in the domain.
        - All locations mentioned in static 'link' facts are the primary locations.
          Locations from initial state/goal are also considered.
        - The location graph is connected (or relevant parts are connected).
        - Spanners are identified by facts like '(at ?s ?l)' and '(usable ?s)'
          or '(carrying ?m ?s)' and '(usable ?s)'.
        - Nuts to be tightened are specified in the goal using the '(tightened ?n)' predicate.

    Heuristic Initialization:
        - Parses static 'link' facts to build the location graph.
        - Identifies all relevant locations from static facts, initial state, and goal.
        - Computes all-pairs shortest paths between locations using BFS.
        - Identifies the set of nuts that need to be tightened from the goal state.
        - Identifies the man object name (by finding the object involved in an '(at ...)'
          fact in the initial state that is not a goal nut or a usable spanner).

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the goal is already reached (all goal nuts are tightened). If so, return 0.
        2. Identify the man's current location.
        3. Identify the set of loose nuts that are part of the goal. Let k be the count.
        4. Identify usable spanners currently carried by the man. Let c be the count.
        5. Identify usable spanners at locations and their locations. Let a be the count.
        6. Calculate the total number of usable spanners available (carried or at locations): total_usable = c + a.
        7. If the number of loose goal nuts (k) is greater than the total usable spanners, the goal is unreachable from this state; return infinity.
        8. Calculate the number of additional spanners needed from locations: needed_from_locs = max(0, k - c).
        9. Identify the locations of the loose goal nuts.
        10. Identify the locations of usable spanners currently at locations.
        11. Select the 'needed_from_locs' locations from the usable spanner locations that are closest to the man's current location. If a location has multiple usable spanners, it can contribute multiple times towards the 'needed_from_locs' count, but the location itself is only added once to the set of target locations.
        12. Combine the set of loose nut locations and the set of selected spanner locations into a set of target locations.
        13. Calculate the maximum shortest path distance from the man's current location to any location in the set of target locations. This is the estimated travel cost. If the set of target locations is empty, the travel cost is 0. If any target location is unreachable, the travel cost is infinity.
        14. The heuristic value is the sum of:
            - The number of loose goal nuts (k) (representing tighten actions).
            - The number of spanners needed from locations (needed_from_locs) (representing pickup actions).
            - The estimated travel cost.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goals_nuts = set()
        self.locations = set()
        self.adj = defaultdict(set)
        self.dist = {}
        self.man_name = None

        # 1. Parse static facts to build location graph and identify locations
        for fact_str in task.static:
            fact = parse_fact(fact_str)
            if fact[0] == 'link':
                l1, l2 = fact[1], fact[2]
                self.adj[l1].add(l2)
                self.adj[l2].add(l1)
                self.locations.add(l1)
                self.locations.add(l2)

        # Identify goal nuts
        goal_nuts_set = set()
        for goal_fact_str in task.goals:
            goal_fact = parse_fact(goal_fact_str)
            if goal_fact[0] == 'tightened':
                goal_nuts_set.add(goal_fact[1])
        self.goals_nuts = goal_nuts_set

        # Identify potential spanners from initial state (usable spanners)
        potential_spanners = set()
        for fact_str in task.initial_state:
             fact = parse_fact(fact_str)
             if fact[0] == 'usable':
                 potential_spanners.add(fact[1])

        # Identify locations and locatables from initial state
        potential_locatables = set()
        for fact_str in task.initial_state:
            fact = parse_fact(fact_str)
            if fact[0] == 'at':
                locatable, location = fact[1], fact[2]
                potential_locatables.add(locatable)
                self.locations.add(location) # Add locations from initial state

        # Find the man: a locatable that is not a goal nut or a potential spanner
        man_candidates = potential_locatables - self.goals_nuts - potential_spanners

        if len(man_candidates) == 1:
             self.man_name = list(man_candidates)[0]
        elif len(man_candidates) > 1:
             # If multiple candidates, pick one deterministically (e.g., first alphabetically)
             self.man_name = sorted(list(man_candidates))[0]
             logging.warning(f"Multiple man candidates found: {man_candidates}. Assuming man is {self.man_name}.")
        else:
             # Fallback: Try to find an object in initial state involved in 'carrying'
             for fact_str in task.initial_state:
                 fact = parse_fact(fact_str)
                 if fact[0] == 'carrying':
                     self.man_name = fact[1]
                     logging.warning(f"Man not found via 'at' facts, using 'carrying' fact: {self.man_name}.")
                     break # Found the man

             if self.man_name is None:
                 # Last resort: Assume 'bob' if it exists and no other man found
                 all_objects_in_init = {arg for fact_str in task.initial_state for arg in parse_fact(fact_str)[1:]}
                 if 'bob' in all_objects_in_init:
                      self.man_name = 'bob'
                      logging.warning("Man not found, assuming 'bob'.")
                 else:
                      logging.error("Could not identify the man object.")
                      # Heuristic will likely return inf or error if man_name is None


        # 2. Compute all-pairs shortest paths
        # Ensure all known locations are in the distance map structure
        for loc in list(self.locations): # Iterate over a copy
             if loc not in self.dist:
                  self.dist[loc] = {}
             for other_loc in list(self.locations): # Iterate over a copy
                  if other_loc not in self.dist[loc]:
                       self.dist[loc][other_loc] = math.inf

        for start_node in list(self.locations): # Iterate over a copy as locations might be added
            if start_node not in self.dist:
                 self.dist[start_node] = {l: math.inf for l in self.locations}

            self.dist[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node}

            while queue:
                u = queue.popleft()
                for v in self.adj.get(u, []):
                    # Ensure v is in self.locations and self.dist structure
                    if v not in self.locations:
                         self.locations.add(v)
                         # Add v to dist structure for all existing locations
                         for existing_loc in list(self.locations): # Iterate over copy
                              if v not in self.dist.get(existing_loc, {}):
                                   self.dist[existing_loc][v] = math.inf
                              if existing_loc not in self.dist.get(v, {}):
                                   self.dist[v][existing_loc] = math.inf
                         self.dist[v][v] = 0 # Distance from v to itself

                    if v not in visited:
                        visited.add(v)
                        self.dist[start_node][v] = self.dist[start_node][u] + 1
                        queue.append(v)


    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        # Note: task.goal_reached checks if self.task.goals <= state
        # This is correct for the spanner domain goal structure.
        if self.task.goal_reached(state):
            return 0

        man_loc = None
        loose_nuts_in_state = set()
        usable_spanners_carried = set()
        usable_spanners_at_locs = defaultdict(set)
        usable_spanners_in_state = set() # All usable spanners currently existing

        # Parse state
        state_facts = {parse_fact(f) for f in state}
        state_fact_strings = set(state) # Keep original strings for quick lookup

        # First identify all usable spanners
        for fact in state_facts:
             if fact[0] == 'usable':
                 spanner = fact[1]
                 usable_spanners_in_state.add(spanner)

        # Then process other facts using the usable spanners set
        for fact in state_facts:
            if fact[0] == 'at':
                obj, loc = fact[1], fact[2]
                if obj == self.man_name:
                    man_loc = loc
                elif obj in usable_spanners_in_state:
                       # It's a usable spanner at a location
                       usable_spanners_at_locs[loc].add(obj)
            elif fact[0] == 'loose':
                nut = fact[1]
                if nut in self.goals_nuts:
                    loose_nuts_in_state.add(nut)
            elif fact[0] == 'carrying':
                 carrier, spanner = fact[1], fact[2]
                 if carrier == self.man_name and spanner in usable_spanners_in_state:
                      # It's a usable spanner carried by the man
                      usable_spanners_carried.add(spanner)


        # 3. Number of loose goal nuts
        k = len(loose_nuts_in_state)

        # 4. Usable spanners carried
        c = len(usable_spanners_carried)

        # 5. Usable spanners at locations
        a = sum(len(spanners) for spanners in usable_spanners_at_locs.values())
        total_usable_spanners = c + a

        # 7. Check reachability based on spanners
        if k > total_usable_spanners:
            return math.inf # Cannot tighten k nuts with fewer usable spanners

        # 8. Spanners needed from locations
        needed_from_locs = max(0, k - c)

        # 9. Locations of loose goal nuts
        nut_locs = {}
        # Need to find locations of loose nuts. This info is in the state.
        # Re-iterate state facts to find locations of loose nuts identified earlier
        for fact in state_facts:
             if fact[0] == 'at':
                  obj, loc = fact[1], fact[2]
                  if obj in loose_nuts_in_state:
                       nut_locs[obj] = loc

        L_nuts_loose = {loc for loc in nut_locs.values()}

        # 10. Locations of usable spanners at locations
        L_spanners_usable_locs = list(usable_spanners_at_locs.keys())

        # 11. Select needed_from_locs closest spanner locations
        # Sort spanner locations by distance from man_loc, considering multiple spanners per location
        spanner_locs_with_dist_and_count = []
        for l_s in L_spanners_usable_locs:
             if man_loc is None or l_s not in self.dist.get(man_loc, {}):
                  # Man location unknown or spanner location not in precomputed distances
                  # This scenario should ideally be handled by returning inf earlier
                  # or by the BFS correctly setting distance to inf.
                  # If man_loc is None, distances are undefined.
                  # If l_s is not in dist[man_loc], dist is inf.
                  # Let's rely on the max_dist check later.
                  dist_to_ls = self.dist.get(man_loc, {}).get(l_s, math.inf)
             else:
                  dist_to_ls = self.dist[man_loc][l_s]

             if dist_to_ls == math.inf:
                  # If any usable spanner location is unreachable, and we need spanners, return inf.
                  # This check is implicitly handled by the total_usable_spanners check and the max_dist check.
                  pass # Keep inf distance

             num_spanners_at_loc = len(usable_spanners_at_locs[l_s])
             spanner_locs_with_dist_and_count.append((dist_to_ls, l_s, num_spanners_at_loc))

        # Sort by distance
        spanner_locs_with_dist_and_count.sort()

        L_spanners_needed = set()
        spanners_taken_count = 0
        # Iterate through sorted locations, taking spanners until needed_from_locs is met
        for dist_to_ls, l_s, num_spanners_at_loc in spanner_locs_with_dist_and_count:
             if spanners_taken_count >= needed_from_locs:
                  break # Already selected enough spanners

             can_take = min(num_spanners_at_loc, needed_from_locs - spanners_taken_count)
             if can_take > 0:
                  L_spanners_needed.add(l_s) # Add the location to the set of needed locations
                  spanners_taken_count += can_take


        # 12. Combine target locations
        All_target_locations = L_nuts_loose | L_spanners_needed

        # 13. Estimated travel cost
        travel_cost = 0
        if man_loc is None:
             # Man location not found in state, cannot compute travel
             logging.error("Man location not found in state.")
             return math.inf

        if All_target_locations:
             max_dist = 0
             unreachable_target = False
             for target_loc in All_target_locations:
                  # Ensure target_loc is in the distance map from man_loc
                  if man_loc not in self.dist or target_loc not in self.dist[man_loc]:
                       logging.warning(f"Distance from man_loc ({man_loc}) to target_loc ({target_loc}) not precomputed.")
                       unreachable_target = True
                       break # Cannot compute distance

                  dist_to_target = self.dist[man_loc][target_loc]

                  if dist_to_target == math.inf:
                       unreachable_target = True
                       break # Cannot reach this target

                  max_dist = max(max_dist, dist_to_target)

             if unreachable_target:
                  travel_cost = math.inf
             else:
                  travel_cost = max_dist
        # else: All_target_locations is empty, travel_cost remains 0. Correct.


        # 14. Calculate heuristic value
        if travel_cost == math.inf:
             return math.inf

        h_value = k + needed_from_locs + travel_cost

        return h_value
