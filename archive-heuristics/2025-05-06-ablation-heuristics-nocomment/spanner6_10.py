# Required imports
from fnmatch import fnmatch
from collections import deque
import math # For infinity

# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

class spannerHeuristic: # Inherit from Heuristic in actual usage
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    This heuristic estimates the cost to reach the goal state (all goal nuts
    tightened) by summing the estimated number of actions (tighten, pickup)
    and an estimated walk cost. The walk cost is estimated based on the
    minimum distances required to visit spanner and nut locations in an
    idealized alternating sequence.

    Assumptions:
    - The state representation is a frozenset of strings like '(predicate arg1 arg2)'.
    - PDDL facts can be parsed by removing parentheses and splitting by spaces.
    - Locations are connected by 'link' predicates, forming a directed graph.
      Distances are shortest path distances in this graph.
    - Nut locations are static and provided in the initial state or static facts.
    - Spanner initial locations are provided in the initial state or static facts.
    - A spanner is usable if the '(usable ?s)' fact is present in the state.
      The 'tighten_nut' action consumes the 'usable' property.
    - The man can carry at most one spanner at a time (implicitly, due to no drop/multiple carry actions).
    - Problem instances are solvable with the available usable spanners.
    - Object types can be inferred from naming conventions (e.g., 'nut' prefix for nuts,
      'spanner' prefix for spanners, the single 'man' object is identified by being
      a locatable that is not a nut or spanner). This is a simplification
      as type information is not directly available in the Task object provided.

    Heuristic Initialization:
    1. Parse static facts to build the location graph based on 'link' predicates.
       Collect all unique locations mentioned in links, initial object locations,
       and goal/initial state 'at' facts.
    2. Compute all-pairs shortest path distances between all identified locations
       using Breadth-First Search (BFS). Store these distances.
    3. Parse static facts and initial state to identify the static locations of nuts
       and initial locations of spanners.
    4. Parse goal facts to identify the set of nuts that need to be tightened.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify the current state components by iterating through the state facts:
        - Man's current location.
        - The spanner the man is currently carrying (if any).
        - The set of usable spanners currently at locations (facts '(at ?s ?l)' and '(usable ?s)').
        - The set of goal nuts that are currently loose (fact '(loose ?n)' and ?n is a goal nut).
        - The set of spanners that are currently usable (fact '(usable ?s)').
    2.  Count the number of loose goal nuts, let this be `k`.
    3.  If `k` is 0, the goal is reached, return 0.
    4.  Initialize the heuristic value `h` with `k` (representing the `tighten_nut` actions needed).
    5.  Determine if the man is currently carrying a spanner that is also usable in the current state.
    6.  Calculate the number of `pickup_spanner` actions needed from locations. This is `k` if the man
        is not carrying a usable spanner, or `k-1` if he is (assuming he uses the carried one
        for the first nut).
    7.  Check if the number of usable spanners currently at locations is sufficient for the needed pickups.
        If not, the state is likely unsolvable, return `math.inf`.
    8.  Add the number of needed pickups from locations to `h`.
    9.  Estimate the walk cost. This is done by considering an idealized sequence of actions:
        - If the man is carrying a usable spanner, the sequence starts with walking to a nut.
          Estimated cost: `min_dist(man_location, any_loose_nut_location)`.
        - If the man is not carrying a usable spanner, the sequence starts with walking to a spanner.
          Estimated cost: `min_dist(man_location, any_usable_spanner_at_loc)`.
        - After the first step, the sequence alternates between going from a nut location to a spanner location,
          and from a spanner location to a nut location.
        - For the remaining `k-1` nuts (if k > 1), there are `k-1` segments of walking from a nut location
          to a usable spanner location, and `k-1` segments of walking from a usable spanner location
          to a nut location. Estimate these costs using `min_dist(any_loose_nut_location, any_usable_spanner_at_loc)`
          and `min_dist(any_usable_spanner_at_loc, any_loose_nut_location)`.
        - If the man was not carrying a usable spanner initially, the first pickup leads him to a spanner location.
          He then needs to go to a nut location. This adds one extra segment of walking from a spanner location
          to a nut location compared to the case where he started with a spanner.
    10. Add the estimated walk cost to `h`. Handle cases where required locations are unreachable (distance is infinity).
    11. Return the total heuristic value as an integer.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.location_graph = {}  # Adjacency list {loc: [neighbor1, neighbor2, ...]}
        self.all_locations = set()
        self.nut_locations = {}  # {nut_name: location}
        self.all_nuts = set()
        self.all_spanners = set()
        self.all_locatables = set() # To identify objects that can be at locations

        # Collect all locations and locatables from static facts, initial state, and goals
        all_facts_in_task = set(task.initial_state) | set(task.goals) | set(static_facts)

        for fact in all_facts_in_task:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)
            elif parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 self.all_locations.add(loc)
                 self.all_locatables.add(obj)
                 if obj.startswith('nut'):
                     self.nut_locations[obj] = loc # Store initial/static nut location
                     self.all_nuts.add(obj)
                 elif obj.startswith('spanner'):
                     self.all_spanners.add(obj)
            elif parts[0] == 'tightened' and len(parts) == 2:
                  nut = parts[1]
                  self.all_nuts.add(nut)
            elif parts[0] == 'loose' and len(parts) == 2:
                  nut = parts[1]
                  self.all_nuts.add(nut)
            elif parts[0] == 'carrying' and len(parts) == 2:
                  self.all_locatables.add(parts[1]) # Man
                  self.all_locatables.add(parts[2]) # Spanner
                  if parts[2].startswith('spanner'):
                      self.all_spanners.add(parts[2])
            elif parts[0] == 'usable' and len(parts) == 2:
                  if parts[1].startswith('spanner'):
                      self.all_spanners.add(parts[1])

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = self._bfs(start_loc)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # Ensure nut_locations has entries for all goal nuts (should be in initial state 'at')
        for nut in self.goal_nuts:
             if nut not in self.nut_locations:
                 # Try to find initial location from initial state if not in static
                 for fact in task.initial_state:
                      parts = get_parts(fact)
                      if parts[0] == 'at' and len(parts) == 3 and parts[1] == nut:
                           self.nut_locations[nut] = parts[2]
                           break


    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all reachable locations."""
        distances = {loc: math.inf for loc in self.all_locations}
        if start_loc not in self.all_locations:
             # Start location is not in the set of known locations, cannot reach anything
             return distances

        distances[start_loc] = 0
        queue = deque([start_loc])
        visited = {start_loc}

        while queue:
            current_loc = queue.popleft()

            # Check if current_loc exists as a key in the graph (isolated locations won't have neighbors)
            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if neighbor in self.all_locations and neighbor not in visited: # Ensure neighbor is a known location
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated cost to reach the goal state (non-negative integer or infinity).
        """
        state = node.state

        # 1. Identify current state components
        man_location = None
        carried_spanner = None
        usable_spanners_at_locs = {} # {spanner_name: location}
        loose_goal_nuts = set() # {nut_name}
        usable_spanners_in_state = set() # {spanner_name}
        man_name = None # Assuming there's only one man, find his name

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Identify the man object (locatable that is not a nut or spanner)
                if obj in self.all_locatables and obj not in self.all_nuts and obj not in self.all_spanners:
                     man_name = obj # Found the man object name
                     man_location = loc
            elif parts[0] == 'carrying' and len(parts) == 2:
                 carrier, obj = parts[1], parts[2]
                 # Assuming the carrier is the man
                 if man_name is not None and carrier == man_name and obj in self.all_spanners:
                     carried_spanner = obj
            elif parts[0] == 'loose' and len(parts) == 2:
                 nut = parts[1]
                 if nut in self.goal_nuts:
                     loose_goal_nuts.add(nut)
            elif parts[0] == 'usable' and len(parts) == 2:
                 spanner = parts[1]
                 if spanner in self.all_spanners:
                     usable_spanners_in_state.add(spanner)

        # Now process 'at' facts for usable spanners at locations
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj in self.all_spanners and obj in usable_spanners_in_state:
                     usable_spanners_at_locs[obj] = loc

        # 2. Count loose goal nuts
        k = len(loose_goal_nuts)

        # If k is 0, all goal nuts are tightened
        if k == 0:
            return 0

        # 3. Estimate tighten actions
        h = k

        # 4. Determine if man is carrying a usable spanner
        man_carrying_usable = False
        if carried_spanner and carried_spanner in usable_spanners_in_state:
             man_carrying_usable = True

        # 5. Estimate pickup actions needed from locations
        # Each loose nut needs a distinct usable spanner.
        # If man carries a usable one, he has one ready for the first nut.
        # The remaining k-1 nuts need spanners picked up from locations.
        # If man doesn't carry a usable one, all k nuts need spanners picked up from locations.
        num_pickups_needed_from_locs = k - (1 if man_carrying_usable else 0)

        # 6. Check if enough usable spanners are available at locations
        if num_pickups_needed_from_locs > len(usable_spanners_at_locs):
             return math.inf # Unsolvable state: not enough usable spanners at locations

        h += num_pickups_needed_from_locs

        # 7. Estimate walk cost
        L_man = man_location
        # Get locations of loose goal nuts, ensuring they are in nut_locations (populated in init)
        L_nuts = {self.nut_locations[n] for n in loose_goal_nuts if n in self.nut_locations}
        L_spanners = set(usable_spanners_at_locs.values()) # Locations of usable spanners at locations

        # Check if man_location is valid and reachable from itself (always true if in all_locations)
        if L_man is None or L_man not in self.dist:
             # Man's location is unknown or not in the graph - should not happen in valid states
             return math.inf

        # Calculate minimum distances needed for walk cost estimation
        min_dist_man_to_nut = math.inf
        if L_nuts:
            for loc_n in L_nuts:
                if loc_n in self.dist[L_man]: # Check if reachable from man's location
                     min_dist_man_to_nut = min(min_dist_man_to_nut, self.dist[L_man][loc_n])

        min_dist_man_to_usable_spanner_at_loc = math.inf
        if L_spanners:
            for loc_s in L_spanners:
                 if loc_s in self.dist[L_man]: # Check if reachable from man's location
                     min_dist_man_to_usable_spanner_at_loc = min(min_dist_man_to_usable_spanner_at_loc, self.dist[L_man][loc_s])

        min_dist_nut_to_usable_spanner_at_loc = math.inf
        if L_nuts and L_spanners:
            for loc_n in L_nuts:
                if loc_n in self.dist: # Check if nut location is a valid start for BFS
                    for loc_s in L_spanners:
                        if loc_s in self.dist[loc_n]: # Check if reachable from nut location
                            min_dist_nut_to_usable_spanner_at_loc = min(min_dist_nut_to_usable_spanner_at_loc, self.dist[loc_n][loc_s])

        min_dist_usable_spanner_at_loc_to_nut = math.inf
        if L_spanners and L_nuts:
            for loc_s in L_spanners:
                 if loc_s in self.dist: # Check if spanner location is a valid start for BFS
                    for loc_n in L_nuts:
                        if loc_n in self.dist[loc_s]: # Check if reachable from spanner location
                            min_dist_usable_spanner_at_loc_to_nut = min(min_dist_usable_spanner_at_loc_to_nut, self.dist[loc_s][loc_n])

        # Add walk cost based on the sequence
        if man_carrying_usable:
            # Sequence: Man -> Nut1 -> Spanner2 -> Nut2 -> ... -> SpannerK -> NutK
            # First step: Man to closest nut
            if min_dist_man_to_nut == math.inf: return math.inf # Cannot reach any nut location
            h += min_dist_man_to_nut

            # Remaining steps: (Nut -> Spanner -> Nut) sequence k-1 times
            if k > 1:
                # Need k-1 spanners from locations. We checked enough exist, so L_spanners is not empty.
                # L_nuts is not empty (k>1 implies k>0).
                if min_dist_nut_to_usable_spanner_at_loc == math.inf or min_dist_usable_spanner_at_loc_to_nut == math.inf: return math.inf # Cannot connect nuts and spanners
                h += (k - 1) * min_dist_nut_to_usable_spanner_at_loc
                h += (k - 1) * min_dist_usable_spanner_at_loc_to_nut
        else:
            # Sequence: Man -> Spanner1 -> Nut1 -> Spanner2 -> Nut2 -> ... -> SpannerK -> NutK
            # First step: Man to closest usable spanner at location
            # Need k spanners from locations. We checked enough exist, so L_spanners is not empty.
            if min_dist_man_to_usable_spanner_at_loc == math.inf: return math.inf # Cannot reach any usable spanner location
            h += min_dist_man_to_usable_spanner_at_loc

            # Next step: Spanner to closest nut (k times in total sequence)
            # L_nuts is not empty (k>0).
            if min_dist_usable_spanner_at_loc_to_nut == math.inf: return math.inf # Cannot reach any nut location from spanner locations
            h += k * min_dist_usable_spanner_at_loc_to_nut

            # Remaining steps: (Nut -> Spanner) sequence k-1 times
            if k > 1:
                # Need k-1 spanners from locations. L_spanners is not empty.
                if min_dist_nut_to_usable_spanner_at_loc == math.inf: return math.inf # Cannot reach any spanner location from nut locations
                h += (k - 1) * min_dist_nut_to_usable_spanner_at_loc

        # 8. Return total heuristic value
        # Ensure heuristic is integer (distances are integers, counts are integers)
        return int(h)
