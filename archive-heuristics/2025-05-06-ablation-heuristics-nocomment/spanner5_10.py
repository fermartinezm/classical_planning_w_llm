# from heuristics.heuristic_base import Heuristic # Uncomment this line in the actual environment

from fnmatch import fnmatch
from collections import deque
import sys # Needed for float('inf') comparison

# Helper function to parse PDDL facts
def get_parts(fact):
    """Removes parentheses and splits a fact string into parts."""
    return fact[1:-1].split()

# Helper function to match fact parts with patterns
def match(fact, *args):
    """Checks if fact parts match given patterns."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Mock Heuristic base class if running standalone
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass
    # print("Warning: Using mock Heuristic base class.")


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        Estimates the cost to reach the goal (tighten all specified nuts)
        by summing the estimated costs of:
        1. Tightening each loose goal nut.
        2. Picking up necessary spanners.
        3. Walking to nut locations and spanner locations.

        The walk cost is estimated using a greedy strategy:
        If the man needs a spanner (doesn't carry enough for remaining nuts),
        he walks to the nearest usable spanner at a location and picks it up.
        If the man has a spanner, he walks to the nearest loose goal nut and tightens it.
        This process repeats until all goal nuts are tightened.
        Distances between locations are precomputed using BFS.

    Assumptions:
        - The domain follows the PDDL definition provided.
        - Nuts do not move from their initial locations. Their location is determined
          from the initial state or current state facts like '(at nutX locationY)'.
        - Spanners, once used for tightening, become unusable.
        - The location graph is connected (or at least, all relevant locations
          (man start, initial nut locations, initial spanner locations) are in the same
          connected component). The heuristic might return a large value if
          required locations are unreachable.
        - The heuristic is non-negative. It is 0 if and only if the goal is reached.
        - It is not guaranteed to be admissible, but aims to be informative for GBFS.
        - The man object is named 'bob'.

    Heuristic Initialization:
        - Parses static facts to build the location graph based on 'link' predicates.
        - Computes all-pairs shortest paths between locations using BFS.
        - Stores the set of nuts that are part of the goal condition.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the goal is already reached (all goal nuts tightened). If yes, return 0.
        2. Extract dynamic state information: man's current location, set of loose goal nuts,
           set of usable spanners carried by the man, and dictionary of usable spanners
           at locations ({spanner_name: location}).
        3. Check if the total number of available usable spanners (carried + at locations)
           is less than the number of loose goal nuts. If so, the problem is likely
           unsolvable in this domain; return a large value (1000000).
        4. Initialize costs: walk_cost = 0, pickup_cost = 0, tighten_cost = 0.
        5. Initialize mutable sets/counts for the greedy simulation:
           `current_loc = man_location`
           `spanners_carried_count = count of usable spanners carried`
           `nuts_to_tighten_sim = set of loose goal nuts`
           `usable_spanners_at_loc_sim = dictionary {spanner_name: location}`
           `nut_current_locations_sim = dictionary {nut_name: location}` (copy from extracted state info)
        6. Enter a loop that continues as long as there are nuts left to tighten (`nuts_to_tighten_sim` is not empty):
           a. If `spanners_carried_count > 0`:
              - The man has a spanner and can tighten a nut.
              - Find the nut in `nuts_to_tighten_sim` that is nearest to `current_loc` using the precomputed distances (`self.get_distance`).
              - Add the distance to this nut's location (`nut_current_locations_sim[nearest_nut]`) to `walk_cost`.
              - Update `current_loc` to the nut's location.
              - Decrement `spanners_carried_count`.
              - Remove the chosen nut from `nuts_to_tighten_sim`.
              - Increment `tighten_cost` (cost of the tighten action).
           b. If `spanners_carried_count == 0`:
              - The man needs a spanner.
              - Find the usable spanner in `usable_spanners_at_loc_sim` that is nearest to `current_loc`.
              - If no usable spanners are available at locations (`usable_spanners_at_loc_sim` is empty), return the large unsolvable value (1000000).
              - Add the distance to this spanner's location (`usable_spanners_at_loc_sim[nearest_spanner]`) to `walk_cost`.
              - Update `current_loc` to the spanner's location.
              - Increment `spanners_carried_count`.
              - Remove the chosen spanner from `usable_spanners_at_loc_sim`.
              - Increment `pickup_cost` (cost of the pickup action).
        7. The loop terminates when `nuts_to_tighten_sim` is empty.
        8. The total heuristic value is `walk_cost + pickup_cost + tighten_cost`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Precompute location graph and distances
        self.location_graph = {}
        self.all_locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "link", "*", "*"):
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        self.all_pairs_distances = {}
        for start_node in self.all_locations:
            distances = {loc: float('inf') for loc in self.all_locations}
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_loc = queue.popleft()

                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_loc] + 1
                            queue.append(neighbor)

            for end_node in self.all_locations:
                self.all_pairs_distances[(start_node, end_node)] = distances[end_node]

        # Get goal nut names from goal facts (these are the nuts we care about)
        self.goal_nut_names = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}


    def get_distance(self, loc1, loc2):
        """Returns the precomputed shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        return self.all_pairs_distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # Extract dynamic state information
        man_location = None
        usable_spanners_carried = set()
        usable_spanners_at_loc = {} # {spanner_name: location}
        nut_current_locations = {} # {nut_name: location}

        # Identify the man (assuming name 'bob')
        man_name = 'bob' # Hardcoded based on examples

        # Collect spanner names mentioned in state that are usable or carried
        all_spanners_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if match(fact, "usable", "*"):
                 spanner_name = parts[1]
                 all_spanners_in_state.add(spanner_name)
             elif match(fact, "carrying", man_name, "*"):
                 spanner_name = parts[2]
                 all_spanners_in_state.add(spanner_name)

        # Populate locations and carried status
        for fact in state:
             parts = get_parts(fact)
             if match(fact, "at", "*", "*"):
                 obj, loc = parts[1], parts[2]
                 if obj == man_name:
                     man_location = loc
                 elif obj in self.goal_nut_names: # It's a goal nut
                     nut_current_locations[obj] = loc
                 elif obj in all_spanners_in_state and '(usable ' + obj + ')' in state:
                      usable_spanners_at_loc[obj] = loc
             elif match(fact, "carrying", man_name, "*"):
                  spanner_name = parts[2]
                  if spanner_name in all_spanners_in_state and '(usable ' + spanner_name + ')' in state:
                       usable_spanners_carried.add(spanner_name)

        # Identify loose goal nuts
        loose_goal_nuts = {
            nut_name for nut_name in self.goal_nut_names
            if '(loose ' + nut_name + ')' in state
        }

        # Check for unsolvability (not enough usable spanners)
        num_loose_goal_nuts = len(loose_goal_nuts)
        num_usable_spanners_available = len(usable_spanners_carried) + len(usable_spanners_at_loc)

        if num_usable_spanners_available < num_loose_goal_nuts:
            # Not enough spanners in the entire state to tighten all required nuts
            return 1000000 # Large number representing infinity

        # Greedy Walk Cost Calculation
        current_loc = man_location
        spanners_carried_count = len(usable_spanners_carried)
        nuts_to_tighten_sim = set(loose_goal_nuts)
        usable_spanners_at_loc_sim = dict(usable_spanners_at_loc) # Create a mutable copy
        nut_current_locations_sim = dict(nut_current_locations) # Create a mutable copy


        walk_cost = 0
        pickup_cost = 0
        tighten_cost = 0

        while nuts_to_tighten_sim:
            if spanners_carried_count > 0:
                # Man has a spanner, go to nearest nut
                min_dist = float('inf')
                nearest_nut = None
                nearest_nut_loc = None

                for nut in nuts_to_tighten_sim:
                    nut_loc = nut_current_locations_sim.get(nut)
                    # Ensure nut location is known and reachable
                    if nut_loc is not None:
                        dist = self.get_distance(current_loc, nut_loc)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_nut = nut
                            nearest_nut_loc = nut_loc

                if nearest_nut is None or min_dist == float('inf'):
                     # Cannot reach any remaining nut. Unsolvable.
                     return 1000000

                walk_cost += min_dist
                current_loc = nearest_nut_loc
                spanners_carried_count -= 1
                nuts_to_tighten_sim.remove(nearest_nut)
                tighten_cost += 1

            else: # spanners_carried_count == 0
                # Man needs a spanner, go to nearest usable spanner at a location
                min_dist = float('inf')
                nearest_spanner = None
                nearest_spanner_loc = None

                # Find nearest usable spanner at a location
                for spanner, loc in usable_spanners_at_loc_sim.items():
                    dist = self.get_distance(current_loc, loc)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_spanner = spanner
                        nearest_spanner_loc = loc

                if nearest_spanner is None or min_dist == float('inf'):
                    # Cannot get a spanner, but still need to tighten nuts. Unsolvable.
                    return 1000000 # Large number representing infinity

                walk_cost += min_dist
                current_loc = nearest_spanner_loc
                spanners_carried_count += 1
                del usable_spanners_at_loc_sim[nearest_spanner]
                pickup_cost += 1

        # Total heuristic is the sum of estimated actions
        return walk_cost + pickup_cost + tighten_cost
