from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose goal nuts. It simulates the process sequentially, greedily choosing
    the cheapest next nut to tighten based on the man's current location
    and available usable spanners.

    # Assumptions:
    - The man can carry multiple spanners.
    - Using a spanner makes it unusable, but the man continues carrying it.
    - The man object can be identified from the initial state facts (fragile).
    - All relevant locations (man, nuts, spanners) are part of the linked graph.

    # Heuristic Initialization
    - Builds the location graph from `link` facts and precomputes all-pairs
      shortest paths using BFS.
    - Identifies the set of nuts that need to be tightened (goal nuts).
    - Attempts to identify the man object from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic simulates the process of tightening nuts one by one:
    1. Identify the man's current location, the set of loose goal nuts,
       the set of usable spanners (distinguishing between carried and available on the ground),
       and the location of each relevant object.
    2. If all goal nuts are tightened, the heuristic is 0.
    3. While there are still loose goal nuts:
       a. Check if there are any usable spanners available (either carried or on the ground).
          If not, the remaining nuts cannot be tightened, return infinity.
       b. Determine the minimum cost to tighten *any* of the remaining loose goal nuts.
          This cost is the minimum over all remaining loose nuts `n`:
          - If the man is currently carrying a usable spanner: The cost is the distance
            from the man's current location to the nut's location plus 1 (for the tighten action).
            The best nut in this case is the closest one.
          - If the man is *not* currently carrying a usable spanner: The cost involves
            first getting a usable spanner from the ground, then going to the nut.
            The cost for a specific nut `n` and an available usable spanner `s` is:
            `dist(man_loc, spanner_loc) + 1 (pickup) + dist(spanner_loc, nut_loc) + 1 (tighten)`.
            The best option is the minimum of this cost over all available usable spanners `s`.
          - The overall minimum cost for this step is the minimum of the costs from
            the "carrying" option (if applicable) and the "get spanner" option.
       c. Add the minimum cost found to the total heuristic value.
       d. Update the state for the next iteration of the heuristic calculation:
          - The man's location becomes the location of the nut just tightened.
          - The tightened nut is removed from the set of loose goal nuts.
          - The spanner used is removed from the set of usable spanners (either carried or available).
          - If a carried spanner was used, the man is no longer carrying *that specific* usable spanner.
            If an available spanner was used, it was picked up and used, and is now unusable.
            The heuristic only needs to track which spanners remain usable and where the man is.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and goal information.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Needed to identify the man object

        # Build the location graph from link facts
        self.location_graph = defaultdict(set)
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                self.location_graph[loc1].add(loc2)
                self.location_graph[loc2].add(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Precompute all-pairs shortest paths (BFS from each location)
        self.distances = {}
        # Ensure all locations mentioned in initial state and goals are included, even if isolated
        all_mentioned_locations = set(self.all_locations)
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                  all_mentioned_locations.add(get_parts(fact)[2])
        for goal in self.goals:
             if match(goal, "at", "*", "*"): # Goals might involve 'at' for locatables other than man
                  all_mentioned_locations.add(get_parts(goal)[2])

        for start_loc in list(all_mentioned_locations):
             # Only compute BFS if the location is part of the linked graph, otherwise distances remain empty/inf
             if start_loc in self.location_graph:
                self.distances[start_loc] = self._bfs(start_loc)
             else:
                 # If a location is isolated, it's unreachable from any linked location
                 self.distances[start_loc] = {}


        # Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Identify the man object from the initial state (fragile assumption)
        self.man_obj = None
        potential_men = set()
        for fact in initial_state:
             if match(fact, "carrying", "*", "*"):
                 potential_men.add(get_parts(fact)[1])
             elif match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # Assume the man object name contains 'man' or 'bob'
                 if 'bob' in obj.lower() or 'man' in obj.lower():
                      potential_men.add(obj)

        if potential_men:
             # Pick one deterministically (e.g., alphabetically)
             self.man_obj = sorted(list(potential_men))[0] if potential_men else None

        if self.man_obj is None:
             # Last resort: just find any object at a location in the initial state
             for fact in initial_state:
                 if match(fact, "at", "*", "*"):
                     self.man_obj = get_parts(fact)[1]
                     break

        if self.man_obj is None:
             print("Warning: Could not identify man object from initial state.")


    def _bfs(self, start_node):
        """Compute shortest distances from start_node to all reachable nodes within the linked graph."""
        distances = {start_node: 0}
        queue = deque([start_node])
        # No need for q_set if checking `in distances`

        while queue:
            current_node = queue.popleft()
            dist = distances[current_node]

            # Only explore neighbors if the current node is part of the linked graph
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if neighbor not in distances: # Check if visited
                        distances[neighbor] = dist + 1
                        queue.append(neighbor)

        return distances

    def get_distance(self, loc1, loc2):
        """Get the precomputed shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Check if loc1 was a starting point for BFS and if loc2 was reached
        return self.distances.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Find man's current location
        loc_m = None
        if self.man_obj:
            for fact in state:
                if match(fact, "at", self.man_obj, "*"):
                    loc_m = get_parts(fact)[2]
                    break

        if loc_m is None:
             # Man object not found in state 'at' predicate or man_obj wasn't identified.
             # If goal is not reached, this state is problematic/unsolvable.
             return float('inf')

        # Identify loose goal nuts, usable spanners, carried spanners, and locations
        loose_goal_nuts = set()
        loc_map = {} # Map object to its location
        usable_spanners_in_state = set() # All usable spanners mentioned in state
        carried_spanners_in_state = set() # All spanners man is carrying

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1:]
                loc_map[obj] = loc
            elif parts[0] == "loose":
                nut = parts[1]
                if nut in self.goal_nuts:
                    loose_goal_nuts.add(nut)
            elif parts[0] == "usable":
                spanner = parts[1]
                usable_spanners_in_state.add(spanner)
            elif parts[0] == "carrying":
                 m, s = parts[1:]
                 if m == self.man_obj:
                     carried_spanners_in_state.add(s)

        # If no loose goal nuts, goal should be reached (checked at start)

        h = 0
        current_loc_m = loc_m
        # Keep track of usable spanners available *not* being carried
        current_usable_spanners_available = set(usable_spanners_in_state) - carried_spanners_in_state
        # Keep track of usable spanners *being* carried
        current_usable_spanners_carried = set(usable_spanners_in_state) & carried_spanners_in_state

        # Process nuts sequentially in the heuristic calculation
        # Sort nuts for deterministic heuristic value
        current_loose_goal_nuts = sorted(list(loose_goal_nuts))

        while current_loose_goal_nuts:
            # Check if any usable spanner exists (carried or available)
            if not current_usable_spanners_carried and not current_usable_spanners_available:
                 # Cannot tighten remaining nuts if no usable spanners left/carried
                 return float('inf')

            # Find the best next nut to tighten and the spanner to use for this step
            best_nut = None
            best_spanner_for_step = None # The spanner object to be used in this step
            min_cost_this_step = float('inf')

            # Option 1: Use a usable spanner currently carried
            if current_usable_spanners_carried:
                # Man can go directly to a nut
                for nut in current_loose_goal_nuts:
                    nut_loc = loc_map.get(nut)
                    if nut_loc is None: continue # Nut location unknown? Problematic state.

                    dist_to_nut = self.get_distance(current_loc_m, nut_loc)
                    if dist_to_nut == float('inf'): continue # Cannot reach this nut

                    cost_this_step = dist_to_nut + 1 # walk + tighten
                    if cost_this_step < min_cost_this_step:
                        min_cost_this_step = cost_this_step
                        best_nut = nut
                        # Pick any usable carried spanner to use (e.g., the first one alphabetically)
                        best_spanner_for_step = sorted(list(current_usable_spanners_carried))[0]


            # Option 2: Man needs to get a usable spanner first (if Option 1 wasn't possible or wasn't cheaper)
            if current_usable_spanners_available:
                 for nut in current_loose_goal_nuts:
                     nut_loc = loc_map.get(nut)
                     if nut_loc is None: continue

                     # Find the best available spanner for this specific nut from current location
                     best_available_spanner_for_this_nut = None
                     min_cost_to_get_spanner_and_reach_nut = float('inf')

                     for spanner in current_usable_spanners_available:
                         spanner_loc = loc_map.get(spanner)
                         if spanner_loc is None: continue

                         dist_to_spanner = self.get_distance(current_loc_m, spanner_loc)
                         if dist_to_spanner == float('inf'): continue

                         cost_to_get_spanner = dist_to_spanner + 1 # walk + pickup
                         dist_from_spanner_to_nut = self.get_distance(spanner_loc, nut_loc)
                         if dist_from_spanner_to_nut == float('inf'): continue

                         total_cost_for_this_spanner = cost_to_get_spanner + dist_from_spanner_to_nut + 1 # + tighten

                         if total_cost_for_this_spanner < min_cost_to_get_spanner_and_reach_nut:
                             min_cost_to_get_spanner_and_reach_nut = total_cost_for_this_spanner
                             best_available_spanner_for_this_nut = spanner

                     # Compare the best cost found for this nut (by getting an available spanner)
                     # with the overall minimum cost found so far for this step.
                     if min_cost_to_get_spanner_and_reach_nut < min_cost_this_step:
                          min_cost_this_step = min_cost_to_get_spanner_and_reach_nut
                          best_nut = nut
                          best_spanner_for_step = best_available_spanner_for_this_nut # This is the spanner to use

            # If we couldn't find any reachable nut with any available/carried spanner
            if best_nut is None or min_cost_this_step == float('inf'):
                 return float('inf')

            # Add the cost for the chosen best step
            h += min_cost_this_step

            # Update state for the next iteration of heuristic calculation
            current_loc_m = loc_map.get(best_nut) # Man is now at the nut's location
            current_loose_goal_nuts.remove(best_nut)

            # The spanner used in this step becomes unusable.
            # Remove it from whichever set it came from.
            if best_spanner_for_step in current_usable_spanners_carried:
                 current_usable_spanners_carried.remove(best_spanner_for_step)
            elif best_spanner_for_step in current_usable_spanners_available:
                 current_usable_spanners_available.remove(best_spanner_for_step)
            # Man is no longer carrying *this specific* usable spanner for the next step calculation,
            # even though the PDDL doesn't remove the 'carrying' fact. The heuristic simulates
            # that this spanner is now unusable and won't be considered again. Other carried
            # spanners (if any) remain in current_usable_spanners_carried if they were usable.


        return h
