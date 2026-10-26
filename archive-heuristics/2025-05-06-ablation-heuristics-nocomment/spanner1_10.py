from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Helper to parse PDDL fact string into parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if fact matches predicate and arguments (with wildcards)."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        Estimates the cost to reach the goal by summing the estimated costs
        to tighten each loose nut that is part of the goal. The cost for
        each nut is estimated sequentially: the man travels from his current
        location to the nearest available usable spanner, picks it up,
        travels to the nut's location, and tightens the nut. The man's
        location is updated after each nut is tightened in the heuristic's
        internal calculation sequence, and the used spanner is marked as
        unavailable for subsequent nuts in the calculation. Shortest path
        distances between locations are precomputed using BFS. The nuts
        are processed in increasing order of their distance from the man's
        initial location.

    Assumptions:
        - The input task is solvable.
        - Links between locations are bidirectional for travel purposes, even
          though the PDDL 'link' predicate is used in one direction in the
          'walk' action. (Standard assumption unless specified otherwise).
        - Nuts do not move from their initial locations.
        - Spanners, if not carried, are at a location specified by an '(at S L)' fact.
        - A spanner becomes unusable after one 'tighten_nut' action and cannot
          become usable again within the scope of a single problem instance.
        - The number of usable spanners in the initial state is sufficient to
          tighten all goal nuts in solvable instances.

    Heuristic Initialization:
        - Parses static facts to identify locations and links.
        - Builds an adjacency list representation of the location graph, assuming
          bidirectional links.
        - Computes all-pairs shortest paths between locations using BFS and stores
          them in a dictionary `self.dist`. If locations are disconnected, unreachable
          pairs will not have an entry, or distance will be infinity.
        - Parses the initial state (`task.initial_state`) to identify the initial
          location of the man (`self.initial_man_location`) and the initial
          locations of all nuts (`self.nut_locations`).
        - Identifies the set of nuts that are part of the goal state (`self.goal_nuts`)
          from `task.goals`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the goal is already reached by verifying if all nuts in
           `self.goal_nuts` are present as `(tightened ...)` facts in the current state.
           If so, return 0.
        2. Parse the current state (`node.state`) to extract:
           - The man's current location (`man_location`).
           - The set of spanners currently carried by the man (`carried_spanners`).
           - The set of spanners that are currently usable (`usable_spanners_in_state`).
           - A dictionary mapping spanners (not carried) to their current locations
             (`spanner_locations_in_state`).
           - The set of nuts that are currently tightened (`tightened_nuts_in_state`).
        3. Identify the set of loose nuts that are also goal nuts (`loose_goal_nuts_in_state`).
           These are nuts in `self.goal_nuts` that are not in `tightened_nuts_in_state`.
        4. If there are no `loose_goal_nuts_in_state`, the goal is effectively reached
           (or the remaining goal nuts are already tightened), return 0.
        5. Identify the set of usable spanners available for the heuristic calculation.
           This starts with all spanners in `usable_spanners_in_state`. This set will
           be modified during the calculation.
        6. If the number of `loose_goal_nuts_in_state` exceeds the number of
           `available_spanners_for_heuristic`, the state is likely unsolvable in this
           domain, return `math.inf`.
        7. Initialize the total heuristic cost `h = 0`.
        8. Initialize the man's effective location for the current step of the
           calculation to his actual location in the state (`current_man_location_for_heuristic`).
        9. Create a list of nuts to process (`nuts_to_process`) from `loose_goal_nuts_in_state`.
           Sort this list by the shortest distance from the man's initial location
           (`self.initial_man_location`) to the location of each nut.
        10. Iterate through the sorted `nuts_to_process`. For the current nut `N`:
            a. Get the location of the nut `L_N` from `self.nut_locations`.
            b. Find the minimum cost to get a usable spanner from the current
               `available_spanners_for_heuristic` pool to the man at `L_N`.
            c. Iterate through each spanner `S` in `available_spanners_for_heuristic`:
                i. Determine the spanner's actual location in the current state (`actual_L_S`).
                   If the man is carrying `S` (checked against `carried_spanners` from the state snapshot),
                   `actual_L_S` is the man's actual location (`man_location`). Otherwise, look up
                   its location in `spanner_locations_in_state`. Handle cases where location is unknown
                   (e.g., spanner not at a location and not carried, though unlikely for usable spanners).
                ii. Calculate the cost for the man to acquire spanner `S` and bring it to `L_N`,
                    starting from `current_man_location_for_heuristic`:
                    - If the man is already carrying `S` (in the actual state): Cost is
                      `dist(current_man_location_for_heuristic, L_N)` (just walk to the nut).
                    - If the man is not carrying `S`: Cost is `dist(current_man_location_for_heuristic, actual_L_S)`
                      (walk to spanner) + 1 (pickup) + `dist(actual_L_S, L_N)` (walk with spanner to nut).
                    - Use `.get(..., math.inf)` for distance lookups to handle unreachable locations.
                iii. Keep track of the minimum such cost (`min_spanner_trip_cost`) and the
                     corresponding spanner (`best_spanner`).
            d. If no usable spanner could be found that can reach the nut location (e.g., all remaining
               spanners are in disconnected components), return `math.inf`.
            e. Add `min_spanner_trip_cost + 1` (the +1 is for the `tighten_nut` action) to the
               total heuristic cost `h`.
            f. Update `current_man_location_for_heuristic` to `L_N` (the man is now effectively
               at the nut's location for the next step).
            g. Remove `best_spanner` from `available_spanners_for_heuristic` as it's considered
               used for this nut in the heuristic calculation sequence.
        11. Return the total heuristic cost `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.location_links = {}
        self.locations = set()
        self.nut_locations = {}
        self.goal_nuts = set()
        self.initial_man_location = None

        # Parse static facts for links and locations
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.location_links.setdefault(loc1, []).append(loc2)
                self.location_links.setdefault(loc2, []).append(loc1) # Assume bidirectional links
                self.locations.add(loc1)
                self.locations.add(loc2)
            # Other static facts like object types are not strictly needed for this heuristic

        # Parse initial state for initial locations of man and nuts
        for initial_fact in task.initial_state:
             initial_parts = get_parts(initial_fact)
             if initial_parts[0] == 'at':
                 obj, loc = initial_parts[1], initial_parts[2]
                 if obj.startswith('bob'): # Assuming 'bob' is the man's name
                     self.initial_man_location = loc
                 elif obj.startswith('nut'):
                     self.nut_locations[obj] = loc
             # We don't need initial spanner locations or loose/usable facts here,
             # as we get current state info in __call__.

        # Identify goal nuts from task.goals
        for goal_fact in self.goals:
            goal_parts = get_parts(goal_fact)
            if goal_parts[0] == 'tightened' and goal_parts[1].startswith('nut'):
                self.goal_nuts.add(goal_parts[1])

        # Compute all-pairs shortest paths
        self.dist = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        dist = {}
        for start_node in self.locations:
            dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            dist[start_node][start_node] = 0

            while q:
                current_loc, current_dist = q.popleft()

                if current_loc in self.location_links:
                    for neighbor in self.location_links[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            dist[start_node][neighbor] = current_dist + 1
                            q.append((neighbor, current_dist + 1))
        return dist

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        goal_reached = True
        for goal_nut in self.goal_nuts:
            if '(tightened ' + goal_nut + ')' not in state:
                goal_reached = False
                break
        if goal_reached:
            return 0

        # Parse state to get current information
        man_location = None
        carried_spanners = set()
        usable_spanners_in_state = set()
        spanner_locations_in_state = {} # For spanners NOT carried
        tightened_nuts_in_state = set()

        man_name = None # Assume only one man, find his name

        # Find the man's name first
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at' and parts[1].startswith('bob'): # Assuming 'bob' is the man's name
                 man_name = parts[1]
                 break # Found the man

        if man_name is None:
             # Man not found in state? Should not happen in valid states.
             return math.inf # Indicate error or unsolvable state

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj == man_name:
                    man_location = loc
                elif obj.startswith('spanner'):
                    spanner_locations_in_state[obj] = loc
                # Nut locations are assumed static, stored in self.nut_locations
            elif parts[0] == 'carrying':
                m, s = parts[1], parts[2]
                if m == man_name:
                     carried_spanners.add(s)
            elif parts[0] == 'usable':
                s = parts[1]
                usable_spanners_in_state.add(s)
            elif parts[0] == 'tightened':
                n = parts[1]
                tightened_nuts_in_state.add(n)

        # If man_location wasn't found, something is wrong.
        if man_location is None:
             return math.inf # Indicate error or unsolvable state


        # 3. Identify loose nuts that are also goal nuts
        loose_goal_nuts_in_state = {
            nut for nut in self.goal_nuts
            if nut not in tightened_nuts_in_state
        }

        if len(loose_goal_nuts_in_state) == 0:
             # This case should be covered by the initial goal_reached check, but good to be safe.
             return 0

        # 5. Identify usable spanners currently available for the heuristic calculation
        available_spanners_for_heuristic = set(usable_spanners_in_state)

        # 6. Check if enough usable spanners exist
        if len(loose_goal_nuts_in_state) > len(available_spanners_for_heuristic):
             return math.inf # Not enough spanners to tighten all required nuts

        # 7. Initialize heuristic cost
        h = 0

        # 8. Initialize man's effective location for calculation
        current_man_location_for_heuristic = man_location

        # 9. Create and sort nuts to process
        nuts_to_process = list(loose_goal_nuts_in_state)
        # Sort nuts by distance from the initial man location
        # Ensure initial_man_location and nut_locations are valid keys in self.dist
        try:
            nuts_to_process.sort(key=lambda n: self.dist.get(self.initial_man_location, {}).get(self.nut_locations.get(n), math.inf))
        except KeyError:
             # Should not happen if initial_man_location and nut_locations are correctly populated
             # and locations are in self.dist keys.
             return math.inf # Indicate error


        # 10. Iterate through nuts to tighten
        for nut in nuts_to_process:
            L_N = self.nut_locations.get(nut)
            if L_N is None:
                 # Location of goal nut not found? Should not happen.
                 return math.inf

            min_spanner_trip_cost = math.inf
            best_spanner = None

            # 10b. Find the best spanner for this nut from available ones
            spanners_to_check = list(available_spanners_for_heuristic) # Iterate over a copy

            for spanner in spanners_to_check:
                # 10c.i. Determine spanner's actual location in the current state
                is_carried_in_state = spanner in carried_spanners # Check against actual state
                actual_L_S = man_location if is_carried_in_state else spanner_locations_in_state.get(spanner)

                # If a usable spanner somehow doesn't have a location in the state (not carried and not at a location), skip it.
                if actual_L_S is None:
                     continue

                # 10c.ii. Calculate cost to get man+spanner to L_N, starting from current_man_location_for_heuristic
                cost_man_to_spanner = self.dist.get(current_man_location_for_heuristic, {}).get(actual_L_S, math.inf)

                if cost_man_to_spanner == math.inf:
                    # Cannot reach spanner location from current heuristic location
                    spanner_trip_cost = math.inf
                else:
                    if is_carried_in_state:
                        # Man already has the spanner. Just need to go from current_man_location_for_heuristic to L_N.
                        spanner_trip_cost = self.dist.get(current_man_location_for_heuristic, {}).get(L_N, math.inf)
                    else:
                        # Man needs to go from current_man_location_for_heuristic to actual_L_S, pick up (cost 1), go from actual_L_S to L_N.
                        cost_pickup = 1
                        cost_spanner_to_nut = self.dist.get(actual_L_S, {}).get(L_N, math.inf)

                        if cost_spanner_to_nut == math.inf:
                             # Cannot reach nut location from spanner location
                             spanner_trip_cost = math.inf
                        else:
                             spanner_trip_cost = cost_man_to_spanner + cost_pickup + cost_spanner_to_nut

                if spanner_trip_cost < min_spanner_trip_cost:
                    min_spanner_trip_cost = spanner_trip_cost
                    best_spanner = spanner

            # 10d. Check if a usable spanner was found that can reach the nut
            if best_spanner is None or min_spanner_trip_cost == math.inf:
                 # No usable spanner found that can reach the nut location from the current heuristic location.
                 # This state is likely unsolvable or requires actions not considered by the heuristic.
                 return math.inf

            # 10e. Add cost for this nut
            # The cost is the travel+pickup cost plus the tighten action cost (1)
            h += min_spanner_trip_cost + 1

            # 10f. Update man's effective location for the next nut
            current_man_location_for_heuristic = L_N

            # 10g. Remove the used spanner from the available pool for the heuristic calculation
            available_spanners_for_heuristic.remove(best_spanner)

        # 11. Return total heuristic cost
        return h
