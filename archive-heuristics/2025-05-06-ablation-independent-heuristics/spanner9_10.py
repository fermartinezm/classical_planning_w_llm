from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def parse_fact(fact_str):
    """Parses a PDDL fact string into a list of strings."""
    # Remove surrounding brackets and split by space
    # Handle potential extra spaces
    parts = fact_str[1:-1].split()
    return parts

# Helper function for BFS
def bfs(start_node, graph):
    """Computes shortest path distances from start_node in a graph."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Ensure current_node is a valid key in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal (tighten all required nuts) by summing
    the estimated costs for tightening each remaining loose nut in the goal set.
    The cost for tightening a single nut is estimated based on the travel
    required for the man to reach the nut's location and acquire/use a usable
    spanner. It greedily selects the nearest nut and the most convenient
    spanner (carried or nearest on the ground) in a simulated sequence.
    Shortest path distances between locations are precomputed using BFS.

    Assumptions:
    - There is exactly one man.
    - The goal only consists of (tightened nut) predicates.
    - The initial state is solvable (i.e., there are enough usable spanners
      in total across all locations to tighten all goal nuts).
    - The location graph defined by 'link' predicates is undirected (although
      PDDL links are typically bidirectional, the provided domain doesn't
      explicitly define inverse links, so we assume bidirectionality for BFS).
    - Objects typed as 'man', 'spanner', 'nut', 'location' are identifiable
      from initial state facts and goal facts based on predicate names ('at',
      'carrying', 'usable', 'loose', 'tightened', 'link'). Specifically,
      the man is attempted to be identified from 'carrying' or 'at' facts
      in the initial state.

    Heuristic Initialization:
    - Parses static 'link' facts to build the location graph.
    - Identifies all unique locations mentioned in static links and initial state 'at' facts.
    - Computes all-pairs shortest paths between locations using BFS, storing
      distances in a dictionary `self.location_distances`.
    - Extracts the set of goal nuts from the task's goal literals.
    - Attempts to identify the man's name from initial state facts.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify the set of nuts that are in the goal state and are currently loose.
        If this set is empty, the heuristic is 0 (goal reached).
    2.  Identify the man's current location.
    3.  Identify if the man is currently carrying a usable spanner.
    4.  Identify the locations of all currently usable spanners that are on the ground.
    5.  Count the number of nuts to tighten (`k`) and the number of currently usable spanners (`m`).
    6.  If `k > m`, return infinity, as there aren't enough usable spanners to tighten all required nuts from this state.
    7.  Initialize the heuristic value `h_value` to 0.
    8.  Initialize the current man location to the actual man location from the state.
    9.  Initialize the set of available usable spanners on the ground.
    10. Initialize the list of nuts to process (the loose goal nuts).
    11. Create dictionaries mapping nuts and spanners to their current locations based on the state.
    12. While there are nuts left to process:
        a.  Select the nut `n_best` from the remaining nuts that is closest to the current man location using precomputed distances.
        b.  Get the location of `n_best`, `loc_n_best`.
        c.  If the man is currently carrying a usable spanner:
            i.  Calculate the cost to walk to `loc_n_best` and tighten the nut: `dist(current_man_loc, loc_n_best) + 1`.
            ii. Add this cost to `h_value`.
            iii. Update `current_man_loc` to `loc_n_best`.
            iv. The carried spanner becomes unusable, so set the carried spanner status to None.
        d.  If the man is not currently carrying a usable spanner:
            i.  Find the usable spanner `s_best` from the available spanners on the ground whose location `loc_s_best` minimizes the cost to acquire it and use it for `n_best`: `dist(current_man_loc, loc_s_best) + 1 (pickup) + dist(loc_s_best, loc_n_best) + 1 (tighten)`.
            ii. If no usable spanners are available on the ground or reachable, return infinity.
            iii. Add the minimum cost found to `h_value`.
            iv. Update `current_man_loc` to `loc_n_best`.
            v. Remove `s_best` from the set of available usable spanners on the ground.
        e.  Remove `n_best` from the list of nuts to process.
    13. Return the final `h_value`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goal_nuts = set()
        self.locations = set()
        self.location_graph = {}
        self.location_distances = {}
        self.man_name = None

        # Extract goal nuts
        for goal_fact_str in task.goals:
            parts = parse_fact(goal_fact_str)
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # Build location graph and find all locations from static links
        for fact_str in task.static:
            parts = parse_fact(fact_str)
            if parts[0] == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.location_graph.setdefault(loc1, []).append(loc2)
                # Assuming links are bidirectional
                self.location_graph.setdefault(loc2, []).append(loc1)

        # Find all locations and the man's name from initial state 'at' facts
        # Identify man name: Look for 'carrying' first, then 'at' if not found
        for fact_str in task.initial_state:
             parts = parse_fact(fact_str)
             if parts[0] == 'carrying' and len(parts) == 3:
                  self.man_name = parts[1] # Man is the first argument of 'carrying'
                  break # Found the man

        if self.man_name is None:
             # If not found via 'carrying', look for an 'at' fact where the object
             # is not a spanner or nut based on other initial state facts.
             initial_spanners = {parse_fact(f)[1] for f in task.initial_state if f.startswith('(usable ')}
             initial_nuts = {parse_fact(f)[1] for f in task.initial_state if f.startswith('(loose ')} | {n for n in self.goal_nuts if '(tightened {})'.format(n) not in task.initial_state}

             for fact_str in task.initial_state:
                  parts = parse_fact(fact_str)
                  if parts[0] == 'at' and len(parts) == 3:
                       item, loc = parts[1], parts[2]
                       self.locations.add(loc)
                       if item not in initial_spanners and item not in initial_nuts:
                            self.man_name = item
                            break # Found the man

        # Add any locations from initial state 'at' facts that weren't in links
        for fact_str in task.initial_state:
             parts = parse_fact(fact_str)
             if parts[0] == 'at' and len(parts) == 3:
                  self.locations.add(parts[2])


        # Ensure all locations from graph are in the set
        for loc in list(self.location_graph.keys()): # Iterate over a copy as we might add keys
             self.locations.add(loc)
             for neighbor in self.location_graph[loc]:
                  self.locations.add(neighbor)

        # Ensure all locations are in the graph dictionary even if they have no links
        for loc in self.locations:
             self.location_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths
        for start_loc in self.locations:
            self.location_distances[start_loc] = bfs(start_loc, self.location_graph)

    def get_distance(self, loc1, loc2):
        """Retrieves precomputed distance between two locations."""
        # Return infinity if either location is unknown or unreachable
        if loc1 not in self.location_distances or loc2 not in self.location_distances.get(loc1, {}):
             return float('inf')
        return self.location_distances[loc1][loc2]


    def __call__(self, node):
        state = node.state

        # Parse state to get dynamic information
        nuts_to_tighten = set()
        loose_nuts_locs = {} # Map nut to location
        spanner_locs = {} # Map spanner to location
        usable_spanners = set() # All usable spanners (carried or on ground)
        man_loc = None
        carried_spanner = None # Store spanner name if carried

        for fact_str in state:
            parts = parse_fact(fact_str)
            if parts[0] == 'loose' and len(parts) == 2 and parts[1] in self.goal_nuts:
                nuts_to_tighten.add(parts[1])
            elif parts[0] == 'at' and len(parts) == 3:
                item, loc = parts[1], parts[2]
                if item == self.man_name:
                    man_loc = loc
                # Assume anything else at a location is a spanner or nut
                # (based on domain structure and typical instances)
                # Check if it's a nut (either in goal, or currently loose/tightened)
                elif item in self.goal_nuts or '(loose {})'.format(item) in state or '(tightened {})'.format(item) in state:
                     # Only store location for loose goal nuts that need processing
                     if item in nuts_to_tighten:
                          loose_nuts_locs[item] = loc
                else: # Assume it's a spanner
                     spanner_locs[item] = loc
            elif parts[0] == 'usable' and len(parts) == 2:
                 usable_spanners.add(parts[1])
            elif parts[0] == 'carrying' and len(parts) == 3 and parts[1] == self.man_name:
                 carried_spanner = parts[2]

        # Filter usable spanners to get those on the ground
        usable_spanners_on_ground = {s for s in usable_spanners if s in spanner_locs}

        # Check if carried spanner is usable
        if carried_spanner is not None and carried_spanner not in usable_spanners:
             carried_spanner = None # Man is carrying an unusable spanner

        k = len(nuts_to_tighten)
        m = len(usable_spanners_on_ground) + (1 if carried_spanner else 0) # Total currently usable spanners

        # 6. Check solvability based on usable spanners
        if k == 0:
            return 0 # Goal reached

        # If we need k nuts tightened, we need k usable spanners *in total* throughout the plan.
        # If at the current state, the number of *currently* usable spanners (on ground + carried)
        # is less than the number of nuts still needing tightening, it's a dead end
        # unless there's a way to make spanners usable again (not in this domain).
        # So, if k > m, it's an unsolvable state from here.
        if k > m:
             return float('inf')

        # 7. Initialize heuristic value
        h_value = 0
        current_man_loc = man_loc
        current_carried_spanner = carried_spanner
        available_spanners_on_ground_set = set(usable_spanners_on_ground) # Use a mutable set

        # 10. List of nuts to process
        nuts_to_process = list(nuts_to_tighten) # Use a list to remove elements

        # Ensure all nuts in nuts_to_process have a location in loose_nuts_locs
        # (They should, based on PDDL structure, but defensive check)
        nuts_to_process = [n for n in nuts_to_process if n in loose_nuts_locs]

        # 12. While loop
        while nuts_to_process:
            # a. Select the nut closest to the current man location
            best_nut = None
            min_dist_to_nut = float('inf')

            # If man_loc is None (shouldn't happen in valid states but defensive),
            # or if no remaining nut location is reachable, return inf.
            if current_man_loc is None:
                 return float('inf')

            reachable_nuts = []
            for nut in nuts_to_process:
                 loc_n = loose_nuts_locs.get(nut) # Use .get for safety
                 if loc_n is not None and loc_n in self.locations: # Ensure nut location is known and in graph
                      dist_to_nut = self.get_distance(current_man_loc, loc_n)
                      if dist_to_nut != float('inf'):
                           reachable_nuts.append((nut, dist_to_nut))


            if not reachable_nuts:
                 # No remaining nuts are at known locations or reachable
                 # If there are nuts to process but none are reachable, it's a dead end
                 return float('inf')

            best_nut, min_dist_to_nut = min(reachable_nuts, key=lambda item: item[1])

            # min_dist_to_nut is guaranteed not to be inf here because we filtered reachable_nuts

            loc_n_best = loose_nuts_locs[best_nut]

            # b. If man is carrying a usable spanner
            if current_carried_spanner is not None:
                cost = min_dist_to_nut + 1 # walk + tighten
                h_value += cost
                current_man_loc = loc_n_best
                current_carried_spanner = None # Spanner becomes unusable
            # c. If man is not carrying a usable spanner
            else:
                best_spanner_on_ground = None
                min_spanner_cost = float('inf')

                # Find the best usable spanner on the ground
                # Consider only spanners whose locations are known
                available_spanners_with_locs = [(s, spanner_locs[s]) for s in available_spanners_on_ground_set if s in spanner_locs]

                if not available_spanners_with_locs:
                     # Need a spanner but none are available on the ground (or their locs are unknown)
                     # This state is likely unsolvable if k > 0 and no carried spanner.
                     # This should be caught by k > m check, but this is a more specific check.
                     return float('inf')

                reachable_spanners = []
                for spanner, loc_s in available_spanners_with_locs:
                    # Cost to get spanner and use it for the current best nut
                    # dist(man_loc, loc_s) + 1 (pickup) + dist(loc_s, loc_n_best) + 1 (tighten)
                    dist_man_to_s = self.get_distance(current_man_loc, loc_s)
                    dist_s_to_nut = self.get_distance(loc_s, loc_n_best)

                    if dist_man_to_s != float('inf') and dist_s_to_nut != float('inf'):
                         cost = dist_man_to_s + 1 + dist_s_to_nut + 1
                         reachable_spanners.append((spanner, cost))


                # If no usable spanner on the ground is reachable
                if not reachable_spanners:
                    return float('inf')

                best_spanner_on_ground, min_spanner_cost = min(reachable_spanners, key=lambda item: item[1])


                h_value += min_spanner_cost
                current_man_loc = loc_n_best # Man ends up at the nut location
                available_spanners_on_ground_set.remove(best_spanner_on_ground) # Spanner is used

            # d. Remove the processed nut
            nuts_to_process.remove(best_nut)

        # 13. Return final heuristic value
        return h_value
