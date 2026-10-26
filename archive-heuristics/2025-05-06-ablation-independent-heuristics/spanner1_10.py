import math
from collections import deque
from heuristics.heuristic_base import Heuristic
from task import Operator, Task # Assuming Task and Operator are available

# Helper function to parse a fact string like '(at bob shed)'
def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple."""
    # Remove parentheses and split by space
    parts = fact_str[1:-1].split()
    return tuple(parts)

# Helper function for Breadth-First Search
def bfs(start_node, graph):
    """Computes shortest path distances from start_node in an unweighted graph."""
    distances = {node: float('inf') for node in graph}
    if start_node in graph: # Ensure start_node is a valid graph node
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in graph: # Check if node has neighbors (important for isolated nodes)
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

    return distances

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    the estimated minimum cost for each loose goal nut to be tightened.
    The estimated cost for a single loose nut includes the cost of the
    tighten action itself, plus the estimated travel and pickup costs
    required to get the man to the nut's location carrying a usable spanner.
    This sum is an overestimate as it doesn't account for shared travel
    or spanner pickups for multiple nuts, but it aims to provide a more
    informative estimate than simple goal counting or relaxation heuristics
    by incorporating domain-specific actions and state properties (locations,
    carrying spanners, usable spanners).

    Assumptions:
    - There is exactly one man object in the domain. The heuristic attempts
      to identify the man by finding the object in an initial state `(at ?obj ?loc)`
      fact that is not identified as a nut or spanner.
    - Nut locations are static (do not change during planning).
    - Link facts define an undirected graph of locations.
    - The goal is always a conjunction of `(tightened ?n)` facts for specific nuts.
    - Spanners become unusable after one use (`tighten_nut` effect).
    - The heuristic assumes the man can only effectively use one spanner per tighten action.
    - The heuristic assumes the man can only carry one spanner at a time for the purpose of estimating pickup costs (the domain doesn't explicitly forbid carrying multiple, but the `carrying` predicate structure and `tighten_nut` effects imply single-spanner use).

    Heuristic Initialization:
    1. Parses all location names and link connections from the static facts
       to build an adjacency list representation of the location graph.
    2. Computes all-pairs shortest path distances between all locations using BFS,
       storing the results in a dictionary `self.distances`.
    3. Identifies the name of the man object from the initial state.
    4. Identifies the names of all goal nuts from the goal state.
    5. Records the initial location of each goal nut.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state `s`:
    1. Identify the man's current location in state `s`.
    2. Determine if the man is currently carrying a usable spanner.
    3. Identify all goal nuts that are still loose in state `s`. If none are loose, the heuristic is 0.
    4. Identify all usable spanners that are currently at specific locations (not being carried) in state `s`.
    5. Check if there are enough usable spanners available (carried or at locations) to tighten all loose goal nuts. If not, the goal is unreachable from this state, and the heuristic returns infinity.
    6. Initialize the total heuristic value `h` to 0.
    7. For each loose goal nut `n` at its location `l_n`:
        a. Estimate the cost to get the man to location `l_n` carrying a usable spanner, plus the cost of the `tighten_nut` action (which is 1).
        b. This cost depends on the man's current location (`M_loc`), whether he is carrying a usable spanner (`Carried_Usable`), and the locations of available usable spanners (`Available_Spanner_Locations`).
        c. Calculate the minimum cost `cost_n` to achieve the preconditions for tightening nut `n` and perform the tighten action:
           - If man is at `l_n` AND carrying usable spanner: `cost_n = 1` (just the tighten action).
           - If man is at `l_n` BUT NOT carrying usable spanner: Man needs to pick up a spanner from the nearest available location `l_s`, then tighten. `cost_n = min(dist(l_n, l_s) + 1 for l_s in Available_Spanner_Locations) + 1`.
           - If man is NOT at `l_n` BUT carrying usable spanner: Man needs to walk to `l_n`, then tighten. `cost_n = dist(M_loc, l_n) + 1`.
           - If man is NOT at `l_n` AND NOT carrying usable spanner: Man needs to walk to the nearest available spanner location `l_s`, pick it up, walk from `l_s` to `l_n`, then tighten. `cost_n = min(dist(M_loc, l_s) + 1 + dist(l_s, l_n) for l_s in Available_Spanner_Locations) + 1`.
           - Note: If `Available_Spanner_Locations` is empty when a spanner is needed, this implies unreachability, which should ideally be caught by the earlier check, but the min calculation will yield infinity.
        d. Add `cost_n` to the total heuristic value `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task

        # --- Parse static information and build graph ---
        self.locations = set()
        self.graph = {} # Adjacency list for locations
        self.nut_names_all = set() # All nut names mentioned in init/goal
        self.spanner_names_all = set() # All spanner names mentioned in init
        self.locatable_names_all = set() # All locatable names mentioned in init (man, nuts, spanners)

        # Collect locations and links
        for fact_str in task.static:
            fact = parse_fact(fact_str)
            if fact[0] == 'link':
                l1, l2 = fact[1], fact[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1) # Links are bidirectional

        # Ensure all locations mentioned in initial state/goals are in graph nodes
        for fact_str in task.initial_state | task.goals:
             fact = parse_fact(fact_str)
             if fact[0] == 'at':
                 obj, loc = fact[1], fact[2]
                 self.locations.add(loc)
                 self.graph.setdefault(loc, set()) # Add location node even if no links yet
                 self.locatable_names_all.add(obj)
             elif fact[0] in ('loose', 'tightened'):
                 nut = fact[1]
                 self.nut_names_all.add(nut)
                 self.locatable_names_all.add(nut)
             elif fact[0] == 'usable':
                 spanner = fact[1]
                 self.spanner_names_all.add(spanner)
                 self.locatable_names_all.add(spanner)
             elif fact[0] == 'carrying':
                 man, spanner = fact[1], fact[2]
                 self.locatable_names_all.add(man)
                 self.locatable_names_all.add(spanner)


        # Compute all-pairs shortest paths
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = bfs(loc, self.graph)

        # --- Parse initial state information ---
        self.nut_locations = {} # Initial location of each nut
        initial_locatables_at_loc = set() # Locatables explicitly placed at a location in init

        for fact_str in task.initial_state:
            fact = parse_fact(fact_str)
            if fact[0] == 'at':
                obj, loc = fact[1], fact[2]
                initial_locatables_at_loc.add(obj)
                if obj in self.nut_names_all:
                    self.nut_locations[obj] = loc

        # Identify man name (assume the locatable in initial state 'at' fact that isn't a nut or spanner)
        man_candidates = initial_locatables_at_loc - self.nut_names_all - self.spanner_names_all
        if len(man_candidates) == 1:
             self.man_name = list(man_candidates)[0]
        else:
             # Fallback: Find the object carrying a spanner in initial state
             man_candidates = set()
             for fact_str in task.initial_state:
                 fact = parse_fact(fact_str)
                 if fact[0] == 'carrying':
                     man_candidates.add(fact[1])

             if len(man_candidates) == 1:
                  self.man_name = list(man_candidates)[0]
             else:
                  # As a last resort, assume 'bob' based on examples if no other man candidate found
                  if not man_candidates and not initial_locatables_at_loc:
                       self.man_name = 'bob' # Handle case where man might not be in (at) or (carrying) in init
                  elif initial_locatables_at_loc:
                       # If still ambiguous, pick one from initial_locatables_at_loc that isn't a known nut/spanner
                       for obj in initial_locatables_at_loc:
                           if obj not in self.nut_names_all and obj not in self.spanner_names_all:
                               self.man_name = obj
                               break
                       if not hasattr(self, 'man_name'):
                            self.man_name = 'bob' # Final fallback
                  else:
                       self.man_name = 'bob' # Final fallback


        # --- Parse goal information ---
        self.goal_nut_names = set()
        for fact_str in task.goals:
            fact = parse_fact(fact_str)
            if fact[0] == 'tightened':
                self.goal_nut_names.add(fact[1])

        # Ensure we know the location of all goal nuts (should be in initial state)
        # If a goal nut's location isn't in initial state, it's likely an invalid problem
        # or the nut moves (not in this domain). We'll rely on self.nut_locations.get(nut) later.


    def __call__(self, node):
        state = node.state

        # 1. Identify man's current location
        man_loc = None
        for fact_str in state:
            fact = parse_fact(fact_str)
            if fact[0] == 'at' and fact[1] == self.man_name:
                man_loc = fact[2]
                break

        if man_loc is None:
             # Man's location is unknown, likely an unreachable state or invalid state representation
             return float('inf')

        # 2. Determine if man is currently carrying a usable spanner
        carrying_usable_spanner = False
        carried_spanner = None
        for fact_str in state:
            fact = parse_fact(fact_str)
            if fact[0] == 'carrying' and fact[1] == self.man_name:
                carried_spanner = fact[2]
                break
        if carried_spanner and f'(usable {carried_spanner})' in state:
             carrying_usable_spanner = True

        # 3. Identify loose goal nuts
        loose_goal_nuts = {nut for nut in self.goal_nut_names if f'(loose {nut})' in state}

        # 4. Identify usable spanners at locations
        usable_spanners_at_locs = {} # {spanner_name: location}
        for fact_str in state:
            fact = parse_fact(fact_str)
            # Check if it's an 'at' fact for a known spanner
            if fact[0] == 'at' and fact[1] in self.spanner_names_all:
                 spanner_name = fact[1]
                 spanner_loc = fact[2]
                 # Check if the spanner is usable
                 if f'(usable {spanner_name})' in state:
                      usable_spanners_at_locs[spanner_name] = spanner_loc

        available_spanner_locations = set(usable_spanners_at_locs.values())

        # 5. Check if goal reached
        if not loose_goal_nuts:
            return 0

        # 6. Check reachability (enough usable spanners)
        num_usable_spanners_available = len(usable_spanners_at_locs) + (1 if carrying_usable_spanner else 0)
        if num_usable_spanners_available < len(loose_goal_nuts):
             return float('inf')

        # 7. Calculate heuristic by summing estimated costs for each loose nut
        h = 0
        for nut in loose_goal_nuts:
            nut_loc = self.nut_locations.get(nut)
            if nut_loc is None:
                 # Location of a loose goal nut is unknown, problem is likely unsolvable
                 return float('inf')

            # Cost to tighten this specific nut
            cost_n = 1 # Cost of the tighten_nut action itself

            # Cost to get man to nut_loc carrying a usable spanner
            if man_loc == nut_loc and carrying_usable_spanner:
                # Already at location with spanner
                pass # cost_n += 0
            elif man_loc == nut_loc and not carrying_usable_spanner:
                # At location, but need spanner. Go get nearest one and come back.
                if not available_spanner_locations: return float('inf') # No spanners to pick up
                min_dist_to_spanner_from_nut = float('inf')
                for s_loc in available_spanner_locations:
                     if nut_loc in self.distances and s_loc in self.distances[nut_loc]:
                          min_dist_to_spanner_from_nut = min(min_dist_to_spanner_from_nut, self.distances[nut_loc][s_loc])
                if min_dist_to_spanner_from_nut == float('inf'): return float('inf') # Cannot reach any spanner from nut_loc
                cost_n += min_dist_to_spanner_from_nut + 1 # travel + pickup
                # Note: This assumes the man returns to nut_loc after picking up.

            elif man_loc != nut_loc and carrying_usable_spanner:
                # Not at location, but have spanner. Just need to travel to nut.
                if man_loc not in self.distances or nut_loc not in self.distances[man_loc]: return float('inf') # Cannot reach nut
                cost_n += self.distances[man_loc][nut_loc] # travel

            elif man_loc != nut_loc and not carrying_usable_spanner:
                # Not at location, need spanner. Go get nearest spanner, then go to nut.
                if not available_spanner_locations: return float('inf') # No spanners to pick up
                min_travel_pickup_and_reach_nut = float('inf')
                for s_loc in available_spanner_locations:
                     if man_loc in self.distances and s_loc in self.distances[man_loc] and nut_loc in self.distances[s_loc]:
                          travel_pickup_reach = self.distances[man_loc][s_loc] + 1 + self.distances[s_loc][nut_loc]
                          min_travel_pickup_and_reach_nut = min(min_travel_pickup_and_reach_nut, travel_pickup_reach)
                if min_travel_pickup_and_reach_nut == float('inf'): return float('inf') # Cannot reach spanner or nut
                cost_n += min_travel_pickup_and_reach_nut

            h += cost_n

        return h
