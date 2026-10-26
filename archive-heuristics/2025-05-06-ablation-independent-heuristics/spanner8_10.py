from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available
from collections import deque # For BFS queue
import math # For float('inf')

# Helper function to parse a PDDL fact string
def parse_fact(fact_str):
    """
    Parses a PDDL fact string like '(predicate arg1 arg2)' into a tuple
    (predicate, [arg1, arg2]).
    """
    # Removes outer parentheses and splits by spaces
    # Handles cases like '(at spanner1 location1)' -> ['at', 'spanner1', 'location1']
    # Handles cases like '(tightened nut1)' -> ['tightened', 'nut1']
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal (tighten all loose nuts) by summing:
    1. The number of loose nuts (representing tighten actions).
    2. The number of spanners that need to be picked up.
    3. The shortest distance from the man's current location to the nearest
       location where a useful action can be performed (either picking up
       a spanner if needed, or tightening a nut).

    Assumptions:
    - The domain follows the PDDL structure provided.
    - Nut locations are static and provided in the static facts as '(at nut_name loc_name)'.
    - Spanners become unusable after one use (implied by the effect of tighten_nut).
    - The man object is named 'bob' (based on example instances). A more robust
      implementation would identify the man object dynamically from the task definition.
    - The location graph defined by 'link' predicates is connected for relevant locations,
      or locations not connected by links are handled as isolated nodes in distance calculations.

    Heuristic Initialization:
    - Parses static facts to build a graph of locations based on 'link' predicates.
    - Identifies the static locations of all nuts mentioned in the static facts.
    - Computes all-pairs shortest paths between locations that are part of the
      static link graph or are static nut locations using BFS and stores them
      in a distance map (self.dist_map). Locations appearing later in states
      (like the man's initial location if isolated) are handled on the fly in __call__.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the goal is already reached (no loose nuts). If so, return 0.
    2. Identify the man's current location from the state by finding the fact '(at bob ?l)'.
    3. Identify which usable spanners the man is carrying from the state by finding facts '(carrying bob ?s)' and '(usable ?s)'.
    4. Identify which usable spanners are at specific locations from the state by finding facts '(at ?s ?l)' and '(usable ?s)' where the spanner is not carried.
    5. Count the total number of loose nuts (N_loose) by checking which goal nuts
       are not yet tightened in the state (i.e., goal fact '(tightened nut_name)' is not in state).
    6. Count the number of usable spanners the man is carrying (N_carried).
    7. Count the total number of usable spanners available (carried or at locations) (N_usable).
    8. If the number of loose nuts exceeds the total number of usable spanners,
       the problem is unsolvable from this state, return infinity.
    9. Calculate the number of spanners that still need to be picked up (N_pickups = max(0, N_loose - N_carried)).
    10. If pickups are needed (N_pickups > 0) but there are no usable spanners available at any location,
        the problem is unsolvable from this state, return infinity.
    11. Initialize the base heuristic value as N_loose (for tighten actions) + N_pickups (for pickup actions).
    12. Determine the set of target locations the man needs to reach. This includes
        locations of all loose nuts and, if N_pickups > 0, locations of usable spanners
        at locations.
    13. Calculate the shortest distance from the man's current location to the
        nearest location in the set of target locations using the precomputed distance map.
        If the man's current location or a target location was not part of the initial
        distance map calculation (e.g., an isolated location), compute distances from
        that location on the fly.
    14. If no target location is reachable from the man's current location and there are loose nuts,
        the state is a dead end, return infinity.
    15. Add this minimum distance to the base heuristic value.
    16. Return the total heuristic value.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Find the man's name (assuming it's 'bob' based on examples)
        # A robust way would parse the domain file or rely on task object providing this.
        self.man_name = 'bob' # !!! Assumption based on examples !!!

        # Precompute location graph and identify static nut locations
        self.location_graph = {}
        self.nut_locations = {}
        initial_locations_for_bfs = set() # Locations to start BFS from initially

        # Parse static facts
        for fact_str in self.static:
            pred, args = parse_fact(fact_str)
            if pred == 'link':
                l1, l2 = args
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Links are bidirectional
                initial_locations_for_bfs.add(l1)
                initial_locations_for_bfs.add(l2)
            elif pred == 'at' and len(args) == 2 and args[0].startswith('nut'):
                 # Assuming nut locations are static and given in static facts
                 nut, loc = args
                 self.nut_locations[nut] = loc
                 initial_locations_for_bfs.add(loc)

        # Keep track of all known locations encountered so far
        self.all_known_locations = set(initial_locations_for_bfs)

        # Compute all-pairs shortest paths for initial set of relevant locations
        self.dist_map = {}
        # Iterate over a copy because _bfs_from_start might add new locations
        for start_node in list(self.all_known_locations):
             self.dist_map[start_node] = self._bfs_from_start(start_node)


    def _bfs_from_start(self, start_node):
        """Helper to run BFS from a single start node."""
        # Ensure start_node is in the set of all known locations
        self.all_known_locations.add(start_node)

        # Initialize distances for ALL known locations at the start of this BFS run
        distances = {node: math.inf for node in self.all_known_locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()

            # Ensure u is in the graph keys before accessing neighbors
            if u in self.location_graph:
                for v in self.location_graph[u]:
                    # If we discover a new location via links, add it
                    if v not in self.all_known_locations:
                         self.all_known_locations.add(v)
                         # Add it to distances map for current BFS
                         distances[v] = math.inf # Initialize distance for the new node

                    if distances[v] == math.inf:
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances


    def get_man_location(self, state):
        """Finds the man's current location in the state."""
        # Assuming man object name is self.man_name ('bob')
        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'at' and len(args) == 2 and args[0] == self.man_name:
                 return args[1]
        return None # Man location not found (should not happen in valid states)

    def get_nut_location(self, nut_name):
        """Gets the static location of a nut."""
        # Nut locations are precomputed in __init__ from static facts
        return self.nut_locations.get(nut_name) # Returns None if nut not found (should not happen for goal nuts)


    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        loose_nuts = set()
        # Find all nuts mentioned in goals
        goal_nuts = set()
        for goal_fact in self.goals:
             pred, args = parse_fact(goal_fact)
             if pred == 'tightened' and len(args) == 1 and args[0].startswith('nut'):
                  goal_nuts.add(args[0])

        # Check which goal nuts are NOT tightened in the current state
        tightened_nuts_in_state = set()
        for fact_str in state:
             pred, args = parse_fact(fact_str)
             if pred == 'tightened' and len(args) == 1 and args[0].startswith('nut'):
                  tightened_nuts_in_state.add(args[0])

        loose_nuts = goal_nuts - tightened_nuts_in_state

        N_loose = len(loose_nuts)

        if N_loose == 0:
            return 0 # Goal reached

        # 2. Find man's location
        man_loc = self.get_man_location(state)
        if man_loc is None:
             # Should not happen in a valid state, but handle defensively
             return math.inf

        # Ensure man_loc is in the distance map if it wasn't covered in __init__
        if man_loc not in self.dist_map:
             self.dist_map[man_loc] = self._bfs_from_start(man_loc)
             # Also need to compute distances *to* man_loc from other nodes if they exist
             # This is handled by running BFS from other nodes if man_loc is reachable.
             # If man_loc is isolated, distances from other nodes to man_loc will remain inf.


        # 3. Identify usable spanners
        usable_spanners_carried = set()
        usable_spanners_at_loc = {} # {location: {spanner_name, ...}}
        # We only care about usable spanners

        # Find all usable spanners in the state
        usable_spanner_names_in_state = set()
        for fact_str in state:
             pred, args = parse_fact(fact_str)
             if pred == 'usable' and len(args) == 1 and args[0].startswith('spanner'):
                  usable_spanner_names_in_state.add(args[0])

        # Determine location/status of usable spanners
        for spanner_name in usable_spanner_names_in_state:
             is_carried = False
             # Check if carried by the man
             carrying_fact = f'(carrying {self.man_name} {spanner_name})'
             if carrying_fact in state:
                  usable_spanners_carried.add(spanner_name)
                  is_carried = True

             if not is_carried:
                  # Check if at a location
                  for fact_str in state:
                       pred, args = parse_fact(fact_str)
                       if pred == 'at' and len(args) == 2 and args[0] == spanner_name:
                            loc = args[1]
                            usable_spanners_at_loc.setdefault(loc, set()).add(spanner_name)
                            # Ensure spanner location is in distance map
                            if loc not in self.dist_map:
                                 self.dist_map[loc] = self._bfs_from_start(loc)
                            break # Found location for this spanner

        N_carried = len(usable_spanners_carried)
        N_at_loc = sum(len(s_set) for s_set in usable_spanners_at_loc.values())
        N_usable = N_carried + N_at_loc # Total usable spanners available

        # 8. Check for unsolvable state
        if N_loose > N_usable:
             return math.inf

        # 9. Calculate pickups needed
        N_pickups = max(0, N_loose - N_carried)

        # 10. Check if pickups are needed but no spanners are available at locations
        if N_pickups > 0 and not usable_spanner_locs:
             # Man needs spanners but none are available to pick up
             return math.inf

        # 11. Base cost
        base_cost = N_loose + N_pickups

        # 12. Determine target locations
        loose_nut_locs = set()
        for nut in loose_nuts:
             loc = self.get_nut_location(nut)
             if loc:
                  loose_nut_locs.add(loc)
                  # Ensure nut location is in distance map
                  if loc not in self.dist_map:
                       self.dist_map[loc] = self._bfs_from_start(loc)
             # else: nut location not found? Problematic, assume unsolvable.
             # This shouldn't happen if nuts in goals are also in static facts with locations.
             # If it happens, loose_nut_locs will be empty, leading to inf later if N_loose > 0.


        usable_spanner_locs = set(usable_spanners_at_loc.keys())

        target_locs = set(loose_nut_locs)
        if N_pickups > 0:
             target_locs.update(usable_spanner_locs)

        # Handle case where loose nuts exist but their locations weren't found
        if N_loose > 0 and not loose_nut_locs:
             # This implies a goal nut's location wasn't in static facts.
             # Problem is likely ill-defined or static parsing is wrong.
             # Treat as unsolvable.
             return math.inf


        # 13. Calculate min distance to target
        min_dist_to_target = math.inf

        # Check if man_loc is a valid key in dist_map before accessing
        if man_loc in self.dist_map:
            for target_l in target_locs:
                # Check if target_l is a valid key in the inner dict
                if target_l in self.dist_map[man_loc]:
                    min_dist_to_target = min(min_dist_to_target, self.dist_map[man_loc][target_l])
                # else: target_l is not reachable from man_loc. min_dist remains inf.
        # else: man_loc is not in dist_map. This case should be covered by adding man_loc to dist_map above.


        # 14. Check reachability
        if min_dist_to_target == math.inf and N_loose > 0:
             # Cannot reach any required location (nut or spanner) from man's current location
             return math.inf

        # 15. Add walking cost
        # If min_dist_to_target is inf, it means no target is reachable.
        # If N_loose > 0, this is a dead end, already handled.
        # If N_loose == 0, this case is skipped.
        # So, if we reach here and min_dist_to_target is inf, it must be that target_locs was empty
        # (which only happens if N_loose == 0). In that case, walking cost is 0.
        walking_cost = min_dist_to_target if min_dist_to_target != math.inf else 0

        # 16. Return total heuristic
        return base_cost + walking_cost
