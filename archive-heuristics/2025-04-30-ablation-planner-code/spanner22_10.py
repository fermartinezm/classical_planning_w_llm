from fnmatch import fnmatch
from collections import deque
# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy base class for standalone testing if needed
# In the actual planning environment, the Heuristic base class will be provided.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            self.objects = task.objects

        def __call__(self, node):
            raise NotImplementedError


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the fact has at least as many parts as the pattern args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It sums the number of loose goal nuts (for tighten actions), the number of
    spanners that need to be picked up, and the estimated walk distance to reach
    the first required item (either a nut or a spanner) from the man's current location.

    # Assumptions
    - Each nut requires one tighten action.
    - Each tighten action consumes one usable spanner.
    - The heuristic simplifies the walk cost by only considering the minimum distance
      to the closest required location (nut or spanner) for the initial movement.
      Subsequent movements between items are not explicitly modeled in the walk cost sum.
    - There is exactly one man object in the domain.
    - A goal nut is considered 'loose' if it is not explicitly 'tightened' in the state.

    # Heuristic Initialization
    - Identify all objects and their types from the task definition.
    - Identify all locations.
    - Build the location graph based on `link` static facts.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify the set of nuts that are part of the goal.
    - Identify the man object.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the man and his current location from the current state.
    2.  Identify all goal nuts and determine which ones are currently *not* `tightened`. Let `K` be the count of these untightened goal nuts.
    3.  If `K` is 0, the heuristic is 0 (goal state).
    4.  Identify usable spanners: those currently carried by the man and those on the ground that are `usable`. Count the total number of usable spanners available (`N_usable_total`).
    5.  If the number of untightened goal nuts `K` exceeds the total number of usable spanners, the problem is likely unsolvable in this domain (spanners are consumed), return infinity.
    6.  Initialize heuristic `h = 0`.
    7.  Add the cost for tightening each untightened goal nut: `h += K`. (Each tighten action costs 1).
    8.  Count the number of usable spanners the man is currently carrying (`N_carried_usable`).
    9.  Add the cost for picking up the necessary additional usable spanners: `h += max(0, K - N_carried_usable)`. (Assumes each pickup action acquires one spanner and costs 1).
    10. Add the estimated walk cost.
        - Find the set of locations of untightened goal nuts (`UntightenedGoalNutLocs`).
        - Find the set of locations of usable spanners on the ground (`UsableSpannerLocsOnGround`).
        - Calculate the minimum distance from the man's location to any untightened goal nut location (`min_dist_to_nut`).
        - Calculate the number of spanners that need to be picked up from the ground (`num_pickups_needed`). If `num_pickups_needed > 0` and there are usable spanners on the ground, find the minimum distance from the man's location to any usable spanner location on the ground (`min_dist_to_spanner`).
    11. Check if any required location is unreachable (distance is infinity). If so, return infinity.
    12. Add the calculated minimum distances (if reachable) to `h`.
    13. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting domain information and precomputing distances.
        """
        # Call the base class constructor if necessary, depending on the actual Heuristic base
        # super().__init__(task)
        self.goals = task.goals
        self.static = task.static
        self.objects = task.objects

        # Map object names to their types
        self.object_types = {}
        for obj_str in self.objects:
             parts = obj_str.split(' - ')
             if len(parts) == 2:
                 self.object_types[parts[0].strip()] = parts[1].strip()

        # Identify all locations
        self.locations = {
            name for name, type_name in self.object_types.items() if type_name == 'location'
        }

        # Build adjacency list for the location graph
        self.adj = {loc: [] for loc in self.locations}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations are valid and in our set
                if loc1 in self.locations and loc2 in self.locations:
                    self.adj[loc1].append(loc2)
                    self.adj[loc2].append(loc1) # Assuming links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.dist = {loc: {l: float('inf') for l in self.locations} for loc in self.locations}
        for start_node in self.locations:
            self.dist[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node}
            while queue:
                u = queue.popleft()
                # Check if u is a valid location before accessing self.adj
                if u in self.adj:
                    for v in self.adj[u]:
                        if v not in visited:
                            visited.add(v)
                            self.dist[start_node][v] = self.dist[start_node][u] + 1
                            queue.append(v)

        # Identify goal nuts
        self.goal_nuts = set()
        # task.goals is a frozenset of goal facts like '(tightened nut1)'
        for goal_fact in self.goals:
            if match(goal_fact, "tightened", "*"):
                _, nut_name = get_parts(goal_fact)
                self.goal_nuts.add(nut_name)

        # Identify the man object (assuming only one)
        self.man_name = None
        for name, type_name in self.object_types.items():
             if type_name == 'man':
                 self.man_name = name
                 break


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_loc = None
        if self.man_name:
            for fact in state:
                if match(fact, "at", self.man_name, "*"):
                    man_loc = get_parts(fact)[2]
                    break

        # If man's location is not found or is not a known location, something is wrong
        if man_loc is None or man_loc not in self.locations:
             return float('inf')


        # 2. Identify untightened goal nuts and their locations
        untightened_goal_nuts = set()
        untightened_goal_nut_locs = set() # Locations of untightened goal nuts

        # First, find locations of all nuts in the state
        nut_locations = {} # Map nut name to its location
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name, loc_name = get_parts(fact)[1:3]
                # Check if obj_name is a nut
                if self.object_types.get(obj_name) == 'nut':
                     nut_locations[obj_name] = loc_name

        # Then, check which goal nuts are NOT tightened
        for nut_name in self.goal_nuts:
            if f"(tightened {nut_name})" not in state:
                 untightened_goal_nuts.add(nut_name)
                 # Get its location.
                 if nut_name in nut_locations:
                     untightened_goal_nut_locs.add(nut_locations[nut_name])
                 else:
                     # Location of untightened goal nut unknown, problem likely unsolvable
                     return float('inf')


        K = len(untightened_goal_nuts)

        # 3. If K is 0, goal is reached
        if K == 0:
            return 0

        # 4. Identify usable spanners
        usable_spanners_carried = set()
        usable_spanners_on_ground = set()
        spanner_locations_on_ground = {} # Map spanner name to location

        # Find carried usable spanners
        if self.man_name:
            for fact in state:
                if match(fact, "carrying", self.man_name, "*"):
                    spanner_name = get_parts(fact)[2]
                    # Check if the carried spanner is usable
                    if f"(usable {spanner_name})" in state:
                        usable_spanners_carried.add(spanner_name)

        # Find usable spanners on the ground and their locations
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj_name, loc_name = get_parts(fact)[1:3]
                 # Check if obj_name is a spanner and is usable
                 if self.object_types.get(obj_name) == 'spanner' and f"(usable {obj_name})" in state:
                     # Ensure it's not a spanner the man is carrying (shouldn't be at location if carried)
                     if obj_name not in usable_spanners_carried:
                         usable_spanners_on_ground.add(obj_name)
                         spanner_locations_on_ground[obj_name] = loc_name


        N_carried_usable = len(usable_spanners_carried)
        N_usable_on_ground = len(usable_spanners_on_ground)
        N_usable_total = N_carried_usable + N_usable_on_ground

        # 5. Check solvability based on available spanners
        if K > N_usable_total:
            return float('inf')

        # 6. Initialize heuristic
        h = 0

        # 7. Cost for tightening nuts (1 action per nut)
        h += K

        # 8. Cost for picking up spanners (1 action per spanner needed beyond carried ones)
        # We need K usable spanners in total. We have N_carried_usable.
        # We need to pick up max(0, K - N_carried_usable) more.
        num_pickups_needed = max(0, K - N_carried_usable)
        h += num_pickups_needed

        # 9. Calculate minimum distances
        min_dist_to_nut = float('inf')
        if untightened_goal_nut_locs:
            min_dist_to_nut = min(self.dist[man_loc].get(L_n, float('inf')) for L_n in untightened_goal_nut_locs)

        min_dist_to_spanner = float('inf')
        if num_pickups_needed > 0 and spanner_locations_on_ground:
             min_dist_to_spanner = min(self.dist[man_loc].get(L_s, float('inf')) for L_s in spanner_locations_on_ground.values())


        # 10. Check reachability for required locations
        if K > 0 and min_dist_to_nut == float('inf'):
             # Cannot reach any untightened goal nut
             return float('inf')
        if num_pickups_needed > 0 and min_dist_to_spanner == float('inf'):
             # Need pickups but cannot reach any usable spanner on the ground
             return float('inf')

        # 11. Add the calculated minimum distances (if reachable) to h
        # If we reached here, required locations are reachable.
        if K > 0:
             h += min_dist_to_nut
        if num_pickups_needed > 0:
             h += min_dist_to_spanner


        # 12. Return h
        return h
