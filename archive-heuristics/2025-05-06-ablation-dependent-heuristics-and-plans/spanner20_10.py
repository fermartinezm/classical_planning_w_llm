# Need to import Heuristic base class and fnmatch
from fnmatch import fnmatch
from collections import deque # For BFS
import math # For infinity

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a fact matches a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path in the location graph
def bfs(start_node, graph):
    """
    Performs BFS to find shortest distances from start_node to all other nodes.
    Returns a dictionary {location: distance}.
    """
    distances = {node: math.inf for node in graph}
    if start_node not in graph:
        # Start node is not in the graph of locations (e.g., it's an object name)
        # or it's an isolated location not linked to anything.
        # If it's in the graph keys, BFS will run. If not, distances remain inf.
        # If it's a valid location but isolated, its distance to itself is 0.
        if start_node in distances:
             distances[start_node] = 0
        return distances # Cannot start BFS if start_node is not a graph node

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        u = queue.popleft()
        if u in graph: # Ensure the node exists in the graph keys
            for v in graph[u]:
                if distances[v] == math.inf:
                    distances[v] = distances[u] + 1
                    queue.append(v)
    return distances

# class spannerHeuristic(Heuristic): # Assuming Heuristic base class exists
class spannerHeuristic: # Define as standalone class if base is not provided
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It sums the costs for tightening actions, picking up needed spanners, and the
    estimated travel costs for the man and the spanners.

    # Assumptions
    - All goal nuts are located at the same single location.
    - The man can carry multiple spanners.
    - Tightening a nut consumes one usable spanner, making it unusable, but the man continues carrying it.
    - The graph of locations connected by 'link' predicates is undirected (BFS treats it as such).
    - The man object is named 'bob' (or contains 'bob' case-insensitive).

    # Heuristic Initialization
    - Builds a graph of locations based on 'link' predicates, including all locations mentioned in the initial state and goals.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the location of the goal nuts (assuming they are all at the same location as specified in the initial state for goal nuts).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of loose nuts in the current state (`num_loose`). If none are loose, the heuristic is 0.
    2. Count the number of usable spanners the man is currently carrying (`num_carried_usable`).
    3. Count the total number of usable spanners available (carried or on the ground) (`total_usable`).
    4. If the number of loose nuts exceeds the total number of usable spanners, the problem is unsolvable from this state, return infinity.
    5. The base heuristic cost is the number of loose nuts (for the 'tighten_nut' actions).
    6. Add the cost for picking up spanners from the ground: This is the number of loose nuts minus the number of usable spanners the man is already carrying (if positive) (`needed_from_ground`). Each pickup costs 1.
    7. Calculate the walk cost:
       - Find the man's current location (`l_m`) and the location of the loose nuts (`l_nut`, precomputed).
       - Find the locations of all usable spanners on the ground (`usable_on_ground_map`).
       - If the man is carrying enough usable spanners for all loose nuts (`num_carried_usable >= num_loose`), the walk cost is simply the distance from the man's current location to the nut location: `dist(l_m, l_nut)`.
       - If the man needs to pick up spanners from the ground (`num_carried_usable < num_loose`):
         - Find the `needed_from_ground` usable spanners on ground whose locations are closest to the nut location (`l_nut`). Get their distances from `l_nut`. Let these sorted distances be `d_1 <= d_2 <= ... <= d_{needed_from_ground}`, corresponding to locations `l_s_1, l_s_2, ..., l_s_{needed_from_ground}`.
         - If the man starts carrying *some* usable spanners (`num_carried_usable > 0`): The walk cost is the distance from his current location to the nut location (`dist(l_m, l_nut)`), plus the sum of round-trip distances (nut location <-> spanner location) for fetching the `needed_from_ground` spanners. Cost: `dist(l_m, l_nut) + sum_{i=1}^{needed_from_ground} (dist(l_nut, l_s_i) + dist(l_s_i, l_nut))`.
         - If the man starts carrying *no* usable spanners (`num_carried_usable == 0`): The walk cost is the distance from his current location to the nut location (`dist(l_m, l_nut)`), plus the sum of round-trip distances (nut location <-> spanner location) for fetching *all* `num_loose` spanners needed from the ground. Cost: `dist(l_m, l_nut) + sum_{i=1}^{num_loose} (dist(l_nut, l_s_i) + dist(l_s_i, l_nut))`, where `l_s_i` are the locations of the `num_loose` closest usable spanners from `l_nut`.
    8. The total heuristic is the sum of the base cost (tighten + pickup) and the calculated walk cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and computing distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Build the location graph from 'link' facts and all mentioned locations
        self.locations = set()

        # Add all locations mentioned in initial state and goals to the set of locations
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                  # Assuming the second argument of 'at' is always a location
                  self.locations.add(parts[2])
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "at":
                  # Assuming the second argument of 'at' is always a location
                  self.locations.add(parts[2])
             # Goal might be just (tightened nut1), need to find nut location from initial state

        # Add locations from link facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Initialize graph with all identified locations
        self.graph = {loc: [] for loc in self.locations}

        # Add links to the graph
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                loc1, loc2 = parts[1], parts[2]
                # Ensure locations exist in graph keys (they should if added from self.locations)
                if loc1 in self.graph and loc2 in self.graph:
                    self.graph[loc1].append(loc2)
                    self.graph[loc2].append(loc1) # Assuming links are bidirectional

        # 2. Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.graph:
            self.distances[start_loc] = bfs(start_loc, self.graph)

        # 3. Identify the location of goal nuts
        # Assuming all goal nuts are at the same location in the initial state
        self.nut_location = None
        goal_nuts = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == "tightened"}

        if goal_nuts:
            # Find location of the first goal nut in the initial state
            for nut in goal_nuts:
                for fact in initial_state:
                    parts = get_parts(fact)
                    if parts[0] == "at" and parts[1] == nut:
                        self.nut_location = parts[2]
                        break # Found location for one goal nut
                if self.nut_location:
                     break # Found location for at least one goal nut

        # If nut_location is still None, it means goal nuts weren't found in initial 'at' facts
        # This might happen if the goal is already met in the initial state, or problem definition is unusual.
        # The heuristic will return 0 if goals are met. If not met and nut_location is None,
        # the heuristic might return a large value or inf depending on subsequent checks.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find man's current location
        man_location = None
        carried_spanners = set()
        usable_spanners = set()
        loose_nuts = set()
        # Map location to list of usable spanners on the ground at that location
        usable_on_ground_map = {loc: [] for loc in self.locations}

        # First pass to identify usable spanners and loose nuts
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "usable":
                  usable_spanners.add(parts[1])
             elif parts[0] == "loose":
                  loose_nuts.add(parts[1])

        # Second pass to find locations and carried items
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                # Assuming there's only one man and his name contains 'bob'
                if 'bob' in obj.lower():
                     man_location = loc
                # Track usable spanners on the ground
                if obj in usable_spanners:
                     if loc in self.locations: # Ensure location is valid
                         usable_on_ground_map[loc].append(obj)
            elif parts[0] == "carrying":
                carrier, spanner = parts[1], parts[2]
                # Assuming 'bob' is the man
                if 'bob' in carrier.lower():
                    carried_spanners.add(spanner)

        # Filter usable_on_ground_map to exclude carried spanners
        # Rebuild usable_on_ground_map considering carried spanners
        usable_on_ground_map = {loc: [] for loc in self.locations}
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "at":
                 obj, loc = parts[1], parts[2]
                 if obj in usable_spanners and obj not in carried_spanners:
                      if loc in self.locations:
                           usable_on_ground_map[loc].append(obj)


        carried_usable_spanners = {s for s in carried_spanners if s in usable_spanners}
        num_carried_usable = len(carried_usable_spanners)

        num_loose = len(loose_nuts)
        num_usable_on_ground = sum(len(spanners) for spanners in usable_on_ground_map.values())
        total_usable = num_carried_usable + num_usable_on_ground

        # If there are loose nuts but nut location wasn't found in init (problematic)
        # or if man_location is not found (problematic state)
        if num_loose > 0 and (self.nut_location is None or man_location is None):
             # This indicates a problem with parsing or state representation not matching assumptions
             # Return a high value indicating difficulty/impossibility
             return math.inf # Cannot proceed without man or nut location

        # Check if solvable (enough spanners exist)
        if num_loose > total_usable:
            return math.inf # Unsolvable state

        # Base cost: tighten actions + pickup actions
        h = num_loose # Cost for tighten_nut actions
        needed_from_ground = max(0, num_loose - num_carried_usable)
        h += needed_from_ground # Cost for pickup_spanner actions

        # Walk cost
        walk_cost = 0
        l_m = man_location
        l_nut = self.nut_location

        if num_loose > 0: # Only incur walk cost if there are nuts to tighten
            if num_carried_usable >= num_loose:
                # Man has enough spanners, just needs to walk to the nut location
                if l_m in self.distances and l_nut in self.distances[l_m]:
                     walk_cost = self.distances[l_m][l_nut]
                else:
                     return math.inf # Cannot reach nut location

            else: # num_carried_usable < num_loose, need to fetch spanners
                needed = num_loose - num_carried_usable

                # Find the usable spanners on ground whose locations are closest to l_nut.
                # We need distances from l_nut to these locations.
                dist_nut_to_ugl_list = []
                for loc, spanners in usable_on_ground_map.items():
                     if spanners:
                          if l_nut in self.distances and loc in self.distances[l_nut]:
                               # Add distance for each spanner at this location
                               for _ in spanners:
                                    dist_nut_to_ugl_list.append((self.distances[l_nut][loc], loc))
                          else:
                               return math.inf # Cannot reach spanner location

                # Sort by distance from l_nut
                sorted_dist_nut_to_ugl = sorted(dist_nut_to_ugl_list)

                # Ensure we have enough locations/spanners for num_loose (or needed)
                # This is guaranteed by total_usable check, but defensive check
                required_spanners_for_walk_cost = num_loose if num_carried_usable == 0 else needed_from_ground
                if len(sorted_dist_nut_to_ugl) < required_spanners_for_walk_cost:
                     return math.inf

                # Cost to get man to nut location
                if l_m in self.distances and l_nut in self.distances[l_m]:
                     walk_cost += self.distances[l_m][l_nut]
                else:
                     return math.inf # Cannot reach nut location

                # Cost to fetch needed spanners from l_nut (round trips)
                # If num_carried_usable > 0, fetch 'needed_from_ground' spanners.
                # If num_carried_usable == 0, fetch 'num_loose' spanners.
                num_fetches_from_nut = needed_from_ground if num_carried_usable > 0 else num_loose

                for i in range(num_fetches_from_nut):
                     fetch_loc = sorted_dist_nut_to_ugl[i][1] # Location of the i-th closest spanner from l_nut
                     if l_nut in self.distances and fetch_loc in self.distances[l_nut]:
                          dist_to_loc = self.distances[l_nut][fetch_loc]
                          walk_cost += dist_to_loc + dist_to_loc # Walk there and back
                     else:
                          return math.inf # Cannot reach spanner location from nut location


        # Total heuristic = base cost (tighten + pickup) + walk cost
        h += walk_cost

        return h
