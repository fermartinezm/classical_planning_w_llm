# Imports needed
from heuristics.heuristic_base import Heuristic
from collections import deque # For BFS
import math # For float('inf')

# Heuristic class definition
class spannerHeuristic(Heuristic):
    """
    spannerHeuristic estimates the cost to reach a goal state in the spanner domain.

    Summary:
    The heuristic estimates the total number of actions required, comprising
    tighten actions, spanner pickup actions, and travel actions. It sums
    the number of loose nuts (representing tighten actions), the number of
    additional spanners that need to be picked up (representing pickup actions),
    and an estimate of the travel cost. The travel cost is estimated as the
    sum of shortest path distances from the man's current location to each
    location that needs to be visited (all loose nut locations and the
    locations of the closest usable spanners that need to be picked up).

    Assumptions:
    - There is exactly one man object in the domain.
    - Nut locations are static (they do not change during planning).
    - The graph of locations connected by links is undirected and unweighted.
    - Object names can be reliably identified by type (e.g., 'bob' is a man,
      'spanner1' is a spanner, 'nut1' is a nut). This is handled by parsing
      the task.objects string to get object types.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes static information:
    - Parses the task's object definitions to map object names to types.
    - Parses static 'link' facts to build an adjacency list representation of the location graph.
    - Parses initial state 'at' facts for nuts to store their static locations.
    - Computes all-pairs shortest paths between all locations using BFS, storing distances in a matrix.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to identify:
        - The man's current location.
        - Which objects the man is carrying.
        - Which objects are at which locations.
        - Which nuts are loose.
        - Which spanners are usable.
    2.  Identify the set of usable spanners the man is currently carrying. Count them (`k_c`).
    3.  Identify the set of usable spanners located at various places (not carried). Store their locations. Count them (`k_a`).
    4.  Count the number of loose nuts (`k`).
    5.  If `k` is 0, the goal is reached, return 0.
    6.  Check if the total number of usable spanners (`k_c + k_a`) is less than the number of loose nuts (`k`). If so, the goal is unreachable, return infinity.
    7.  Calculate the number of additional spanners the man needs to pick up: `k_needed = max(0, k - k_c)`.
    8.  Identify the locations of all loose nuts (`loose_nut_locations`).
    9.  Identify the locations of all available usable spanners (not carried).
    10. Determine the set of required locations the man must visit:
        - This set includes all `loose_nut_locations`.
        - If `k_needed > 0`, it also includes the locations of the `k_needed` usable spanners (not carried) that are closest to the man's current location.
    11. Calculate the estimated travel cost as the sum of shortest path distances from the man's current location to each unique location in the set of required locations. If any required location is unreachable, return infinity.
    12. The heuristic value is the sum of:
        - The number of loose nuts (`k`, representing tighten actions).
        - The number of needed pickups (`k_needed`, representing pickup actions).
        - The estimated travel cost.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static
        self.task_objects_str = task.objects # Store the raw string
        self.initial_state = task.initial_state # Store initial state to get static nut locations

        # Data structures for static info
        self.locations = set()
        self.links = {} # adjacency list: loc -> set of connected locs
        self.nut_locations = {} # nut -> location (static)
        self.object_types = {} # obj_name -> type
        self.man_name = None # Store the name of the man

        # Parse objects and types from the task.objects string
        self._parse_task_objects(self.task_objects_str)

        # Find the man's name
        for obj, obj_type in self.object_types.items():
            if obj_type == 'man':
                self.man_name = obj
                break
        
        # If no man found, the domain might be malformed or different than expected
        # This heuristic assumes a man exists.

        # Parse static facts (links)
        for fact in task.static:
            parsed = self._parse_fact(fact)
            if parsed and parsed[0] == 'link':
                l1, l2 = parsed[1], parsed[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.links.setdefault(l1, set()).add(l2)
                self.links.setdefault(l2, set()).add(l1)

        # Parse initial nut locations from initial state (nuts don't move)
        for fact in self.initial_state:
            parsed = self._parse_fact(fact)
            if parsed and parsed[0] == 'at' and self.object_types.get(parsed[1]) == 'nut':
                 self.nut_locations[parsed[1]] = parsed[2]

        # Compute all-pairs shortest paths (or just from all locations)
        self.dist = self._compute_all_pairs_shortest_paths()

    def _parse_task_objects(self, objects_str):
         lines = objects_str.strip().split('\n')
         for line in lines:
             line = line.strip()
             if not line or line.startswith(';'):
                 continue
             parts = line.split()
             if not parts: continue # Skip empty lines after strip
             if parts[-1] == '-': # e.g., "shed location1 gate -"
                 if len(parts) < 3: continue # Malformed line
                 obj_type = parts[-2]
                 obj_names = parts[:-2]
             else: # e.g., "bob - man"
                 if len(parts) < 3 or parts[1] != '-': continue # Malformed line
                 obj_type = parts[-1]
                 obj_names = [parts[0]]
             for obj_name in obj_names:
                 self.object_types[obj_name] = obj_type

    def _parse_fact(self, fact_str):
        # Remove parentheses and split by space
        # Handle potential empty fact string or malformed fact
        if not fact_str or not fact_str.startswith('(') or not fact_str.endswith(')'):
            return None
        content = fact_str[1:-1].strip()
        if not content: return None
        parts = content.split()
        return tuple(parts)

    def _compute_all_pairs_shortest_paths(self):
        dist = {l: {l2: math.inf for l2 in self.locations} for l in self.locations}
        for l in self.locations:
            dist[l][l] = 0
            queue = deque([(l, 0)])
            visited = {l}
            while queue:
                curr_loc, curr_d = queue.popleft()
                if curr_loc in self.links:
                    for neighbor in self.links[curr_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            dist[l][neighbor] = curr_d + 1
                            queue.append((neighbor, curr_d + 1))
        return dist

    def __call__(self, node):
        state = node.state

        loc_m = None
        carried_objects = [] # Store all carried objects
        objects_at_locs = {} # loc -> list of objects
        loose_nuts = []
        usable_spanners_in_state = set() # usable spanner names

        for fact in state:
            parsed = self._parse_fact(fact)
            if parsed is None: continue # Skip malformed facts
            if parsed[0] == 'at':
                obj, loc = parsed[1], parsed[2]
                objects_at_locs.setdefault(loc, []).append(obj)
                if self.object_types.get(obj) == 'man' and obj == self.man_name:
                    loc_m = loc
            elif parsed[0] == 'carrying':
                man, obj = parsed[1], parsed[2]
                if self.object_types.get(man) == 'man' and man == self.man_name:
                    carried_objects.append(obj)
            elif parsed[0] == 'loose':
                nut = parsed[1]
                if self.object_types.get(nut) == 'nut':
                    loose_nuts.append(nut)
            elif parsed[0] == 'usable':
                spanner = parsed[1]
                if self.object_types.get(spanner) == 'spanner':
                    usable_spanners_in_state.add(spanner)

        # Ensure man's location is found
        if loc_m is None:
             # Man exists but is not at any location? Should be unreachable.
             # Or man_name wasn't found in __init__ (no man object defined)?
             return math.inf

        # Filter usable spanners from carried objects
        usable_carried = [s for s in carried_objects if self.object_types.get(s) == 'spanner' and s in usable_spanners_in_state]
        k_c = len(usable_carried)

        # Filter usable spanners at locations
        usable_at_locs = {} # loc -> list of usable spanners
        for loc, objs in objects_at_locs.items():
            usable_at_locs[loc] = [obj for obj in objs if self.object_types.get(obj) == 'spanner' and obj in usable_spanners_in_state]

        k_a = sum(len(spanners) for spanners in usable_at_locs.values())
        k = len(loose_nuts)

        # Check for goal state
        if k == 0:
            return 0

        # Check for dead end (not enough usable spanners)
        if k_c + k_a < k:
            return math.inf

        # Calculate k_needed pickups
        k_needed = max(0, k - k_c)

        # Get locations of loose nuts
        loose_nut_locations = [self.nut_locations[n] for n in loose_nuts]

        # Identify locations of available usable spanners (not carried)
        available_spanners_with_loc = [] # List of (distance_from_man, location, spanner)
        for loc, spanners in usable_at_locs.items():
            for spanner in spanners:
                if loc_m in self.dist and loc in self.dist[loc_m]:
                    available_spanners_with_loc.append((self.dist[loc_m][loc], loc, spanner))
                else:
                     # Unreachable spanner location - should indicate unreachable state
                     return math.inf

        # Sort usable spanners by distance of their location from man
        available_spanners_with_loc.sort()

        # Select the locations of the k_needed closest spanners
        spanners_to_pickup_locs = set()
        for i in range(min(k_needed, len(available_spanners_with_loc))):
            spanners_to_pickup_locs.add(available_spanners_with_loc[i][1]) # Add the location

        # Combine required locations: all nut locations + locations of needed pickups
        required_locs = set(loose_nut_locations) | spanners_to_pickup_locs

        # Calculate sum of distances from man's location to all required unique locations
        travel_cost = 0
        for l in required_locs:
            if loc_m in self.dist and l in self.dist[loc_m]:
                travel_cost += self.dist[loc_m][l]
            else:
                # A required location is unreachable - should indicate unreachable state
                return math.inf

        # Heuristic = k (tighten) + k_needed (pickup) + travel_cost (sum of dists)
        h_value = k + k_needed + travel_cost

        return h_value
