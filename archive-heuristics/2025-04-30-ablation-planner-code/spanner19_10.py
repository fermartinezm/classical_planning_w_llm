# Helper functions from examples
from fnmatch import fnmatch
from collections import defaultdict, deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class exists as in examples
from heuristics.heuristic_base import Heuristic


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all required nuts by summing:
    1. The number of loose nuts that are goal conditions (base action cost for tighten).
    2. The number of usable spanners the man needs to pick up (base action cost for pickup).
    3. The sum of shortest path distances from the man's current location to each loose goal nut's location.
    4. The sum of shortest path distances from the man's current location to the locations of the usable spanners he needs to pick up.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Each tightening action requires one usable spanner.
    - Spanners are consumed (become unusable) after one use.
    - The man can carry multiple spanners (though the heuristic simplifies pickup logic by just counting needed pickups).
    - The shortest path distance between locations is a reasonable estimate for walking cost.
    - The number of spanners needed is the number of loose goal nuts minus the number of usable spanners the man is currently carrying.
    - All locations mentioned in the problem (objects, links, initial state, goals) are part of the graph.
    - The man object is named 'bob'.

    # Heuristic Initialization
    - Parse static facts to build the location graph based on `link` predicates.
    - Identify all locations mentioned in objects and static facts.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify the set of nuts that must be tightened according to the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all nuts that are currently loose AND are part of the goal conditions. Let this set be `LooseGoalNuts`.
    3. If `LooseGoalNuts` is empty, the heuristic is 0 (goal state).
    4. Count the number of loose goal nuts (`N_loose_goal_nuts`).
    5. Identify usable spanners the man is currently carrying (`UsableSpannersCarried`).
    6. Identify usable spanners on the ground and their locations (`UsableSpannersGround`).
    7. Calculate the number of additional usable spanners the man needs to acquire from the ground: `N_spanners_to_acquire = max(0, N_loose_goal_nuts - len(UsableSpannersCarried))`.
    8. Initialize heuristic value `h = 0`.
    9. Add base action costs: `h += N_loose_goal_nuts` (for tighten actions) + `N_spanners_to_acquire` (for pickup actions).
    10. Add walking cost towards nuts: For each nut `n` in `LooseGoalNuts`, find its current location `loc_n`. Add the shortest distance from the man's location to `loc_n` (`dist(man_loc, loc_n)`) to `h`.
    11. Add walking cost towards spanners (if needed): If `N_spanners_to_acquire > 0`, find the `N_spanners_to_acquire` usable spanners on the ground that are closest to the man's current location. Add the sum of the shortest distances from the man's location to these spanner locations (`sum(dist(man_loc, loc_s))`) to `h`. If there are fewer than `N_spanners_to_acquire` usable spanners on the ground, use all available ones for the distance calculation.
    12. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, computing distances,
        and identifying goal nuts.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Identify all locations
        self.locations = set()
        # Locations can appear in objects or static facts (like links)
        # Parse objects first
        if hasattr(task, 'objects') and task.objects:
             # Assuming task.objects is a list of strings like "obj_name - type"
             for obj_str in task.objects:
                 parts = obj_str.split()
                 if len(parts) >= 3 and parts[1] == '-':
                     obj_name = parts[0]
                     obj_type = parts[2]
                     if obj_type == 'location':
                         self.locations.add(obj_name)

        # Parse links to find any locations not listed in objects (less common but safe)
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)

        # 2. Build location graph (adjacency list)
        self.adj = defaultdict(list)
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Links are bidirectional
                self.adj[loc1].append(loc2)
                self.adj[loc2].append(loc1)

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        # Ensure all locations found are in the adj list keys, even if they have no links
        # Iterate over a copy as _bfs might add locations found in state later
        all_current_locations = list(self.locations)
        for loc in all_current_locations:
             if loc not in self.adj:
                  self.adj[loc] = []
             self.distances[loc] = self._bfs(loc)


        # 4. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            # Goal is typically (and (tightened nut1) (tightened nut2) ...)
            # Or just (tightened nut1)
            # Handle (and ...) structure
            if match(goal, "and", "*"):
                 # Simple split might fail on complex PDDL, but for standard goals it's ok
                 # A more robust parser would be needed for general PDDL
                 # Assuming facts inside (and ...) are space-separated and simple
                 content = goal[4:-1].strip()
                 # Find facts by looking for opening parentheses
                 i = 0
                 while i < len(content):
                     if content[i] == '(':
                         j = i
                         paren_count = 0
                         while j < len(content):
                             if content[j] == '(':
                                 paren_count += 1
                             elif content[j] == ')':
                                 paren_count -= 1
                             if paren_count == 0:
                                 fact_str = content[i:j+1]
                                 parts = get_parts(fact_str)
                                 if parts and parts[0] == "tightened" and len(parts) > 1:
                                     self.goal_nuts.add(parts[1])
                                 i = j + 1 # Move past this fact
                                 break
                             j += 1
                     else:
                         i += 1 # Skip spaces or other characters
            elif match(goal, "tightened", "*"):
                 parts = get_parts(goal)
                 if len(parts) > 1:
                    self.goal_nuts.add(parts[1])


    def _bfs(self, start_loc):
        """Computes shortest path distances from start_loc to all other locations."""
        distances = {loc: float('inf') for loc in self.locations}
        # Add start_loc to locations if it wasn't found during parsing (e.g., initial state location)
        if start_loc not in self.locations:
             self.locations.add(start_loc)
             self.adj[start_loc] = [] # Assume no links unless specified
             distances[start_loc] = float('inf') # Will be updated below if reachable

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            curr = queue.popleft()
            # Ensure curr is in adj (might not be if it was just added)
            if curr in self.adj:
                for neighbor in self.adj[curr]:
                    # Add neighbor to locations and distances if not seen before
                    if neighbor not in self.locations:
                         self.locations.add(neighbor)
                         self.adj[neighbor] = [] # Assume no links unless specified
                         distances[neighbor] = float('inf') # Initialize distance

                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[curr] + 1
                        queue.append(neighbor)

        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).

        # 1. Identify the man's current location, nut locations, and spanner locations/status
        man_loc = None
        nut_locations = {} # nut_name -> location
        usable_spanners_carried = set() # set of spanner names
        usable_spanners_ground = defaultdict(list) # location -> list of spanner names
        all_spanners_usable_status = {} # spanner_name -> bool (is usable?)

        # First pass to get usable status
        for fact in state:
             if match(fact, "usable", "*"):
                 s = get_parts(fact)[1]
                 all_spanners_usable_status[s] = True

        # Second pass to get locations and carrying status
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                # Assume 'bob' is the man, nuts start with 'nut', spanners with 'spanner'
                if obj == 'bob': # Assuming 'bob' is the man
                    man_loc = loc
                elif obj.startswith('nut'):
                    nut_locations[obj] = loc
                elif obj.startswith('spanner'):
                    if all_spanners_usable_status.get(obj, False): # Check if usable
                         usable_spanners_ground[loc].append(obj)
            elif parts[0] == 'carrying':
                 m, s = parts[1], parts[2]
                 if m == 'bob' and s.startswith('spanner'): # Assuming 'bob' is the man
                     if all_spanners_usable_status.get(s, False): # Check if usable
                         usable_spanners_carried.add(s)


        # 2. Identify loose nuts that are goals
        loose_goal_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                if nut in self.goal_nuts:
                    loose_goal_nuts.add(nut)

        # 3. If LooseGoalNuts is empty, the goal is reached.
        if not loose_goal_nuts:
            return 0

        # Ensure man_loc was found and is in our distance map
        if man_loc is None or man_loc not in self.distances:
             # This should not happen in valid states or indicates a graph issue
             # print(f"Error: Man's location {man_loc} not found or not in distance map.")
             return float('inf') # Should not be reachable in solvable problems


        # 4. Calculate N_loose_goal_nuts
        n_loose_goal_nuts = len(loose_goal_nuts)

        # 5. Calculate N_usable_spanners_carried
        n_usable_spanners_carried = len(usable_spanners_carried)

        # 6. Calculate N_spanners_to_acquire
        n_spanners_to_acquire = max(0, n_loose_goal_nuts - n_usable_spanners_carried)

        # 7. Initialize heuristic value
        h = 0

        # 8. Add base action costs
        # Each loose goal nut needs a tighten action (cost 1)
        # Each spanner that needs to be acquired needs a pickup action (cost 1)
        h += n_loose_goal_nuts + n_spanners_to_acquire

        # 9. Add walking cost towards nuts
        # Sum of distances from man's location to each loose goal nut's location
        for nut in loose_goal_nuts:
            loc_n = nut_locations.get(nut)
            # Check if location is known and reachable from man_loc
            if loc_n is None or loc_n not in self.distances[man_loc] or self.distances[man_loc][loc_n] == float('inf'):
                 # This nut is loose and a goal, but its location is unknown or unreachable
                 # print(f"Error: Location of loose goal nut {nut} ({loc_n}) not found or not reachable from man location {man_loc}.")
                 return float('inf') # Should not be reachable

            h += self.distances[man_loc][loc_n]

        # 10. Add walking cost towards spanners (if needed)
        if n_spanners_to_acquire > 0:
            usable_ground_list = []
            for loc_s, spanners in usable_spanners_ground.items():
                 # Check if spanner location is known and reachable from man_loc
                 if loc_s not in self.distances[man_loc] or self.distances[man_loc][loc_s] == float('inf'):
                      # print(f"Error: Spanner location {loc_s} not reachable from man location {man_loc}.")
                      # If a needed spanner is unreachable, the problem is unsolvable
                      return float('inf') # Should not be reachable

                 for s in spanners:
                    usable_ground_list.append((s, loc_s))

            # If there aren't enough usable spanners on the ground, we can only acquire the available ones
            num_to_take = min(n_spanners_to_acquire, len(usable_ground_list))

            if num_to_take > 0:
                # Sort usable spanners on ground by distance from man_loc
                usable_ground_list.sort(key=lambda item: self.distances[man_loc][item[1]])

                # Take the 'num_to_take' closest ones
                closest_spanners = usable_ground_list[:num_to_take]

                # Add the sum of distances to these spanner locations
                h += sum(self.distances[man_loc][loc_s] for s, loc_s in closest_spanners)

        return h
