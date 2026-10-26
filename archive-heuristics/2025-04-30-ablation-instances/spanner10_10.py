from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check is to just zip and check all elements match.
    # This assumes the pattern length is <= fact parts length.
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the number of nuts remaining, the movement cost for the man to reach
    the nut locations, and the cost to acquire usable spanners.

    # Assumptions
    - The goal is to tighten all nuts that are initially loose.
    - Each tighten action consumes one usable spanner.
    - Spanners are single-use.
    - The man is the only agent who can move and perform actions.
    - Nuts stay in their initial locations.
    - There are enough usable spanners available (either carried or on the ground)
      in the initial state to tighten all loose nuts in solvable problems.

    # Heuristic Initialization
    - Build a graph of locations based on 'link' facts from static information.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify all nut objects from the goal conditions.
    - Identify the man object and all spanner objects by inspecting predicates
      in the initial state (e.g., 'carrying', 'usable', 'at').
    - Count the total number of usable spanners available in the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the man's current location (`man_loc`).
    2. Identify all nuts that are currently loose and their locations (`loose_nuts_info`).
    3. Count the number of loose nuts (`num_loose_nuts`). If 0, the heuristic is 0 (goal state).
    4. Check if the problem is solvable from the initial state based on the total number
       of usable spanners available initially. If `num_loose_nuts` exceeds the total
       initial usable spanner count, return infinity.
    5. Initialize the heuristic value `h` with `num_loose_nuts`. This accounts for the
       minimum number of `tighten_nut` actions required.
    6. Estimate the movement cost for the man to reach the locations of the loose nuts.
       Identify the set of distinct locations where loose nuts are present.
       Add the sum of shortest path distances from the man's current location (`man_loc`)
       to each of these distinct nut locations to `h`. If any required nut location
       is unreachable from `man_loc`, return infinity.
    7. Estimate the cost for acquiring usable spanners. The man needs `num_loose_nuts`
       usable spanners in total throughout the plan. Count the number of usable spanners
       the man is currently carrying (`num_carrying_usable`). The man needs to pick up
       `max(0, num_loose_nuts - num_carrying_usable)` additional spanners from the ground.
       Estimate the cost for each needed pickup as 2 actions (1 move towards a spanner
       location + 1 `pickup_spanner` action). Add `max(0, num_loose_nuts - num_carrying_usable) * 2`
       to `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        static_facts = task.static
        self.initial_state = task.initial_state

        # Collect all potential locations from static links, initial state, and goals
        all_potential_locations = set()
        for fact in static_facts:
             if match(fact, "link", "*", "*"):
                  all_potential_locations.add(get_parts(fact)[1])
                  all_potential_locations.add(get_parts(fact)[2])
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                  all_potential_locations.add(get_parts(fact)[2])
        for goal in self.goals:
             if match(goal, "at", "*", "*"): # Goals might specify object locations
                  all_potential_locations.add(get_parts(goal)[2])

        self.locations = all_potential_locations

        # Build location graph (adjacency list) from static 'link' facts
        adj = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations from links are in our set, though they should be
                if loc1 in self.locations and loc2 in self.locations:
                    adj.setdefault(loc1, []).append(loc2)
                    adj.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distance = {}
        for start_loc in list(self.locations): # Iterate over a list copy
             self.distance[start_loc] = self._bfs_distances(start_loc, adj)

        # Identify all nut objects from goal conditions
        self.all_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut_obj = get_parts(goal)
                self.all_nuts.add(nut_obj)

        # Identify man and spanner objects from initial state predicates
        self.man_obj = None
        self.all_spanners = set()
        initial_locatables = set() # Objects found at locations initially

        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                initial_locatables.add(parts[1])
            elif parts[0] == 'carrying':
                # The carrier is the man, the carried is a spanner
                self.man_obj = parts[1]
                self.all_spanners.add(parts[2])
            elif parts[0] == 'usable':
                 # Usable object is a spanner
                 self.all_spanners.add(parts[1])
            # Nuts are identified from goals

        # If man wasn't found carrying, assume it's the locatable that isn't a nut or spanner
        if self.man_obj is None:
             potential_men = initial_locatables - self.all_nuts - self.all_spanners
             if len(potential_men) == 1:
                  self.man_obj = list(potential_men)[0]
             # else: Could be multiple men, or man not involved in 'at' initially.
             # Heuristic might be less reliable. Assume single man found this way.

        # Count total usable spanners available in the initial state
        self.initial_usable_spanners_count = 0
        for spanner_obj in self.all_spanners:
             if f"(usable {spanner_obj})" in self.initial_state:
                  self.initial_usable_spanners_count += 1


    def _bfs_distances(self, start_loc, adj):
        """Compute shortest path distances from start_loc to all other locations."""
        distances = {loc: float('inf') for loc in self.locations}
        if start_loc not in self.locations:
             # Start location is not in our known graph locations
             return distances

        distances[start_loc] = 0
        queue = deque([start_loc])
        visited = {start_loc}

        while queue:
            curr = queue.popleft()
            # Handle locations that might not have any links (isolated)
            if curr not in adj:
                 continue

            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Find man's current location
        man_loc = None
        if self.man_obj:
            for fact in state:
                if match(fact, "at", self.man_obj, "*"):
                    man_loc = get_parts(fact)[2]
                    break

        # If man's location is not found, the state is likely invalid or unreachable
        if man_loc is None or man_loc not in self.locations:
             return float('inf')


        # 2. Find all loose nuts and their locations
        loose_nuts_info = [] # List of (nut_obj, nut_loc)
        nut_locations = {} # nut_obj -> nut_loc (assuming nuts don't move)
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.all_nuts: # Check if the object is one of the known nuts
                     nut_locations[obj] = loc

        loose_nuts_list = [] # List of nut_obj
        for nut_obj in self.all_nuts:
             if f"(loose {nut_obj})" in state:
                  if nut_obj in nut_locations:
                       loose_nuts_info.append((nut_obj, nut_locations[nut_obj]))
                       loose_nuts_list.append(nut_obj)
                  else:
                       # Loose nut exists but its location is unknown in the state? Invalid state.
                       return float('inf')


        num_loose_nuts = len(loose_nuts_list)

        # If all nuts are tightened, goal reached, heuristic is 0.
        if num_loose_nuts == 0:
            return 0

        # Check for unsolvable state: Not enough usable spanners in total available initially
        if num_loose_nuts > self.initial_usable_spanners_count:
             return float('inf')


        # 3. Find usable spanners carried by the man
        num_carrying_usable = 0
        if self.man_obj:
             for fact in state:
                  if match(fact, "carrying", self.man_obj, "*"):
                       carried_spanner = get_parts(fact)[2]
                       # Check if the carried object is a known spanner and is usable
                       if carried_spanner in self.all_spanners and f"(usable {carried_spanner})" in state:
                            num_carrying_usable += 1


        # Calculate heuristic components

        # Component 1: Tighten actions
        h = num_loose_nuts

        # Component 2: Movement cost for the man to reach nuts
        distinct_nut_locations = {loc for _, loc in loose_nuts_info}
        movement_to_nuts_cost = 0

        # Ensure man_loc is a valid key in distance dict
        if man_loc not in self.distance:
             # Man is at a location not in the graph? Invalid state.
             return float('inf')

        for loc in distinct_nut_locations:
            # Check if the nut location is in our known locations and reachable from man_loc
            if loc not in self.distance[man_loc] or self.distance[man_loc][loc] == float('inf'):
                 # Man cannot reach a nut location
                 return float('inf')
            # Add cost only if man is not already at this distinct nut location
            if man_loc != loc:
                 movement_to_nuts_cost += self.distance[man_loc][loc]

        h += movement_to_nuts_cost

        # Component 3: Spanner acquisition cost
        # Man needs num_loose_nuts usable spanners in total. Has num_carrying_usable.
        num_pickups_needed = max(0, num_loose_nuts - num_carrying_usable)

        # Estimate cost for pickups: num_pickups_needed * (estimated_move_cost + pickup_action_cost)
        # Assume estimated_move_cost is 1 (a single step towards a spanner) and pickup_action_cost is 1.
        # This is a simplification; actual move cost depends on spanner locations.
        h += num_pickups_needed * 2

        return h
