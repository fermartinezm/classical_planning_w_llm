from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

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
    # Ensure we have the same number of parts or the pattern uses wildcards appropriately
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It simulates a greedy strategy where
    the man sequentially addresses each loose goal nut. For each nut, if
    a usable spanner is not carried, the man travels to the nearest location
    with a usable spanner, picks it up, travels to the nut's location, and
    tightens the nut. If a usable spanner is carried, the man travels directly
    to the nut's location and tightens it. The spanner becomes unusable after
    one use. This simulation assumes the man can carry only one spanner at a time.

    # Assumptions:
    - There is only one man.
    - The man can carry at most one spanner at a time.
    - Spanners become unusable after tightening one nut.
    - All locations are connected, allowing travel between any two locations (possibly indirectly).
    - All nuts required by the goal start as 'loose'.
    - There are enough usable spanners available in the problem instance to tighten all goal nuts.

    # Heuristic Initialization
    - Identify the man object, all location objects, nut objects, and spanner objects from the task definition.
    - Build the graph of locations based on 'link' facts.
    - Compute all-pairs shortest paths between all locations using BFS.
    - Identify the set of nuts that need to be tightened (goal nuts).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic simulates a greedy sequence of actions to tighten all loose goal nuts:

    1.  Identify the set of nuts that are currently 'loose' and are required to be 'tightened' in the goal state (`loose_goal_nuts`). If this set is empty, the heuristic is 0.
    2.  Determine the man's current location (`man_loc`).
    3.  Determine which usable spanners are available at locations and if the man is carrying a usable one. Count usable spanners per location.
    4.  Initialize the heuristic cost `h = 0`. Set the man's `current_loc` to `man_loc`. Track if the man is `carrying_usable`.
    5.  While there are still `remaining_nuts` in `loose_goal_nuts`:
        a.  If the man is `carrying_usable`:
            i.  Find the nearest remaining nut (`next_nut`) from `current_loc`.
            ii. Add the travel cost (`dist(current_loc, NutLoc[next_nut])`) to `h`.
            iii. Update `current_loc` to `NutLoc[next_nut)`.
            iv. Remove `next_nut` from `remaining_nuts`.
            v.  Add 1 to `h` for the `tighten_nut` action.
            vi. Set `carrying_usable = False` (the spanner is used).
        b.  If the man is NOT `carrying_usable`:
            i.  Find the nearest location (`nearest_spanner_loc`) that has at least one usable spanner available from `current_loc`. If no usable spanners are available anywhere, return infinity (problem likely unsolvable).
            ii. Add the travel cost (`dist(current_loc, nearest_spanner_loc)`) to `h`.
            iii. Update `current_loc` to `nearest_spanner_loc`.
            iv. Add 1 to `h` for the `pickup_spanner` action.
            v.  Decrement the count of usable spanners at `nearest_spanner_loc`.
            vi. Set `carrying_usable = True`.
            vii. Now carrying a spanner, find the nearest remaining nut (`next_nut`) from `current_loc`.
            viii. Add the travel cost (`dist(current_loc, NutLoc[next_nut])`) to `h`.
            ix. Update `current_loc` to `NutLoc[next_nut)`.
            x.  Remove `next_nut` from `remaining_nuts`.
            xi. Add 1 to `h` for the `tighten_nut` action.
            xii. Set `carrying_usable = False` (the spanner is used).
    6.  Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information and computing distances."""
        self.goals = task.goals
        self.static_facts = task.static
        self.objects = task.objects # List of (name, type) tuples

        self.man_object = None
        self.locations = set()
        self.nuts = set()
        self.spanners = set()

        # Identify object types
        for obj_name, obj_type in self.objects:
            if obj_type == 'man':
                self.man_object = obj_name
            elif obj_type == 'location':
                self.locations.add(obj_name)
            elif obj_type == 'nut':
                self.nuts.add(obj_name)
            elif obj_type == 'spanner':
                self.spanners.add(obj_name)

        # Build location graph
        self.adj = defaultdict(list)
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.adj[loc1].append(loc2)
                self.adj[loc2].append(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = self._bfs(start_node)

        # Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

    def _bfs(self, start_node):
        """Perform BFS to find shortest paths from start_node to all other locations."""
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            curr = queue.popleft()
            # Ensure curr is a valid location before accessing self.adj
            if curr not in self.adj:
                 continue
            for neighbor in self.adj[curr]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Get the precomputed shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Check if locations exist and distance is computed
        if loc1 not in self.dist or loc2 not in self.dist.get(loc1, {}):
             # This implies an unreachable location, return infinity
             return float('inf')
        return self.dist[loc1][loc2]

    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # Parse state to get dynamic information
        man_loc = None
        loose_nuts = set()
        usable_spanners_in_state = set() # All usable spanners mentioned in state
        spanner_locations = {} # Map spanner to its location or man object if carried
        nut_locations = {}     # Map nut to its location
        carried_spanner = None # Assuming single spanner capacity

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj == self.man_object:
                    man_loc = loc
                elif obj in self.spanners:
                    spanner_locations[obj] = loc
                elif obj in self.nuts:
                    nut_locations[obj] = loc
            elif parts[0] == 'carrying':
                 m, s = parts[1], parts[2]
                 if m == self.man_object and s in self.spanners:
                     spanner_locations[s] = self.man_object # Mark as carried
                     carried_spanner = s # Store the carried spanner object
            elif parts[0] == 'usable':
                 s = parts[1]
                 if s in self.spanners:
                     usable_spanners_in_state.add(s)
            elif parts[0] == 'loose':
                 n = parts[1]
                 if n in self.nuts:
                     loose_nuts.add(n)
            # We don't need 'tightened' facts from state, we use goal_nuts and loose_nuts

        # Identify loose nuts that are goal nuts
        loose_goal_nuts = {n for n in loose_nuts if n in self.goal_nuts}

        # If all goal nuts are tightened (i.e., none are loose goal nuts), heuristic is 0
        if not loose_goal_nuts:
            return 0

        # Check if the carried spanner is usable
        carrying_usable = (carried_spanner is not None) and (carried_spanner in usable_spanners_in_state)

        # Count usable spanners available at locations
        available_spanner_locations = defaultdict(int)
        for s in usable_spanners_in_state:
            loc = spanner_locations.get(s)
            if loc and loc != self.man_object: # Spanner is usable and at a location (not carried)
                available_spanner_locations[loc] += 1

        h = 0
        current_loc = man_loc
        remaining_nuts = list(loose_goal_nuts) # Use a list to allow removal

        while remaining_nuts:
            if carrying_usable:
                # Go to nearest remaining nut
                nearest_nut = None
                min_dist_to_nut = float('inf')
                for nut in remaining_nuts:
                    nut_loc = nut_locations.get(nut)
                    if nut_loc: # Ensure nut location is known
                        dist = self.get_distance(current_loc, nut_loc)
                        if dist < min_dist_to_nut:
                            min_dist_to_nut = dist
                            nearest_nut = nut

                if nearest_nut is None or min_dist_to_nut == float('inf'):
                    # No remaining nuts or nearest nut is unreachable
                    return float('inf') # Problem likely unsolvable

                h += min_dist_to_nut # Travel to nut
                current_loc = nut_locations[nearest_nut]
                remaining_nuts.remove(nearest_nut)
                h += 1 # Tighten nut
                carrying_usable = False # Spanner used

            else: # Need to get a spanner
                # Find nearest location with usable spanners
                nearest_spanner_loc = None
                min_dist_to_spanner_loc = float('inf')
                for loc, count in available_spanner_locations.items():
                    if count > 0:
                        dist = self.get_distance(current_loc, loc)
                        if dist < min_dist_to_spanner_loc:
                            min_dist_to_spanner_loc = dist
                            nearest_spanner_loc = loc

                if nearest_spanner_loc is None:
                    # No usable spanners left anywhere, but nuts remain. Unsolvable.
                    return float('inf')

                h += min_dist_to_spanner_loc # Travel to spanner location
                h += 1 # Pickup spanner
                current_loc = nearest_spanner_loc
                available_spanner_locations[nearest_spanner_loc] -= 1
                carrying_usable = True # Now carrying a usable spanner

                # Now carrying a spanner, go to nearest remaining nut
                nearest_nut = None
                min_dist_to_nut = float('inf')
                for nut in remaining_nuts:
                    nut_loc = nut_locations.get(nut)
                    if nut_loc: # Ensure nut location is known
                        dist = self.get_distance(current_loc, nut_loc)
                        if dist < min_dist_to_nut:
                            min_dist_to_nut = dist
                            nearest_nut = nut

                if nearest_nut is None or min_dist_to_nut == float('inf'):
                     # No remaining nuts or nearest nut is unreachable
                     return float('inf') # Problem likely unsolvable

                h += min_dist_to_nut # Travel to nut
                current_loc = nut_locations[nearest_nut]
                remaining_nuts.remove(nearest_nut)
                h += 1 # Tighten nut
                carrying_usable = False # Spanner used

        return h
