from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class is available

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or invalid format defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It calculates the cost by simulating a greedy sequence: if a spanner is needed,
    go to the nearest usable spanner and pick it up; then, go to the nearest loose
    goal nut and tighten it. This process repeats until all goal nuts are tightened.

    # Assumptions:
    - There is only one man, and his name is 'bob'.
    - Spanners become permanently unusable after one use.
    - The man can carry only one spanner at a time.
    - All locations involved in links or object placements are part of a single connected graph, or required locations are reachable.
    - All goal nuts and usable spanners have defined locations in the state.

    # Heuristic Initialization
    - Extracts the set of nuts that are goals.
    - Builds a graph of locations based on `link` predicates and locations mentioned in initial state 'at' facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify which nuts are loose and are also goal conditions (`loose_goal_nuts`). If none, the heuristic is 0.
    3. Identify usable spanners currently on the ground.
    4. Check if the man is currently carrying a usable spanner.
    5. Initialize the heuristic cost `h` to 0.
    6. Initialize the man's current location for the simulation.
    7. Initialize the list of nuts still needing tightening (`nuts_to_tighten`).
    8. Initialize the list of usable spanners available on the ground (`usable_spanners_available`).
    9. Initialize a flag indicating if the man currently possesses a usable spanner (`man_has_usable_spanner_now`). This flag starts based on the actual state.
    10. While there are still nuts to tighten:
        a. If the man does *not* have a usable spanner:
           i. Find the usable spanner on the ground that is nearest to the man's current location.
           ii. If no reachable usable spanners are available on the ground, the task is impossible from this state; return infinity.
           iii. Add the distance to this spanner location plus 1 (for the pickup action) to `h`.
           iv. Update the man's current location to the spanner's location.
           v. Mark that the man now has a usable spanner.
           vi. Remove the picked-up spanner from the list of available spanners on the ground.
        b. Find the loose goal nut that is nearest to the man's current location.
        c. If no reachable loose goal nuts remain (should not happen if `nuts_to_tighten` is not empty and locations are known/reachable), return infinity.
        d. Add the distance to this nut location plus 1 (for the tighten action) to `h`.
        e. Update the man's current location to the nut's location.
        f. Remove the tightened nut from the list of nuts needing tightening.
        g. Mark that the man no longer has a usable spanner (it was used).
    11. Return the total calculated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the location graph."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # 1. Extract goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'tightened' and len(parts) > 1:
                self.goal_nuts.add(parts[1])

        # 2. Build location graph and find all locations
        all_mentioned_locations = set()
        adj = {}

        # Collect locations from link facts
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'link' and len(parts) == 3:
                 l1, l2 = parts[1], parts[2]
                 all_mentioned_locations.add(l1)
                 all_mentioned_locations.add(l2)
                 adj.setdefault(l1, []).append(l2)
                 adj.setdefault(l2, []).append(l1) # Links are bidirectional

        # Collect locations from initial state 'at' facts
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 # The second argument of 'at' is a location
                 all_mentioned_locations.add(parts[2])

        # Use all mentioned locations as nodes for distance calculation
        self.locations = list(all_mentioned_locations)

        # Ensure all collected locations are keys in the adjacency list, even if isolated
        self.location_graph = {loc: adj.get(loc, []) for loc in self.locations}


        # 3. Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Find the man's name (assuming 'bob' based on examples)
        # A more robust way would be needed in a general planner
        self.man_name = 'bob'


    def _bfs(self, start_node):
        """Perform BFS from start_node to find distances to all reachable locations."""
        distances = {loc: float('inf') for loc in self.locations}

        # Check if the start node is one of the locations we know about
        if start_node not in self.locations:
             # Cannot compute distances from an unknown location
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Check if node has neighbors defined in the graph
            if u in self.location_graph:
                for v in self.location_graph[u]:
                    # Ensure neighbor is also a known location
                    if v in self.locations and distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Parse state to find relevant facts
        obj_locations = {}
        man_location = None
        man_carrying_spanner = None # Store the spanner object name
        usable_spanners_in_state = set() # Usable spanners anywhere
        loose_nuts_in_state = set() # Loose nuts anywhere

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]

            if predicate == 'at':
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    obj_locations[obj] = loc
                    if obj == self.man_name:
                        man_location = loc
            elif predicate == 'carrying':
                if len(parts) == 3:
                    m, s = parts[1], parts[2]
                    if m == self.man_name:
                        man_carrying_spanner = s
            elif predicate == 'usable':
                if len(parts) == 2:
                    s = parts[1]
                    usable_spanners_in_state.add(s)
            elif predicate == 'loose':
                if len(parts) == 2:
                    n = parts[1]
                    loose_nuts_in_state.add(n)

        # Check if man_location was found. If not, something is wrong or state is malformed.
        if man_location is None:
             # Man's location is unknown, cannot proceed.
             return float('inf')

        # 2. Identify loose nuts that are goals
        loose_goal_nuts = self.goal_nuts.intersection(loose_nuts_in_state)

        # If goal is already reached, h=0
        if not loose_goal_nuts:
            return 0

        # 3. Identify usable spanners on the ground
        usable_spanners_on_ground = {s for s in usable_spanners_in_state if s in obj_locations}

        # 4. Check if man is carrying a usable spanner
        man_has_usable_spanner_now = (man_carrying_spanner is not None) and (man_carrying_spanner in usable_spanners_in_state)

        # 5. Implement greedy calculation loop
        h = 0
        current_man_location = man_location
        nuts_to_tighten = list(loose_goal_nuts) # Use a list to remove elements
        usable_spanners_available_on_ground_list = list(usable_spanners_on_ground) # List to remove from

        while nuts_to_tighten:
            # Need a spanner?
            if not man_has_usable_spanner_now:
                # Find nearest usable spanner on the ground
                nearest_spanner_loc = None
                min_dist_spanner = float('inf')
                spanner_to_pickup = None

                # Check spanners on the ground
                spanners_on_ground_locations = {}
                for s in usable_spanners_available_on_ground_list:
                     if s in obj_locations:
                         spanners_on_ground_locations[s] = obj_locations[s]

                for s, loc_s in spanners_on_ground_locations.items():
                    # Ensure current location and spanner location are in our distance map
                    if current_man_location in self.distances and loc_s in self.distances[current_man_location]:
                        dist = self.distances[current_man_location][loc_s]
                        if dist < min_dist_spanner:
                            min_dist_spanner = dist
                            nearest_spanner_loc = loc_s
                            spanner_to_pickup = s # Keep track of which spanner object

                if nearest_spanner_loc is None or min_dist_spanner == float('inf'):
                    # No reachable usable spanners available on the ground, and man needs one.
                    # Problem likely unsolvable from here.
                    return float('inf')

                # Walk to spanner and pick it up
                h += min_dist_spanner + 1
                current_man_location = nearest_spanner_loc
                man_has_usable_spanner_now = True
                # Remove the picked up spanner from the available list for future pickups
                if spanner_to_pickup in usable_spanners_available_on_ground_list:
                     usable_spanners_available_on_ground_list.remove(spanner_to_pickup)


            # Need to tighten a nut
            nearest_nut_loc = None
            min_dist_nut = float('inf')
            nut_to_tighten = None

            # Find nearest loose goal nut
            nuts_locations = {}
            for nut in nuts_to_tighten:
                if nut in obj_locations:
                    nuts_locations[nut] = obj_locations[nut]

            for nut, loc_n in nuts_locations.items():
                 # Ensure current location and nut location are in our distance map
                if current_man_location in self.distances and loc_n in self.distances[current_man_location]:
                    dist = self.distances[current_man_location][loc_n]
                    if dist < min_dist_nut:
                        min_dist_nut = dist
                        nearest_nut_loc = loc_n
                        nut_to_tighten = nut

            if nearest_nut_loc is None or min_dist_nut == float('inf'):
                # This implies a remaining nut is unreachable or its location is unknown.
                return float('inf') # Should not be reached in solvable problems with known nut locations

            # Walk to nut and tighten it
            h += min_dist_nut + 1
            current_man_location = nearest_nut_loc
            nuts_to_tighten.remove(nut_to_tighten)
            man_has_usable_spanner_now = False # Spanner is used

        return h
