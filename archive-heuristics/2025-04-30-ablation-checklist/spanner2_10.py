from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Basic check for number of parts if no wildcard is used for the predicate name
    if args and not any('*' in arg for arg in args) and len(parts) != len(args):
         return False
    # Match parts with args, allowing args to be shorter if parts match up to that point
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all required nuts.
    The heuristic considers the number of nuts to tighten, the number of
    spanners that need to be picked up, and the estimated travel cost
    for the man to visit all necessary locations (nut locations and spanner
    pickup locations).

    This heuristic is non-admissible but aims to be informative for greedy
    best-first search by estimating the remaining work based on required
    actions (tighten, pickup) and necessary movement.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances between locations
        and identifying goal nuts.

        Args:
            task: The planning task object containing domain information,
                  initial state, goals, operators, and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.objects = task.objects # Needed to identify locations and the man

        # Identify goal nuts (those that need to be tightened)
        # We assume goals are conjunctions of (tightened nutX) facts.
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'tightened'}

        # Identify all locations from task objects
        self.locations = {obj.name for obj in self.objects if obj.type == 'location'}

        # Precompute shortest path distances between all locations using BFS
        self.location_distances = self._precompute_distances()

        # Find the man object (assuming there is exactly one man)
        self.man_name = None
        for obj in self.objects:
            if obj.type == 'man':
                self.man_name = obj.name
                break
        # It's assumed a valid task will have a man. If not, subsequent steps
        # might fail or return infinity if man_name is None.

    def _precompute_distances(self):
        """
        Build a graph from link facts and compute all-pairs shortest paths
        using BFS.

        Returns:
            A dictionary `distances[loc1][loc2]` storing the shortest path
            distance between loc1 and loc2. Returns float('inf') if unreachable.
        """
        locations = list(self.locations) # Use the locations identified in __init__
        adj = {loc: set() for loc in locations} # Initialize adjacency for all locations

        # Build adjacency list from link facts
        for fact in self.static_facts:
            if match(fact, 'link', '*', '*'):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations from links are valid locations from task.objects
                if loc1 in self.locations and loc2 in self.locations:
                    adj[loc1].add(loc2)
                    adj[loc2].add(loc1) # Links are bidirectional
                # else: Link involves an object not defined as a location? Ignore.

        # Compute distances using BFS from each location
        distances = {}
        for start_loc in locations:
            distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                curr_loc, d = q.popleft()
                distances[start_loc][curr_loc] = d

                # Check if curr_loc is a valid location and has neighbors in the graph
                if curr_loc in adj: # adj only contains keys for locations with links
                    for neighbor in adj[curr_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))

        # Fill in unreachable locations with infinity
        for l1 in locations:
            for l2 in locations:
                if l2 not in distances.get(l1, {}): # Use .get for safety if start_loc had no links
                    distances.setdefault(l1, {})[l2] = float('inf')

        return distances

    def get_distance(self, loc1, loc2):
        """
        Helper to get precomputed distance.

        Args:
            loc1: The starting location name.
            loc2: The ending location name.

        Returns:
            The shortest distance between loc1 and loc2, or float('inf')
             if either location is invalid or unreachable.
        """
        # Check if locations exist in our precomputed map
        if loc1 not in self.location_distances or loc2 not in self.location_distances.get(loc1, {}):
             # This means one or both locations were not found among task objects of type location,
             # or the distance wasn't computed (e.g., disconnected graph).
             # If a required location is unreachable, the state is likely a dead end.
             return float('inf')
        return self.location_distances[loc1][loc2]


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.

        Args:
            node: The current search node, containing the state.

        Returns:
            The estimated heuristic cost (non-negative integer or float('inf')).
        """
        state = node.state

        # 1. Identify loose goal nuts and their locations
        # These are the nuts that are goals (need to be tightened) and are currently loose.
        loose_goal_nuts = {
            n for n in self.goal_nuts
            if '(loose ' + n + ')' in state
        }
        num_nuts_to_tighten = len(loose_goal_nuts)

        if num_nuts_to_tighten == 0:
            return 0 # All goal nuts are tightened

        nut_locations = {}
        for nut in loose_goal_nuts:
             # Find location of the nut in the current state
             loc = next((get_parts(f)[2] for f in state if match(f, 'at', nut, '*')), None)
             if loc is None:
                 # A loose goal nut is not at any location? Indicates a problematic state.
                 return float('inf')
             nut_locations[nut] = loc

        # 2. Identify man's current location
        # Assuming self.man_name was successfully found in __init__
        if self.man_name is None:
             # No man object found in the task - cannot solve.
             return float('inf')

        man_location = next((get_parts(f)[2] for f in state if match(f, 'at', self.man_name, '*')), None)
        if man_location is None:
             # Man is not at any location? Problematic state.
             return float('inf')


        # 3. Identify usable spanners (held and ground) and ground spanner locations
        # Spanners the man is currently carrying and are usable.
        usable_spanners_held = {
            get_parts(f)[2] for f in state
            if match(f, 'carrying', self.man_name, '*') and '(usable ' + get_parts(f)[2] + ')' in state
        }
        # Usable spanners currently on the ground at some location.
        usable_spanners_ground_facts = {
             f for f in state
             if match(f, 'at', '*', '*') and '(usable ' + get_parts(f)[1] + ')' in state
        }

        spanner_locations_ground = {
            get_parts(f)[1]: get_parts(f)[2] for f in usable_spanners_ground_facts
        }
        usable_spanners_ground = set(spanner_locations_ground.keys())


        num_spanners_held_usable = len(usable_spanners_held)
        num_spanners_on_ground_usable = len(usable_spanners_ground)
        num_spanners_available_total = num_spanners_held_usable + num_spanners_on_ground_usable

        # 4. Check solvability based on spanners
        # We need one usable spanner for each nut to tighten.
        if num_spanners_available_total < num_nuts_to_tighten:
             # Not enough usable spanners exist in the world to tighten all required nuts.
             # This state is a dead end.
             return float('inf')

        # 5. Estimate cost components

        # Cost component 1: Tighten actions
        # Each loose goal nut requires one tighten action.
        cost = num_nuts_to_tighten

        # Cost component 2: Pickup actions
        # The man needs to pick up a spanner for each nut he tightens, unless he is already
        # carrying enough usable spanners.
        spanners_to_pickup_count = max(0, num_nuts_to_tighten - num_spanners_held_usable)
        cost += spanners_to_pickup_count # Each pickup action costs 1

        # Cost component 3: Travel
        # The man needs to travel to:
        # a) The location of each loose goal nut (to perform the tighten action).
        # b) The location of each spanner he needs to pick up from the ground.

        # The set of locations the man must visit. Starts with nut locations.
        required_locations = set(nut_locations.values())

        # Identify the locations of the spanners he needs to pick up.
        # For simplicity, we assume he picks up the 'spanners_to_pickup_count'
        # usable spanners on the ground that are closest to his *current* location.
        spanners_on_ground_list = [(s, l) for s, l in spanner_locations_ground.items()]

        # Sort ground spanners by distance from the man's current location.
        # This is a greedy choice for estimating travel, not necessarily the optimal
        # sequence of pickups and visits.
        # Ensure man_location is a valid start for distance calculation.
        if man_location not in self.location_distances:
             # Man is in an invalid location? Problematic state.
             return float('inf')

        spanners_on_ground_list.sort(key=lambda item: self.get_distance(man_location, item[1]))

        # Add the locations of the first 'spanners_to_pickup_count' spanners
        # from the sorted list to the set of required locations.
        spanners_added_count = 0
        for spanner, loc in spanners_on_ground_list:
             if spanners_added_count < spanners_to_pickup_count:
                 # Ensure the spanner location is a valid location in the graph
                 if loc in self.locations:
                     required_locations.add(loc)
                     spanners_added_count += 1
                 # else: Ignore spanner at invalid location? Assume valid problem.
             else:
                 break # Added enough spanner pickup locations

        # Estimate travel cost to visit all required locations starting from man_location.
        # We use a greedy approach: always move to the closest unvisited required location.
        travel_cost = 0
        current_loc = man_location
        unvisited_locations = set(required_locations)

        while unvisited_locations:
            closest_loc = None
            min_d = float('inf')

            for loc in unvisited_locations:
                d = self.get_distance(current_loc, loc)
                if d < min_d:
                    min_d = d
                    closest_loc = loc

            if closest_loc is None or min_d == float('inf'):
                 # This means a required location is unreachable from the current location set.
                 # This state is likely a dead end.
                 return float('inf')

            travel_cost += min_d
            current_loc = closest_loc
            unvisited_locations.remove(closest_loc)

        cost += travel_cost

        return cost

