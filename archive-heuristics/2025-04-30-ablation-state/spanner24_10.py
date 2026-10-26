from fnmatch import fnmatch
from collections import defaultdict, deque
# Assuming heuristic_base.py is available in the specified path
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a mock Heuristic class for standalone testing if needed
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


# Helper functions
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

# Heuristic class
class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It considers the cost of tightening actions, picking up spanners, and travel
    between locations. It is designed to be non-admissible to better guide a
    greedy best-first search.

    # Assumptions
    - Actions have unit cost.
    - A spanner becomes unusable after tightening one nut.
    - The man can carry only one spanner at a time.
    - There are enough usable spanners available in the instance to tighten all goal nuts.
      (This is checked, and returns infinity if not).
    - The location graph is connected, allowing travel between any two locations relevant
      to the problem (man's initial location, spanner locations, nut locations).

    # Heuristic Initialization
    - Parses object declarations by examining facts in the initial state and goals
      to identify types (man, spanner, nut, location). This relies on typical PDDL
      structure where objects appear in specific predicate arguments based on type.
    - Parses 'link' facts from static information to build the location graph.
    - Computes all-pairs shortest paths between all identified locations using BFS.
    - Identifies the set of nuts that need to be tightened to reach the goal.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state facts to identify:
       - The man's current location.
       - Whether the man is carrying a spanner, and if it's usable.
       - The location of all objects (spanners, nuts).
       - The usability status of all spanners.
       - The loose/tightened status of all nuts.
    2. Determine the set of nuts that are currently loose and are part of the goal.
    3. If there are no loose goal nuts, the heuristic is 0 (goal state reached).
    4. Count the total number of usable spanners available in the current state
       (carried usable spanner + usable spanners on the ground).
    5. If the number of loose goal nuts exceeds the number of usable spanners, the state
       is likely unsolvable in this domain, return infinity.
    6. Calculate the estimated cost as the sum of several components:
       a.  Action Cost (Tighten): Add 1 for each loose goal nut (representing the `tighten_nut` action).
       b.  Action Cost (Pickup): Add 1 for each spanner that needs to be picked up throughout the plan. This is estimated as the total number of loose goal nuts minus 1 if the man is already carrying a usable spanner at the start of the plan, or 0 if no pickups are needed. This simplifies to `max(0, num_loose_goal_nuts - (1 if carrying usable spanner else 0))`.
       c.  Initial Travel Cost: Estimate the minimum travel cost to get the man to the location of the first nut, potentially including picking up the first spanner if not already carried.
           - If the man is carrying a usable spanner, this is the minimum distance from his current location to any loose goal nut location.
           - If the man is not carrying a usable spanner, this is the minimum distance over all usable spanners on the ground and all loose goal nuts, of (distance from man's location to spanner's location + distance from spanner's location to nut's location). The pickup action cost (1) is accounted for in 6b.
       d.  Inter-Nut Travel Cost: Estimate the travel needed to move between the remaining loose goal nut locations after the first one is visited. This is estimated as (number of loose goal nuts - 1) multiplied by the maximum distance between any two distinct loose goal nut locations. This component makes the heuristic non-admissible and aims to provide a stronger estimate of the required travel compared to using minimum distance.
    7. Sum all calculated costs (6a + 6b + 6c + 6d) to get the final heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and identifying goal nuts.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Extract object types by examining facts in initial state and goals
        # This approach infers types based on predicate usage, assuming standard domain structure.
        self.man_objects = set()
        self.spanner_objects = set()
        self.nut_objects = set()
        self.location_objects = set()
        all_objects = set()

        # Collect all objects mentioned in initial state and goals
        for fact in initial_state | self.goals:
             parts = get_parts(fact)
             all_objects.update(parts[1:]) # Add all arguments as potential objects

        # Categorize objects based on predicates
        for obj in all_objects:
            is_man = any(match(f, "carrying", obj, "*") for f in initial_state | self.goals)
            is_spanner = any(match(f, "carrying", "*", obj) or match(f, "usable", obj) for f in initial_state | self.goals)
            is_nut = any(match(f, "loose", obj) or match(f, "tightened", obj) for f in initial_state | self.goals)
            is_location = any(match(f, "at", "*", obj) or match(f, "link", obj, "*") or match(f, "link", "*", obj) for f in initial_state | self.goals)

            if is_man:
                self.man_objects.add(obj)
            if is_spanner:
                self.spanner_objects.add(obj)
            if is_nut:
                self.nut_objects.add(obj)
            if is_location:
                self.location_objects.add(obj)

        # Build location graph
        self.adj = defaultdict(list)
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations from links are added even if not in initial 'at' facts
                self.location_objects.add(loc1)
                self.location_objects.add(loc2)
                self.adj[loc1].append(loc2)
                self.adj[loc2].append(loc1) # Assuming links are bidirectional

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_node in self.location_objects:
            self.dist[start_node] = self._bfs(start_node)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all reachable nodes."""
        distances = {node: float('inf') for node in self.location_objects}
        if start_node not in distances:
             # This node might not have been added to location_objects, though the logic
             # in __init__ attempts to add all locations from links and 'at' facts.
             # If it's truly not a valid location, distances remain inf.
             pass # Distances initialized to inf is correct.

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Check if u is a valid location and has neighbors in adj
            if u in self.adj:
                for v in self.adj[u]:
                    if v in distances and distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse state facts
        man_loc = None
        carrying_spanner = False
        spanner_carried_obj = None
        object_locations = {}
        spanner_usability = {}
        nut_status = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                object_locations[obj] = loc
                if obj in self.man_objects:
                    man_loc = loc
            elif parts[0] == 'carrying':
                m, s = parts[1], parts[2]
                if m in self.man_objects:
                    carrying_spanner = True
                    spanner_carried_obj = s
            elif parts[0] == 'usable':
                s = parts[1]
                spanner_usability[s] = True
            elif parts[0] == 'loose':
                n = parts[1]
                nut_status[n] = 'loose'
            elif parts[0] == 'tightened':
                n = parts[1]
                nut_status[n] = 'tightened'

        # Identify loose goal nuts and their locations
        loose_goal_nuts = [n for n in self.goal_nuts if nut_status.get(n) == 'loose']
        num_nuts = len(loose_goal_nuts)

        if num_nuts == 0:
            return 0 # Goal reached

        # Check spanner count sufficiency
        usable_spanners_on_ground = [s for s in self.spanner_objects if s in object_locations and spanner_usability.get(s, False)]
        usable_spanners_count = (1 if carrying_spanner and spanner_usability.get(spanner_carried_obj, False) else 0) + len(usable_spanners_on_ground)

        if usable_spanners_count < num_nuts:
            return float('inf') # Unsolvable state based on spanner count

        h = 0

        # 6a. Action Cost (Tighten)
        h += num_nuts

        # 6b. Action Cost (Pickup)
        # Need num_nuts spanners in total. If carrying one usable, need num_nuts - 1 more pickups.
        # If not carrying usable, need num_nuts pickups.
        pickups_needed = num_nuts - (1 if carrying_spanner and spanner_usability.get(spanner_carried_obj, False) else 0)
        h += max(0, pickups_needed)

        # 6c. Initial Travel Cost + First Spanner Acquisition Travel
        min_initial_path_cost = float('inf')
        # Ensure all loose goal nuts have a known location in the current state
        loose_goal_nut_locs = {object_locations[n] for n in loose_goal_nuts if n in object_locations}

        if man_loc is None or not loose_goal_nut_locs:
             # Man's location unknown or no loose goal nuts with known locations - unsolvable
             return float('inf')

        if carrying_spanner and spanner_usability.get(spanner_carried_obj, False):
            # Man is carrying a usable spanner, just needs to travel to the nearest nut
            for nut_loc in loose_goal_nut_locs:
                if man_loc in self.dist and nut_loc in self.dist[man_loc]:
                     min_initial_path_cost = min(min_initial_path_cost, self.dist[man_loc][nut_loc])
        else:
            # Man needs to pick up a spanner first, then go to a nut
            # Ensure all usable spanners on ground have a known location
            usable_spanners_on_ground_locs = {object_locations[s] for s in usable_spanners_on_ground if s in object_locations}
            if not usable_spanners_on_ground_locs:
                 # Should not happen if usable_spanners_count >= num_nuts > 0 and not carrying,
                 # but handle defensively. Implies usable spanners exist but their location is unknown/unreachable.
                 return float('inf')

            for spanner_loc in usable_spanners_on_ground_locs:
                for nut_loc in loose_goal_nut_locs:
                    if man_loc in self.dist and spanner_loc in self.dist[man_loc] and nut_loc in self.dist[spanner_loc]:
                         # Travel man -> spanner + Travel spanner -> nut
                         min_initial_path_cost = min(min_initial_path_cost, self.dist[man_loc][spanner_loc] + self.dist[spanner_loc][nut_loc])

        # If min_initial_path_cost is still infinity, it means some required location is unreachable.
        if min_initial_path_cost == float('inf'):
             return float('inf')

        h += min_initial_path_cost

        # 6d. Inter-Nut Travel Cost
        if num_nuts > 1:
            max_dist_between_any_two_nuts = 0
            nut_loc_list = list(loose_goal_nut_locs)
            # Only calculate max distance if there are at least two distinct locations
            if len(nut_loc_list) > 1:
                for i in range(len(nut_loc_list)):
                    for j in range(i + 1, len(nut_loc_list)):
                        loc1 = nut_loc_list[i]
                        loc2 = nut_loc_list[j]
                        # Ensure locations are in the distance map (reachable)
                        if loc1 in self.dist and loc2 in self.dist[loc1]:
                             max_dist_between_any_two_nuts = max(max_dist_between_any_two_nuts, self.dist[loc1][loc2])
                        else:
                             # If locations are not mutually reachable, this path is impossible
                             # This might indicate an unsolvable state if these nuts must be tightened.
                             # However, the spanner count check and initial path check cover some reachability.
                             # For simplicity, assume graph is connected for relevant locations if not caught earlier.
                             pass # max_dist remains 0 or previous max

                # If max_dist_between_any_two_nuts is still 0 but len(nut_loc_list) > 1,
                # it means all pairs of distinct nut locations are unreachable from each other.
                # This implies unsolvability if multiple nuts are at different, disconnected locations.
                # The initial path cost check might catch this if the first nut is unreachable.
                # If not caught, adding (num_nuts - 1) * 0 is harmless but might be misleading.
                # Let's rely on the initial path cost check and spanner count check for unsolvability.

            h += (num_nuts - 1) * max_dist_between_any_two_nuts

        return h
