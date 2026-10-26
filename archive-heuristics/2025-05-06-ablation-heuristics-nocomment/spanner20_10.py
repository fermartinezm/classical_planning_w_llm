import math
from fnmatch import fnmatch
from collections import deque

# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic


# Helper function to parse PDDL fact string
def get_parts(fact):
    """Parses a PDDL fact string into a list of parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()


# Helper function to match fact parts with patterns
def match(fact, *args):
    """Checks if a fact matches a sequence of pattern arguments."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic: # Inherit from Heuristic in actual use
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        Estimates the cost to reach the goal by summing:
        1. The number of loose nuts that need tightening (representing the tighten actions).
        2. The cost of the pickup action if the man needs to acquire a usable spanner.
        3. The estimated walk cost for the man to travel from his current location to acquire a spanner (if needed) and then visit all locations containing loose nuts that need tightening.

    Assumptions:
        - The goal is to tighten a specific set of nuts.
        - There is exactly one man.
        - The man can carry multiple spanners.
        - A spanner becomes unusable after tightening one nut but remains carried.
        - The 'link' predicate defines an undirected graph for movement.
        - All locations and objects mentioned in the initial state and goals exist and are reachable if the graph is connected.
        - The heuristic assumes the man needs *at least one* usable spanner to start tightening nuts. The cost to acquire this first spanner (pickup action + walk to it) is estimated. This spanner is then assumed to be usable for all remaining nuts (a simplification, as spanners break after one use).
        - The walk cost to visit all necessary locations (spanner location if needed, and all unique loose nut locations) is estimated using a Nearest Neighbor approximation, which is not optimal but efficient.

    Heuristic Initialization:
        - Parses the goal facts to identify the set of nuts that must be tightened.
        - Parses static facts (links) to build a graph representation of locations.
        - Computes all-pairs shortest paths (distances) between all locations using BFS. Stores these distances.
        - Identifies the man, all spanners, and all nuts from the initial state facts.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state.
        2. Parse the state to find:
           - The man's current location.
           - Which spanners the man is carrying.
           - Which spanners are currently usable.
           - Which nuts are currently loose and their locations.
           - The locations of spanners that are not carried.
        3. Determine the set of loose nuts that are also goal nuts. If this set is empty, the goal is reached, return 0.
        4. Calculate the base cost: This is the number of loose goal nuts (each needs a tighten action).
        5. Calculate the spanner acquisition cost and identify a potential spanner location to visit:
           - Check if the man is currently carrying *any* usable spanner.
           - If yes, the spanner pickup cost component is 0, and no specific spanner location needs to be added to the walk path for acquisition.
           - If no, find the closest usable spanner that is currently at a location (not carried).
           - If no such spanner exists, the problem is likely unsolvable from this state; return infinity.
           - Otherwise, the spanner pickup cost is 1 (for the pickup action), and the location of this closest usable spanner is added to the set of locations the man needs to visit.
        6. Calculate the estimated walk cost:
           - Identify the set of unique locations where the loose goal nuts are located.
           - Combine this set with the spanner location identified in step 5 (if a spanner needs to be acquired).
           - Use a Nearest Neighbor approximation starting from the man's current location to visit all locations in the combined set. The cost is the sum of distances along this path.
           - If any required location is unreachable from the current location or other required locations in the set, return infinity.
        7. The total heuristic value is the sum of the base cost (number of loose goal nuts), the spanner pickup cost, and the estimated walk cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # 1. Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # 2. Build location graph and compute distances
        self.location_graph = {}
        locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1)

        self.all_locations = list(locations)
        self.distances = self._compute_all_pairs_shortest_paths()

        # 3. Identify man, spanners, nuts from initial state facts
        self.man = None
        self.all_spanners = set()
        self.all_nuts = set()

        # Infer objects and types from initial state facts
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 obj, loc = parts[1:]
                 # Infer type based on common usage or predicate signature
                 if obj.startswith('bob'): # Assuming 'bob' is the man name prefix
                     self.man = obj
                 elif obj.startswith('spanner'): # Assuming 'spanner' prefix
                     self.all_spanners.add(obj)
                 elif obj.startswith('nut'): # Assuming 'nut' prefix
                     self.all_nuts.add(obj)
             elif parts[0] == 'carrying':
                 m, s = parts[1:]
                 self.man = m
                 self.all_spanners.add(s)
             elif parts[0] == 'usable':
                 s = parts[1]
                 self.all_spanners.add(s)
             elif parts[0] == 'loose' or parts[0] == 'tightened':
                 n = parts[1]
                 self.all_nuts.add(n)

        # Ensure goal nuts are added to all_nuts if they weren't in initial state (unlikely but safe)
        self.all_nuts.update(self.goal_nuts)

        # Ensure man is identified (should be in initial state)
        if self.man is None:
             # Problem definition error or unexpected initial state
             # In a real scenario, we might raise an error or handle it.
             # For this heuristic, we'll assume man is always present.
             pass


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest paths between all pairs of locations using BFS."""
        distances = {loc: {l: math.inf for l in self.all_locations} for loc in self.all_locations}

        for start_node in self.all_locations:
            distances[start_node][start_node] = 0
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_loc, current_dist = queue.popleft()

                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_node][neighbor] = current_dist + 1
                            queue.append((neighbor, current_dist + 1))
        return distances

    def __call__(self, node):
        state = node.state

        # 1. Parse state
        man_location = None
        carried_spanners = set()
        usable_spanners_in_state = set()
        loose_nuts_in_state = set()
        nut_locations = {} # nut -> location
        spanner_locations = {} # spanner -> location (only for spanners *at* a location)

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1:]
                if obj == self.man:
                    man_location = loc
                elif obj in self.all_spanners:
                    spanner_locations[obj] = loc
                elif obj in self.all_nuts:
                    nut_locations[obj] = loc
            elif parts[0] == 'carrying':
                m, s = parts[1:]
                if m == self.man:
                    carried_spanners.add(s)
            elif parts[0] == 'usable':
                s = parts[1]
                usable_spanners_in_state.add(s)
            elif parts[0] == 'loose':
                n = parts[1]
                loose_nuts_in_state.add(n)

        # Ensure man location is found
        if man_location is None:
             # Should not happen in a valid state, but indicates an issue
             return math.inf

        # 2. Identify loose goal nuts
        loose_goal_nuts = self.goal_nuts.intersection(loose_nuts_in_state)

        # Goal reached if no loose goal nuts
        if not loose_goal_nuts:
            return 0

        # 3. Calculate base cost (tighten actions)
        total_heuristic = len(loose_goal_nuts)

        # 4. Calculate spanner acquisition cost and potential spanner location to visit
        man_has_usable_spanner = any(s in usable_spanners_in_state for s in carried_spanners)
        spanner_pickup_cost = 0
        spanner_location_to_visit = None

        if not man_has_usable_spanner:
            # Find closest usable spanner that is currently at a location (not carried)
            available_usable_spanners_at_locs = {
                s for s in usable_spanners_in_state
                if s not in carried_spanners and s in spanner_locations # Check if it's at a location
            }

            if not available_usable_spanners_at_locs:
                # No usable spanners available anywhere (not carried, not at location)
                # This state is likely unsolvable if there are loose goal nuts remaining
                return math.inf

            min_spanner_dist = math.inf
            closest_spanner_loc = None

            for s in available_usable_spanners_at_locs:
                 s_loc = spanner_locations.get(s) # Get location from parsed state
                 # Check if locations are in our graph and reachable
                 if man_location in self.distances and s_loc in self.distances[man_location]:
                     dist = self.distances[man_location][s_loc]
                     if dist < min_spanner_dist:
                         min_spanner_dist = dist
                         closest_spanner_loc = s_loc

            if closest_spanner_loc is None or min_spanner_dist == math.inf:
                 # No reachable usable spanner found
                 return math.inf

            spanner_pickup_cost = 1 # Cost of the pickup action
            spanner_location_to_visit = closest_spanner_loc

        total_heuristic += spanner_pickup_cost

        # 5. Calculate estimated walk cost to visit unique loose nut locations and spanner location (if needed)
        locations_to_visit_set = {nut_locations[n] for n in loose_goal_nuts}
        if spanner_location_to_visit:
             locations_to_visit_set.add(spanner_location_to_visit)

        walk_cost = 0
        current_loc = man_location
        remaining_locations = set(locations_to_visit_set)

        # If man is already at a location that needs visiting, start the path from there
        if current_loc in remaining_locations:
            remaining_locations.remove(current_loc)

        # Nearest Neighbor approximation for TSP
        while remaining_locations:
            min_dist = math.inf
            next_loc = None

            if current_loc not in self.distances:
                 # Current location is not in the graph (should not happen)
                 return math.inf

            for target_loc in remaining_locations:
                if target_loc not in self.distances[current_loc]:
                     # Target location is not in the graph (should not happen)
                     return math.inf

                dist = self.distances[current_loc][target_loc]
                if dist == math.inf:
                    # Cannot reach this location from current path
                    return math.inf

                if dist < min_dist:
                    min_dist = dist
                    next_loc = target_loc

            if next_loc is None:
                 # Should not happen if remaining_locations is not empty and reachable
                 return math.inf

            walk_cost += min_dist
            current_loc = next_loc
            remaining_locations.remove(next_loc)

        total_heuristic += walk_cost

        return total_heuristic
