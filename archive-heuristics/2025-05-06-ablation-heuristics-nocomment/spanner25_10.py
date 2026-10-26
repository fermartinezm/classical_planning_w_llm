from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal by summing:
    1. The number of loose nuts that still need to be tightened (i.e., goal nuts
       that are not yet tightened).
    2. The number of additional usable spanners the man needs to pick up
       to have one for each remaining loose nut.
    3. The shortest path distance from the man's current location to the
       closest relevant location (either a location with an available
       usable spanner if spanners are needed, or a location with a
       loose nut that needs tightening).

    Assumptions:
    - The problem is solvable.
    - There is exactly one man object.
    - Links between locations are bidirectional.
    - A usable spanner is consumed (becomes unusable) after tightening one nut.
    - The goal is to tighten a specific set of nuts.
    - Object types (man, spanner, nut, location) can be identified from task.facts.

    Heuristic Initialization:
    - Parses object types (man, spanner, nut, location) from task.facts.
    - Identifies the name of the man object.
    - Identifies the names of the nuts that are part of the goal (need to be tightened).
    - Builds a graph of locations based on (link ?l1 ?l2) static facts,
      treating links as bidirectional.
    - Computes all-pairs shortest paths between locations using BFS and stores
      them in a dictionary `self.distances`.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to find:
       - The man's current location.
       - The set of nuts that are currently tightened.
       - The set of spanners the man is currently carrying.
       - The set of spanners that are currently usable.
       - The locations of all spanners currently at a location (not carried).
       - The locations of all nuts currently at a location.
    2. Identify the set of goal nuts that have not yet been tightened.
    3. If this set is empty, the goal is reached, return 0.
    4. Calculate the number of untightened goal nuts (`N_untightened_goal_nuts`).
    5. Calculate the number of usable spanners the man is currently carrying (`N_carried_usable`).
    6. Calculate the number of additional usable spanners needed:
       `NeededSpannersToPickUp = max(0, N_untightened_goal_nuts - N_carried_usable)`.
    7. Initialize the heuristic value `h` with the sum of required actions:
       `h = N_untightened_goal_nuts` (for the tighten actions)
       `h += NeededSpannersToPickUp` (for the pickup actions).
    8. Identify the locations of the untightened goal nuts (`NutLocations`).
    9. Identify the locations of available usable spanners (those at a location and usable) (`AvailableUsableSpannerLocations`).
    10. Determine the set of target locations the man needs to reach. This set includes:
        - All `NutLocations` (for the nuts that need tightening).
        - If `NeededSpannersToPickUp > 0`, it also includes all `AvailableUsableSpannerLocations`.
    11. Calculate the minimum shortest path distance from the man's current location to any location in the set of target locations.
    12. Add this minimum distance to the heuristic value `h`.
    13. Return `h`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static
        all_facts = task.facts # Contains type facts like (man bob)

        # 1. Parse object types and names
        self.object_types = {}
        self.man_name = None
        self.spanner_names = set()
        self.nut_names = set()
        self.location_names = set()

        for fact_str in all_facts:
            parts = self.get_parts(fact_str)
            if len(parts) == 2:
                obj_type, obj_name = parts
                self.object_types[obj_name] = obj_type
                if obj_type == 'man':
                    self.man_name = obj_name
                elif obj_type == 'spanner':
                    self.spanner_names.add(obj_name)
                elif obj_type == 'nut':
                    self.nut_names.add(obj_name)
                elif obj_type == 'location':
                    self.location_names.add(obj_name)

        # Identify goal nuts
        self.goal_nuts_to_tighten = {self.get_parts(g)[1] for g in self.goals if self.match(g, "tightened", "*")}


        # 2. Build location graph from static link facts
        graph = {}
        for loc in self.location_names:
            graph[loc] = []

        for fact_str in static_facts:
            if self.match(fact_str, "link", "*", "*"):
                _, l1, l2 = self.get_parts(fact_str)
                if l1 in graph and l2 in graph: # Ensure locations are known
                    graph[l1].append(l2)
                    graph[l2].append(l1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths
        self.distances = {}
        for start_node in self.location_names:
            self.distances[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                current_node, dist = queue.popleft()
                self.distances[start_node][current_node] = dist
                for neighbor in graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

    @staticmethod
    def get_parts(fact):
        """Helper to parse a fact string into its components."""
        # Remove surrounding parentheses and split by space
        return fact[1:-1].split()

    @staticmethod
    def match(fact, *args):
        """Helper to check if a fact matches a predicate pattern."""
        parts = spannerHeuristic.get_parts(fact)
        # Check if the number of parts matches args, and if each part matches the pattern
        return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def __call__(self, node):
        state = node.state

        # 1. Parse current state information
        man_loc = None
        tightened_nuts = set()
        carried_spanners = set()
        usable_spanners = set()
        spanner_locations = {} # map spanner name to location
        nut_locations_in_state = {} # map nut name to location

        for fact_str in state:
            parts = self.get_parts(fact_str)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj == self.man_name:
                    man_loc = loc
                elif obj in self.spanner_names:
                    spanner_locations[obj] = loc
                elif obj in self.nut_names:
                    nut_locations_in_state[obj] = loc
            elif parts[0] == 'tightened' and len(parts) == 2:
                nut = parts[1]
                if nut in self.nut_names:
                    tightened_nuts.add(nut)
            elif parts[0] == 'carrying' and len(parts) == 3:
                 # Assuming (carrying man spanner)
                 man, spanner = parts[1], parts[2]
                 if man == self.man_name and spanner in self.spanner_names:
                     carried_spanners.add(spanner)
            elif parts[0] == 'usable' and len(parts) == 2:
                 spanner = parts[1]
                 if spanner in self.spanner_names:
                     usable_spanners.add(spanner)

        # 2. Identify untightened goal nuts
        untightened_goal_nuts = {n for n in self.goal_nuts_to_tighten if n not in tightened_nuts}
        n_untightened_goal_nuts = len(untightened_goal_nuts)

        # 3. Goal check
        if n_untightened_goal_nuts == 0:
            return 0

        # 4. Calculate required spanners and pickups
        carried_usable_spanners = carried_spanners.intersection(usable_spanners)
        n_carried_usable = len(carried_usable_spanners)
        needed_spanners_to_pick_up = max(0, n_untightened_goal_nuts - n_carried_usable)

        # 7. Initialize heuristic with action counts
        h = n_untightened_goal_nuts  # Cost for tighten actions
        h += needed_spanners_to_pick_up # Cost for pickup actions

        # 8. Identify relevant locations
        # Only consider locations of goal nuts that are not yet tightened
        nut_locations = {nut_locations_in_state[n] for n in untightened_goal_nuts if n in nut_locations_in_state}
        available_usable_spanner_locations = {l for s, l in spanner_locations.items() if s in usable_spanners}

        # 10. Determine target locations for walk cost
        target_locations = set()
        if needed_spanners_to_pick_up > 0:
            target_locations.update(available_usable_spanner_locations)
        if n_untightened_goal_nuts > 0: # Always true if we reach here
            target_locations.update(nut_locations)

        # 11. Calculate minimum distance to a target location
        min_dist_to_target = float('inf')

        # Ensure man_loc is valid and in the graph distances
        if man_loc is not None and man_loc in self.distances:
            for loc in target_locations:
                 if loc in self.distances.get(man_loc, {}): # Ensure target location is reachable
                     min_dist_to_target = min(min_dist_to_target, self.distances[man_loc][loc])

        # 12. Add walk cost to heuristic
        # If min_dist_to_target is still inf, it means required locations are unreachable from man_loc.
        # For solvable problems, this shouldn't happen if target_locations is not empty.
        # target_locations is not empty if n_untightened_goal_nuts > 0 (since nut_locations will contain at least one location).
        # The only edge case is if man_loc is not in self.distances, which implies man is in an isolated location not in the link graph.
        if min_dist_to_target != float('inf'):
             h += min_dist_to_target
        # else: The heuristic remains the action count. This might underestimate if walk is required but unreachable,
        # but for solvable problems, reachability is expected.

        # Ensure heuristic is non-negative
        return max(0, h)
