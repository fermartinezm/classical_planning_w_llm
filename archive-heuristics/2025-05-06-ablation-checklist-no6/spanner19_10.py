from fnmatch import fnmatch
from collections import deque
import math # For infinity

# Assume Heuristic base class is available as in the examples
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass
        # Add dummy methods if needed for testing, e.g., __lt__, __eq__
        def __lt__(self, other):
             return self.__call__(None) < other.__call__(None)
        def __eq__(self, other):
             return self.__call__(None) == other.__call__(None)


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

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost based on the number of nuts still needing tightening,
    the number of spanners needing to be picked up, and the estimated walk
    distance for the man to visit necessary locations (nut locations and
    spanner pickup locations).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and
        precomputing shortest path distances.
        """
        self.goals = task.goals
        self.static = task.static
        self.task_objects = task.objects # Store task objects to find man, nuts, spanners by type

        # Identify man object name
        self.man_name = None
        for obj_name, obj_type in self.task_objects:
            if obj_type == 'man':
                self.man_name = obj_name
                break
        # In a real scenario, handle if man_name is None (invalid task)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

        # Build location graph and compute distances
        self._build_location_graph()

    def _build_location_graph(self):
        """Build adjacency list and compute all-pairs shortest paths."""
        locations = set()
        # Collect all locations from link facts
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)

        # Add locations from initial state 'at' facts to ensure all relevant locations are nodes
        # Note: Accessing initial_state from __init__ might require passing it or accessing task.initial_state
        # If task object provides initial_state, use it. Otherwise, rely on links covering the graph.
        # Assuming locations in links cover the movable graph.

        self.locations = list(locations)
        self.loc_to_idx = {loc: i for i, loc in enumerate(self.locations)}
        self.idx_to_loc = {i: loc for i, loc in enumerate(self.locations)}
        num_locs = len(self.locations)

        # Build adjacency list
        adj = {loc: [] for loc in self.locations}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                if loc1 in adj and loc2 in adj: # Ensure locations are in our graph nodes
                    adj[loc1].append(loc2)
                    adj[loc2].append(loc1) # Links are bidirectional for walk

        self.distances = {}
        # Compute shortest paths from each location using BFS
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc, adj)

    def _bfs(self, start_loc, adj):
        """Perform BFS from start_loc to find distances to all reachable locations."""
        distances = {loc: math.inf for loc in self.locations}
        if start_loc in distances: # Ensure start_loc is in our graph nodes
            distances[start_loc] = 0
            queue = deque([start_loc])

            while queue:
                curr_loc = queue.popleft()
                curr_dist = distances[curr_loc]

                for neighbor in adj.get(curr_loc, []):
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = curr_dist + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Get the precomputed shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # This means loc1 or loc2 is not a node in our graph.
             # This can happen if an object is placed at a location not connected by links.
             # Treat as unreachable for movement purposes.
             return math.inf
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse state to find object locations and status
        current_locations = {} # {obj_name: loc_name}
        carried_status = {} # {man_name: {spanner_name: True}}
        usable_status = set() # {spanner_name}
        loose_status = set() # {nut_name}
        tightened_status = set() # {nut_name}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at":
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif predicate == "carrying":
                 man, spanner = parts[1], parts[2]
                 if man not in carried_status:
                     carried_status[man] = {}
                 carried_status[man][spanner] = True
            elif predicate == "usable":
                 spanner = parts[1]
                 usable_status.add(spanner)
            elif predicate == "loose":
                 nut = parts[1]
                 loose_status.add(nut)
            elif predicate == "tightened":
                 nut = parts[1]
                 tightened_status.add(nut)

        # Check if all goal nuts are tightened
        all_goals_tightened = True
        for goal_nut in self.goal_nuts:
             if goal_nut not in tightened_status:
                 all_goals_tightened = False
                 break

        if all_goals_tightened:
             return 0

        # 1. Identify loose goal nuts and their locations
        loose_goal_nuts_with_loc = []
        for nut in self.goal_nuts:
             if nut in loose_status:
                 if nut in current_locations:
                     loose_goal_nuts_with_loc.append((nut, current_locations[nut]))
                 # else: loose goal nut exists but has no location? Problematic state.
                 # Assuming all locatable objects have an 'at' fact.

        N_loose_goals = len(loose_goal_nuts_with_loc)

        # If N_loose_goals is 0 but not all goals are tightened, it implies
        # some goal nuts are neither loose nor tightened. This heuristic assumes
        # goal nuts are either loose or tightened. If N_loose_goals > 0, proceed.
        # If N_loose_goals == 0 but all_goals_tightened is False, it's an invalid state
        # for this heuristic's assumptions. Return infinity.
        if N_loose_goals == 0 and not all_goals_tightened:
             return math.inf


        # 2. Find man's location
        man_loc = current_locations.get(self.man_name)

        if man_loc is None or man_loc not in self.locations:
             # Man is not at a known location in the graph. Unreachable targets.
             return math.inf

        # 3. Find usable spanners carried by man and on the ground
        usable_spanners_carried = [
            s for s in carried_status.get(self.man_name, {}) if s in usable_status
        ]

        usable_spanners_on_ground = [] # List of (spanner_name, location)
        for obj_name, obj_loc in current_locations.items():
             # Identify spanner objects by type from task objects
             is_spanner = False
             for task_obj_name, task_obj_type in self.task_objects:
                 if task_obj_name == obj_name and task_obj_type == 'spanner':
                     is_spanner = True
                     break

             if is_spanner:
                 # Check if it's on the ground (not carried by man) and usable
                 is_carried = self.man_name in carried_status and obj_name in carried_status[self.man_name]
                 if not is_carried and obj_name in usable_status:
                     usable_spanners_on_ground.append((obj_name, obj_loc))


        num_usable_spanners_carried = len(usable_spanners_carried)
        num_usable_spanners_on_ground = len(usable_spanners_on_ground)
        total_usable_spanners_available = num_usable_spanners_carried + num_usable_spanners_on_ground

        # Check if enough spanners exist in total for the remaining loose goals
        if total_usable_spanners_available < N_loose_goals:
             # Unsolvable from this state
             return math.inf

        # 4. Calculate number of spanners to pickup
        num_spanners_to_pickup = max(0, N_loose_goals - num_usable_spanners_carried)

        # 5. Identify locations to visit
        target_locations = set(loc for nut, loc in loose_goal_nuts_with_loc)

        # Filter reachable usable spanners on ground and sort by distance from man
        reachable_usable_spanners_on_ground = [
            (s, loc) for s, loc in usable_spanners_on_ground if self.get_distance(man_loc, loc) != math.inf
        ]

        # If we need more spanners than are reachable on the ground, and we don't carry enough, it's unsolvable.
        # This check is slightly redundant with the total_usable_spanners_available check if all locations are connected,
        # but important if the graph is disconnected.
        if num_spanners_to_pickup > len(reachable_usable_spanners_on_ground):
             return math.inf # Not enough reachable usable spanners

        # Sort reachable usable spanners on ground by distance from man
        reachable_usable_spanners_on_ground.sort(key=lambda item: self.get_distance(man_loc, item[1]))

        # Add locations of the required number of closest usable spanners to target locations
        for i in range(min(num_spanners_to_pickup, len(reachable_usable_spanners_on_ground))):
             target_locations.add(reachable_usable_spanners_on_ground[i][1])

        # If any target location is unreachable from man_loc, it's unsolvable
        for loc in target_locations:
             if self.get_distance(man_loc, loc) == math.inf:
                 return math.inf

        # 6. Calculate estimated walk cost using nearest neighbor
        estimated_walk_cost = 0
        curr_loc = man_loc
        # Convert set to list for indexed removal in nearest neighbor calculation
        remaining_visits = list(target_locations)

        while remaining_visits:
            min_dist = math.inf
            next_loc = None
            next_idx = -1

            for i, target_loc in enumerate(remaining_visits):
                dist = self.get_distance(curr_loc, target_loc)
                if dist < min_dist:
                    min_dist = dist
                    next_loc = target_loc
                    next_idx = i

            # Should not happen if unreachable targets were filtered, but as a safeguard
            if min_dist == math.inf:
                 return math.inf

            estimated_walk_cost += min_dist
            curr_loc = next_loc
            remaining_visits.pop(next_idx)

        # 7. Calculate total heuristic value
        # h = (tighten actions) + (pickup actions) + (walk actions)
        h = N_loose_goals + num_spanners_to_pickup + estimated_walk_cost

        return h
