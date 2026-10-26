from heuristics.heuristic_base import Heuristic
from task import Operator, Task

import heapq
import logging
from collections import deque

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum required
    actions: the number of nuts to tighten, the number of spanners the man
    needs to pick up, and an estimate of the travel cost. The travel cost
    is approximated as the shortest distance from the man's current location
    to the closest required location (either a nut location or a spanner
    pickup location if spanners are needed).

    Assumptions:
    - The task is solvable (enough usable spanners exist in total).
    - There is exactly one man object.
    - Links between locations are bidirectional.
    - All relevant locations are defined as 'location' objects in the task file.
    - The goal only consists of (tightened ?n) facts for some nuts ?n.
    - Objects defined as 'spanner' in task.objects are spanners.
    - Objects defined as 'nut' in task.objects are nuts.

    Heuristic Initialization:
    1. Identify the man object name from the task's initial state or objects.
    2. Collect all location object names from task.objects.
    3. Build the location graph (adjacency list) from static (link) facts, only including known locations.
    4. Compute all-pairs shortest paths between all known locations using BFS. Store
       these distances in a dictionary for quick lookup during heuristic computation.
    5. Identify the set of goal nuts from task.goals.
    6. Precompute the static location for each nut from the initial state.
    7. Identify all spanner object names from task.objects.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state (frozenset of facts).
    2. Find the man's current location by iterating through state facts. If not found, return infinity (invalid state).
    3. Identify the set of loose nuts in the state that are also goal nuts.
    4. Count the number of loose goal nuts (`num_loose_goal_nuts`).
    5. If `num_loose_goal_nuts` is 0, the goal is reached, return 0.
    6. Identify the set of spanners the man is currently carrying by iterating through state facts.
    7. Identify the set of usable spanners by iterating through state facts.
    8. Identify the set of usable spanners the man is currently carrying (`carried_usable_spanners`).
    9. Count the number of carried usable spanners (`num_carried_usable_spanners`).
    10. Calculate the number of additional usable spanners the man needs to pick up:
        `needed_spanners_to_pickup = max(0, num_loose_goal_nuts - num_carried_usable_spanners)`.
    11. Identify the locations of all loose goal nuts using the precomputed nut locations.
    12. Identify the locations of all usable spanners that are currently at a location
        (not carried by the man) by iterating through state facts.
    13. Determine the set of required locations the man might need to visit. This includes
        all locations of loose goal nuts. If `needed_spanners_to_pickup > 0`, it also includes
        locations identified in step 12 that have available usable spanners.
    14. Calculate the minimum shortest distance from the man's current location
        to any of the required locations identified in step 13 using the precomputed distances.
        If there are no required locations (e.g., all goal nuts are tightened), this distance is 0.
        If the man's location is not in the distance map or no required location is reachable, return infinity (dead end).
    15. The heuristic value is the sum of:
        - `num_loose_goal_nuts` (minimum tighten actions)
        - `needed_spanners_to_pickup` (minimum pickup actions)
        - The minimum travel distance calculated in step 14.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.man_name = self._get_man_name(task.initial_state, task.objects)
        self.all_locations = self._get_all_locations(task.objects)
        self.adj_list, _ = self._build_location_graph_from_objects_and_static(task.objects, task.static)
        self.distances = self._compute_all_pairs_shortest_paths(self.adj_list, self.all_locations)
        self.goal_nuts = self._get_goal_nuts(task.goals)
        self.nut_locations = self._get_nut_locations(task.initial_state, task.objects) # Precompute nut locations
        self.all_spanners = self._get_all_spanners(task.objects) # Precompute spanner names


    def _parse_fact(self, fact_str):
        # Remove outer parentheses and split by space
        parts = fact_str[1:-1].split()
        predicate = parts[0]
        objects = parts[1:]
        return predicate, objects

    def _get_man_name(self, initial_state, task_objects):
        # Find the man object name
        for fact_str in initial_state:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'at':
                obj_name = objs[0]
                # Check if this object is a man
                for obj_def in task_objects:
                     if obj_def.split(" - ")[0] == obj_name and obj_def.split(" - ")[1] == "man":
                         return obj_name
        # Fallback: Find first object of type man from task.objects
        for obj_def in task_objects:
             parts = obj_def.split(" - ")
             if len(parts) == 2 and parts[1] == "man":
                 return parts[0]
        return None # Should not happen in a valid spanner domain task

    def _get_all_locations(self, task_objects):
        locations = set()
        for obj_def in task_objects:
            parts = obj_def.split(" - ")
            if len(parts) == 2 and parts[1] == "location":
                locations.add(parts[0])
        return list(locations)

    def _build_location_graph_from_objects_and_static(self, task_objects, static_facts):
        locations = self._get_all_locations(task_objects)
        adj_list = {loc: [] for loc in locations}
        for fact_str in static_facts:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'link':
                l1, l2 = objs
                if l1 in adj_list and l2 in adj_list: # Only add links between known locations
                    adj_list[l1].append(l2)
                    adj_list[l2].append(l1) # Assuming links are bidirectional
        return adj_list, locations

    def _bfs(self, start_loc, adj_list, locations):
        distances = {loc: float('inf') for loc in locations}
        if start_loc not in locations:
             # If start_loc is not a known location, it's unreachable from itself.
             # This case should ideally not happen if locations are correctly identified.
             return distances # All distances remain inf

        distances[start_loc] = 0
        queue = deque([start_loc]) # Use deque for efficient pop(0)
        while queue:
            curr_loc = queue.popleft() # Use popleft for BFS
            if curr_loc in adj_list: # Check if curr_loc has neighbors
                for neighbor in adj_list[curr_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[curr_loc] + 1
                        queue.append(neighbor)
        return distances

    def _compute_all_pairs_shortest_paths(self, adj_list, locations):
        all_distances = {}
        for start_loc in locations:
            all_distances[start_loc] = self._bfs(start_loc, adj_list, locations)
        return all_distances

    def _get_goal_nuts(self, goals):
        goal_nuts = set()
        # Goals are typically a frozenset of facts like '(tightened nut1)'
        for goal_fact_str in goals:
            pred, objs = self._parse_fact(goal_fact_str)
            if pred == 'tightened' and len(objs) == 1:
                goal_nuts.add(objs[0])
        return goal_nuts

    def _get_nut_locations(self, initial_state, task_objects):
        nut_locations = {}
        all_nuts = set()
        # Get all nut object names
        for obj_def in task_objects:
            parts = obj_def.split(" - ")
            if len(parts) == 2 and parts[1] == "nut":
                all_nuts.add(parts[0])

        # Find initial location for each nut
        for fact_str in initial_state:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'at' and len(objs) == 2 and objs[0] in all_nuts:
                nut_locations[objs[0]] = objs[1]
        return nut_locations

    def _get_all_spanners(self, task_objects):
         spanners = set()
         for obj_def in task_objects:
             parts = obj_def.split(" - ")
             if len(parts) == 2 and parts[1] == "spanner":
                 spanners.add(parts[0])
         return spanners


    def __call__(self, node):
        state = node.state

        # 1. Find man's current location
        man_loc = None
        for fact_str in state:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'at' and len(objs) == 2 and objs[0] == self.man_name:
                man_loc = objs[1]
                break
        if man_loc is None:
             # Man's location not found in state - indicates an invalid state
             return float('inf')

        # 2. Identify loose goal nuts in the state
        loose_goal_nuts_in_state = set()
        for fact_str in state:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'loose' and len(objs) == 1 and objs[0] in self.goal_nuts:
                 loose_goal_nuts_in_state.add(objs[0])

        # 3. Count loose goal nuts
        num_loose_goal_nuts = len(loose_goal_nuts_in_state)

        # 4. If goal reached, return 0
        if num_loose_goal_nuts == 0:
            return 0

        # 5. Identify carried usable spanners and available usable spanners at locations
        carried_spanners = set()
        usable_spanners = set() # All usable spanners (carried or at location)
        spanner_at_location_facts = {} # Map spanner name to location if at a location

        for fact_str in state:
            pred, objs = self._parse_fact(fact_str)
            if pred == 'carrying' and len(objs) == 2 and objs[0] == self.man_name and objs[1] in self.all_spanners:
                carried_spanners.add(objs[1])
            elif pred == 'usable' and len(objs) == 1 and objs[0] in self.all_spanners:
                usable_spanners.add(objs[0])
            elif pred == 'at' and len(objs) == 2 and objs[0] in self.all_spanners:
                 spanner_at_location_facts[objs[0]] = objs[1]


        carried_usable_spanners = carried_spanners.intersection(usable_spanners)
        num_carried_usable_spanners = len(carried_usable_spanners)

        # 6. Calculate needed spanners to pickup
        needed_spanners_to_pickup = max(0, num_loose_goal_nuts - num_carried_usable_spanners)

        # 7. Identify locations of loose goal nuts
        loose_nut_locations = {self.nut_locations[nut] for nut in loose_goal_nuts_in_state if nut in self.nut_locations}

        # 8. Identify locations of available usable spanners (not carried)
        available_usable_spanners_at_loc = {
            spanner_at_location_facts[s] for s in usable_spanners
            if s not in carried_spanners and s in spanner_at_location_facts
        }

        # 9. Determine required locations for travel
        required_locations = set(loose_nut_locations)
        if needed_spanners_to_pickup > 0:
             # We need to visit locations with available usable spanners
             required_locations.update(available_usable_spanners_at_loc)

        # 10. Calculate minimum distance to a required location
        min_dist_to_required_loc = float('inf')
        if required_locations:
            if man_loc in self.distances:
                min_dist_to_required_loc = min(
                    self.distances[man_loc].get(loc, float('inf')) for loc in required_locations
                )
            else:
                 # Man's location is not in the precomputed distances - invalid state
                 return float('inf')

        # If min_dist_to_required_loc is still inf, it means required_locations was not empty,
        # but man_loc cannot reach any of them. This state is a dead end.
        if min_dist_to_required_loc == float('inf'):
             return float('inf')


        # 11. Calculate heuristic value
        h_value = num_loose_goal_nuts + needed_spanners_to_pickup + min_dist_to_required_loc

        return h_value
