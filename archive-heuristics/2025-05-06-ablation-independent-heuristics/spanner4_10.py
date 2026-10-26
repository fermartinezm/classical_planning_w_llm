from heuristics.heuristic_base import Heuristic
from task import Operator, Task
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove surrounding parentheses and split by spaces
    parts = fact_string.strip('()').split()
    return tuple(parts)

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal by summing:
    1. The number of nuts that still need to be tightened.
    2. The number of usable spanners the man still needs to pick up.
    3. The minimum walking distance from the man's current location to any
       location that contains a nut needing tightening or a usable spanner
       that needs to be picked up.

    Assumptions:
    - The domain is 'spanner' as defined.
    - There is exactly one man object.
    - Nut locations are fixed and can be determined from the initial state.
    - Spanners do not become usable again after being used for tightening.
    - Links between locations are bidirectional.
    - All locations mentioned in facts are part of the location graph.
    - The task object provides access to initial_state, goals, facts, and static.

    Heuristic Initialization:
    - Identifies the man object, spanner objects, and nut objects by parsing task.facts.
    - Stores the goal nuts by parsing task.goals.
    - Stores the fixed locations of goal nuts by parsing task.initial_state.
    - Builds a graph of locations based on 'link' facts from task.static.
    - Computes all-pairs shortest paths between locations using BFS and stores them.

    Step-By-Step Thinking for Computing Heuristic (__call__):
    1. Get the current state.
    2. Find the man's current location in the state. If not found, return infinity.
    3. Identify usable spanners the man is currently carrying.
    4. Identify usable spanners currently located at various places.
    5. Determine the set of goal nuts that are not yet tightened in the current state ('nuts_to_tighten').
    6. If 'nuts_to_tighten' is empty, the goal is reached, return 0.
    7. Calculate the number of additional usable spanners the man needs to pick up ('needed_pickups'). This is the number of nuts to tighten minus the number of usable spanners currently carried, capped at zero.
    8. Check if the total number of usable spanners available in the state (carried + at locations) is less than the number of nuts to tighten. If so, the goal is unreachable, return infinity.
    9. Initialize the heuristic value with the number of nuts to tighten (base cost for tighten actions).
    10. Add the number of needed pickups (base cost for pickup actions).
    11. Determine the set of target locations the man needs to reach:
        - All locations of nuts in 'nuts_to_tighten'.
        - If 'needed_pickups' > 0, find the 'needed_pickups' closest locations containing usable spanners from the man's current location. Add these to the target locations.
    12. Calculate the minimum walking distance from the man's current location to any location in the set of target locations. If no targets exist (should not happen if k>0), return infinity.
    13. Add this minimum walking distance to the heuristic value.
    14. Return the calculated heuristic value.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task

        # --- Initialization: Parse static info and precompute distances ---

        # Identify object types and collect all mentioned locations
        self.man_object = None
        self.spanner_objects = set()
        self.nut_objects = set()
        self.location_objects = set()
        self.nut_locations = {} # Map nut -> fixed location

        # Parse all possible facts to identify typed objects
        for fact_name in task.facts:
            if fact_name.startswith('('):
                parts = parse_fact(fact_name)
                if len(parts) == 2: # e.g., (type object)
                    obj_type, obj_name = parts
                    if obj_type == 'man':
                        self.man_object = obj_name
                    elif obj_type == 'spanner':
                        self.spanner_objects.add(obj_name)
                    elif obj_type == 'nut':
                        self.nut_objects.add(obj_name)
                    elif obj_type == 'location':
                        self.location_objects.add(obj_name)

        if not self.man_object:
             logging.warning("Could not identify the man object from task.facts.")
             # This is a critical issue, heuristic might not work.

        # Get fixed nut locations from initial state
        for fact in task.initial_state:
             if fact.startswith('(at '):
                 parts = parse_fact(fact)
                 if len(parts) == 3:
                     obj, loc = parts[1], parts[2]
                     if obj in self.nut_objects:
                         self.nut_locations[obj] = loc
                     # Add all locations mentioned in initial 'at' facts
                     self.location_objects.add(loc)

        # Get goal nuts
        self.goal_nuts = set()
        for goal in task.goals:
            if goal.startswith('(tightened '):
                parts = parse_fact(goal)
                if len(parts) == 2:
                    self.goal_nuts.add(parts[1])

        # Build location graph from static links and add locations from links
        self.location_graph = {}
        for fact in task.static:
            if fact.startswith('(link '):
                parts = parse_fact(fact)
                if len(parts) == 3:
                    loc1, loc2 = parts[1], parts[2]
                    self.location_objects.add(loc1)
                    self.location_objects.add(loc2)
                    if loc1 not in self.location_graph:
                        self.location_graph[loc1] = set()
                    if loc2 not in self.location_graph:
                        self.location_graph[loc2] = set()
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1) # Links are bidirectional

        # Ensure all known locations are in the graph structure even if they have no links
        for loc in self.location_objects:
             if loc not in self.location_graph:
                  self.location_graph[loc] = set()


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.location_objects:
            self.distances[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            while queue:
                (current_loc, dist) = queue.popleft()
                self.distances[start_loc][current_loc] = dist
                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns infinity if unreachable."""
        return self.distances.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the current state.
        """
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # 1. Find man's current location
        man_loc = None
        if self.man_object:
            man_at_fact_prefix = f'(at {self.man_object} '
            for fact in state:
                if fact.startswith(man_at_fact_prefix):
                    man_loc = parse_fact(fact)[2]
                    break

        if man_loc is None:
            # Man's location is unknown or man object not found (should not happen in valid state)
            logging.warning(f"Man object '{self.man_object}' location not found in state.")
            return float('inf') # Cannot proceed if man location is unknown

        # 2. Identify usable spanners currently carried by the man
        usable_spanners_carried = set()
        man_carrying_prefix = f'(carrying {self.man_object} '
        for fact in state:
            if fact.startswith(man_carrying_prefix):
                spanner = parse_fact(fact)[2]
                if f'(usable {spanner})' in state:
                     usable_spanners_carried.add(spanner)

        # 3. Identify usable spanners at locations
        usable_spanners_at_loc = {} # Map spanner -> location
        for fact in state:
            if fact.startswith('(at '):
                parts = parse_fact(fact)
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    if obj in self.spanner_objects: # Check if it's a spanner
                         if f'(usable {obj})' in state:
                             usable_spanners_at_loc[obj] = loc

        # 4. Identify loose nuts that are goals
        nuts_to_tighten = set()
        nut_locations_to_visit = set()
        for nut in self.goal_nuts:
            if f'(tightened {nut})' not in state:
                # This nut needs tightening
                nuts_to_tighten.add(nut)
                # Get its location (precomputed in __init__)
                if nut in self.nut_locations:
                    nut_locations_to_visit.add(self.nut_locations[nut])
                else:
                    # Nut location not found (should not happen if initial state is complete)
                    logging.warning(f"Location for goal nut '{nut}' not found.")
                    return float('inf')

        k = len(nuts_to_tighten)

        # If k == 0, goal is reached (already checked at the beginning)

        # 5. Calculate number of spanners needed to pick up
        needed_pickups = max(0, k - len(usable_spanners_carried))

        # 6. Check if enough usable spanners exist in total
        total_usable_spanners = len(usable_spanners_carried) + len(usable_spanners_at_loc)
        if total_usable_spanners < k:
            # Not enough usable spanners in the entire problem to tighten all nuts.
            return float('inf')

        # --- Estimate Heuristic Value ---

        # Base cost: 1 for each tighten_nut action
        h_value = k

        # Cost for picking up spanners: 1 for each pickup action needed
        h_value += needed_pickups

        # Walking cost estimate:
        # The man needs to reach all locations in nut_locations_to_visit.
        # The man needs to reach needed_pickups locations from usable_spanners_at_loc.

        target_locations = set(nut_locations_to_visit)

        if needed_pickups > 0:
            # Find the needed_pickups closest usable spanner locations from man_loc
            available_spanner_locations = list(usable_spanners_at_loc.values())
            # Calculate distances to these locations
            distances_to_spanners = []
            for loc in available_spanner_locations:
                 dist = self.get_distance(man_loc, loc)
                 if dist == float('inf'):
                      # Cannot reach this spanner location
                      continue # Skip unreachable spanners
                 distances_to_spanners.append((dist, loc))

            # Sort by distance and take the needed_pickups closest reachable locations
            distances_to_spanners.sort()

            if len(distances_to_spanners) < needed_pickups:
                 # Not enough *reachable* usable spanners
                 return float('inf')

            # Add the locations of the needed_pickups closest spanners to targets
            for i in range(needed_pickups):
                 target_locations.add(distances_to_spanners[i][1])


        # Calculate the minimum walking distance from man_loc to any target location
        min_dist_to_target = float('inf')
        if target_locations:
            for loc in target_locations:
                dist = self.get_distance(man_loc, loc)
                min_dist_to_target = min(min_dist_to_target, dist)

            if min_dist_to_target == float('inf'):
                 # Cannot reach any required target location
                 return float('inf')

            h_value += min_dist_to_target
        # else: target_locations is empty, should only happen if k=0, handled already.

        return h_value
