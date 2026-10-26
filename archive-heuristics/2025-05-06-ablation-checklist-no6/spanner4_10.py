from fnmatch import fnmatch
from collections import deque
import math # For float('inf')

# Assuming Heuristic base class is available as in examples
# from heuristics.heuristic_base import Heuristic

# Mock Heuristic base class if not provided externally
# This mock class is included only for standalone testing purposes
# and should be removed if the actual base class is available.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state to find objects
        pass # Placeholder

    def __call__(self, node):
        pass # Placeholder


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all loose goal nuts.
    The heuristic sums the number of tighten actions, the number of spanner
    pickups needed, and the minimum distance from the man to any required
    location (loose nuts or usable spanners).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task) # Call base class constructor

        # Extract static info in constructor
        self.location_graph = self.build_location_graph(self.static)

        # Identify all objects and their types (heuristically based on naming convention and predicates)
        self.all_objects = set()
        self.all_locations = set()
        self.all_spanners = set()
        self.all_nuts = set()
        self.man_name = None

        # Collect objects and locations from initial state and static facts
        all_locatable_objects_in_init = set()
        for fact in self.initial_state:
             parts = get_parts(fact)
             if len(parts) > 1:
                 self.all_objects.add(parts[1])
             if len(parts) > 2:
                 self.all_objects.add(parts[2])

             if parts[0] == 'at' and len(parts) == 3:
                  obj, loc = parts[1], parts[2]
                  all_locatable_objects_in_init.add(obj)
                  self.all_locations.add(loc)
             elif parts[0] == 'carrying' and len(parts) == 3:
                  man, spanner = parts[1], parts[2]
                  all_locatable_objects_in_init.add(man)
                  all_locatable_objects_in_init.add(spanner) # Spanners are locatable
                  self.all_spanners.add(spanner)
             elif parts[0] == 'usable' and len(parts) == 2:
                  spanner = parts[1]
                  self.all_spanners.add(spanner)
             elif parts[0] == 'loose' and len(parts) == 2:
                  nut = parts[1]
                  self.all_nuts.add(nut)
             elif parts[0] == 'tightened' and len(parts) == 2:
                  nut = parts[1]
                  self.all_nuts.add(nut)

        # Add locations from link facts
        for fact in self.static:
             parts = get_parts(fact)
             if parts[0] == 'link' and len(parts) == 3:
                  self.all_locations.add(parts[1])
                  self.all_locations.add(parts[2])
             elif len(parts) == 2 and parts[0] == 'location': # Explicit location facts
                  self.all_locations.add(parts[1])


        # Infer man object: the locatable object that is not a spanner or nut
        potential_men = all_locatable_objects_in_init - self.all_spanners - self.all_nuts
        if len(potential_men) == 1:
             self.man_name = list(potential_men)[0]
        elif len(potential_men) > 1:
             # Handle multiple men? Domain description implies one man.
             # Assume the first object in initial state that is 'at' a location and isn't a known nut/spanner is the man.
             man_found = False
             for fact in self.initial_state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and len(parts) == 3:
                       obj = parts[1]
                       if obj not in self.all_nuts and obj not in self.all_spanners:
                            self.man_name = obj
                            man_found = True
                            break
             if not man_found:
                  # Fallback: Assume 'bob' if no other man found (based on example)
                  self.man_name = 'bob' # Fragile fallback


        # Identify all goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])


    def build_location_graph(self, static_facts):
        """Builds an adjacency list for locations based on link facts."""
        graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                graph.setdefault(loc1, set()).add(loc2)
                graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional
        return graph

    def bfs_distances(self, graph, start_location, all_locations):
        """Calculates shortest path distances from start_location to all reachable locations."""
        distances = {loc: math.inf for loc in all_locations}
        if start_location in all_locations:
             distances[start_location] = 0
             queue = deque([start_location])
             # visited = {start_location} # Not needed if using distances == inf check

             while queue:
                 current_loc = queue.popleft()
                 if current_loc in graph: # Handle locations that might not be linked to anything
                     for neighbor in graph[current_loc]:
                         if distances[neighbor] == math.inf: # Check if visited using distance value
                             distances[neighbor] = distances[current_loc] + 1
                             queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Estimate the minimum cost to tighten all loose goal nuts."""
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if f'(loose {n})' in state}

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # 2. Find man's current location
        man_location = None
        if self.man_name:
             for fact in state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and len(parts) == 3 and parts[1] == self.man_name:
                       man_location = parts[2]
                       break

        if man_location is None:
             # Man's location not found in state (should not happen in valid states)
             return math.inf # Problem unsolvable from this state


        # 3. Calculate distances from man's current location
        distances = self.bfs_distances(self.location_graph, man_location, self.all_locations)

        # 4. Identify locations of loose goal nuts
        nut_locations = {}
        for nut in loose_goal_nuts:
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1] == nut and len(parts) == 3:
                    nut_locations[nut] = parts[2]
                    break

        # If any loose goal nut location is unreachable, return infinity
        if any(loc not in distances or distances[loc] == math.inf for loc in nut_locations.values()):
             return math.inf


        # 5. Check if man is carrying a usable spanner
        man_carrying_usable_spanner = False
        carried_spanner = None
        if self.man_name:
             for fact in state:
                  parts = get_parts(fact)
                  if parts[0] == 'carrying' and len(parts) == 3 and parts[1] == self.man_name:
                       carried_spanner = parts[2]
                       if f'(usable {carried_spanner})' in state:
                            man_carrying_usable_spanner = True
                       break # Assuming man carries at most one spanner


        # 6. Calculate number of spanner pickups needed
        num_tighten_needed = len(loose_goal_nuts)
        num_pickups_needed = max(0, num_tighten_needed - (1 if man_carrying_usable_spanner else 0))


        # 7. Identify usable spanners on the ground and their locations
        usable_spanners_on_ground = []
        for spanner in self.all_spanners:
             if f'(usable {spanner})' in state:
                  # Check if it's on the ground (not carried)
                  is_carried = False
                  if carried_spanner == spanner: # Check if the one man carries is this spanner
                       is_carried = True
                  if not is_carried:
                       for fact in state:
                            parts = get_parts(fact)
                            if parts[0] == 'at' and parts[1] == spanner and len(parts) == 3:
                                 usable_spanners_on_ground.append((spanner, parts[2]))
                                 break # Found location for this spanner

        # If pickups are needed but no usable spanners are available (neither carried usable nor on ground usable)
        if num_pickups_needed > 0 and not man_carrying_usable_spanner and not usable_spanners_on_ground:
             return math.inf # Problem unsolvable


        # 8. Identify all locations that need to be reached for the first "task"
        # These are the locations of loose goal nuts and, if pickups are needed, usable spanners on the ground.
        required_locations = set(nut_locations.values())
        if num_pickups_needed > 0:
             usable_spanner_ground_locations = {loc for s, loc in usable_spanners_on_ground}
             required_locations.update(usable_spanner_ground_locations)

        # If there are required locations but none are reachable, return infinity
        min_dist_to_required = math.inf
        if required_locations:
             min_dist_to_required = min((distances[loc] for loc in required_locations if loc in distances and distances[loc] != math.inf), default=math.inf)

        if min_dist_to_required == math.inf and required_locations:
             return math.inf # Required locations exist but are unreachable


        # 9. Calculate heuristic value
        # h = num_tighten + num_pickups + min_dist_to_required

        heuristic_value = num_tighten_needed + num_pickups_needed + min_dist_to_required

        # Ensure heuristic is 0 only at goal (checked at the start) and finite for solvable states (handled inf propagation)
        # Ensure heuristic is non-negative (all components are non-negative)

        return heuristic_value

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues
    return fact.strip()[1:-1].split()
