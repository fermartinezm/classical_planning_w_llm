from collections import deque
from fnmatch import fnmatch
# Assume Heuristic base class is available in this environment
# from heuristics.heuristic_base import Heuristic

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

# BFS function
def bfs(graph, start):
    """Computes shortest path distances from start node in an unweighted graph."""
    distances = {node: float('inf') for node in graph}
    if start not in graph:
         # This case should be handled by adding all relevant locations to graph keys in __init__
         return distances # Should not happen if __init__ is correct

    distances[start] = 0
    queue = deque([start])
    while queue:
        curr = queue.popleft()
        # Check if curr is a key in graph before iterating (redundant if __init__ is correct, but safe)
        if curr in graph:
            for neighbor in graph[curr]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[curr] + 1
                    queue.append(neighbor)
    return distances


# Define the heuristic class inheriting from Heuristic
# If Heuristic base class is not provided in the execution environment,
# remove the inheritance and just define the class.
# Assuming it is provided based on problem description and examples.
class spannerHeuristic: # Inherit from Heuristic if available, e.g., class spannerHeuristic(Heuristic):
    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and computing
        shortest path distances.
        """
        self.task = task # Store task for access to goals later

        # Build location graph from static link facts
        self.location_graph = {}
        all_locations_in_links = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Undirected
                all_locations_in_links.add(loc1)
                all_locations_in_links.add(loc2)

        # Collect all locations mentioned in initial state and goals
        # These are potential start/end points for movement
        all_relevant_locations = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                  all_relevant_locations.add(parts[2])
        for goal in task.goals:
             parts = get_parts(goal)
             if parts[0] == 'at':
                  all_relevant_locations.add(parts[2])
        # Add locations from links even if not in init/goal (might be intermediate)
        all_relevant_locations.update(all_locations_in_links)


        # Ensure all relevant locations are keys in the graph dictionary
        # even if they have no links, so BFS can compute distance 0 to self.
        for loc in all_relevant_locations:
            self.location_graph.setdefault(loc, [])


        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.location_graph.keys(): # Iterate over all known locations
             self.dist[start_loc] = bfs(self.location_graph, start_loc)

        # Store goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in task.goals if get_parts(goal)[0] == 'tightened'}


    def __call__(self, node):
        """
        Estimate the minimum number of actions to reach the goal state.
        The heuristic sums the estimated cost for each loose goal nut,
        greedily assigning available spanners and updating the man's location.
        """
        state = node.state

        # 1. Identify loose goal nuts and their locations
        loose_goal_nuts = set()
        nut_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'loose' and parts[1] in self.goal_nuts:
                loose_goal_nuts.add(parts[1])

        if not loose_goal_nuts:
            return 0 # Goal reached

        # Find locations for loose goal nuts
        for nut in loose_goal_nuts:
             for fact in state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and parts[1] == nut:
                       nut_locations[nut] = parts[2]
                       break # Found location for this nut
             # If a loose goal nut has no location in the state, it's unreachable/problematic
             if nut not in nut_locations:
                  return float('inf')


        # 2. Identify man's location and carried spanner
        man_loc = None
        man_name = None
        carried_spanner = None
        usable_carried = False

        # Find man name (assume it's the object in 'at' fact that isn't a spanner or nut)
        spanner_objects_in_state = {parts[1] for fact in state if get_parts(fact)[0] == 'at' and parts[1].startswith('spanner')}
        nut_objects_in_state = {parts[1] for fact in state if get_parts(fact)[0] == 'at' and parts[1].startswith('nut')}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj not in spanner_objects_in_state and obj not in nut_objects_in_state:
                    man_name = obj
                    man_loc = loc
                    break # Found the man

        if man_loc is None:
             # Man location not found, problem state is invalid or unreachable
             return float('inf')


        # Find carried spanner and if it's usable
        if man_name:
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'carrying' and parts[1] == man_name:
                    carried_spanner = parts[2]
                    break
            if carried_spanner and (f'(usable {carried_spanner})' in state):
                 usable_carried = True


        # 3. Identify usable spanners at locations
        usable_spanners_at_loc = set()
        spanner_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'usable' and parts[1] != carried_spanner:
                 usable_spanners_at_loc.add(parts[1])

        for spanner in usable_spanners_at_loc:
             for fact in state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and parts[1] == spanner:
                       spanner_locations[spanner] = parts[2]
                       break # Found location for this spanner
             # If a usable spanner at loc has no location in the state, it's problematic
             if spanner not in spanner_locations:
                  return float('inf')


        # 4. Check solvability (enough usable spanners)
        N_loose_goal = len(loose_goal_nuts)
        N_usable_available = len(usable_spanners_at_loc) + (1 if usable_carried else 0)

        if N_usable_available < N_loose_goal:
            return float('inf') # Unsolvable


        # 5. Calculate heuristic cost
        h = 0
        current_man_loc = man_loc

        # Sort nuts by distance from man's initial location for greedy processing order
        # Using initial man_loc for sorting provides a consistent order.
        nuts_to_process = sorted(list(loose_goal_nuts),
                                 key=lambda n: self.dist[man_loc].get(nut_locations[n], float('inf')))

        # Create a list of available spanners, prioritizing the carried one
        available_spanners_info = [] # List of (spanner_name, location or 'carried')
        if usable_carried:
            available_spanners_info.append((carried_spanner, 'carried'))

        # Add spanners at locations, sorted by distance from initial man location
        spanners_at_loc_list = [(s, spanner_locations[s]) for s in usable_spanners_at_loc]
        # Sort spanners at location by distance from initial man_loc
        spanners_at_loc_list.sort(key=lambda item: self.dist[man_loc].get(item[1], float('inf')))

        available_spanners_info.extend(spanners_at_loc_list)


        for nut in nuts_to_process:
            h += 1 # Cost for tighten_nut action

            nut_loc = nut_locations[nut]

            # Should not happen due to unsolvable check, but defensive
            if not available_spanners_info:
                 return float('inf')

            spanner_info = available_spanners_info.pop(0) # Get the next available spanner
            spanner_to_use, spanner_origin = spanner_info

            if spanner_origin == 'carried':
                # Use the carried spanner (which was the initial carried usable one)
                # Walk from current man location to nut location
                dist_walk = self.dist[current_man_loc].get(nut_loc, float('inf'))
                if dist_walk == float('inf'): return float('inf') # Cannot reach nut
                h += dist_walk
                current_man_loc = nut_loc # Man is now at the nut location
            else: # spanner_origin is a location
                spanner_loc = spanner_origin

                # Walk from current man location to spanner location
                dist_walk_to_spanner = self.dist[current_man_loc].get(spanner_loc, float('inf'))
                if dist_walk_to_spanner == float('inf'): return float('inf') # Cannot reach spanner
                h += dist_walk_to_spanner

                h += 1 # pickup action cost

                # Walk from spanner location to nut location
                dist_walk_to_nut = self.dist[spanner_loc].get(nut_loc, float('inf'))
                if dist_walk_to_nut == float('inf'): return float('inf') # Cannot reach nut from spanner loc
                h += dist_walk_to_nut

                current_man_loc = nut_loc # Man is now at the nut location

        return h
