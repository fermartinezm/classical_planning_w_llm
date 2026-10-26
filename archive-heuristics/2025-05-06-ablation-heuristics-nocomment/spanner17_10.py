from fnmatch import fnmatch
from collections import deque
import math # For float('inf')

# Assume Heuristic base class is imported from the planner's base class
# from heuristics.heuristic_base import Heuristic
# If running standalone or without the planner structure, you might need a dummy
# Heuristic class definition here, like:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): pass


def get_parts(fact):
    """Helper to parse a PDDL fact string into parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to match a fact against a pattern."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args,
    # and if each part matches the corresponding arg pattern.
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal (tightening all specified nuts)
    by summing the estimated costs for three main components:
    1. Tightening the remaining loose goal nuts.
    2. Picking up usable spanners if the man isn't carrying enough.
    3. Moving the man to visit the locations of the loose goal nuts
       and the locations where spanners need to be picked up.
    Movement cost is estimated using a greedy approach similar to a
    Traveling Salesperson Problem starting from the man's current location,
    visiting the necessary nut and spanner pickup locations.

    Assumptions:
    - The problem is solvable (enough usable spanners exist in total and locations are reachable).
    - Links between locations are directed as specified in the PDDL.
    - The state representation includes facts like '(at obj loc)',
      '(carrying man spanner)', '(usable spanner)', '(loose nut)',
      '(tightened nut)'.
    - The task object provides 'goals' (frozenset of goal facts) and
      'static' (frozenset of static facts like links).
    - The name of the man object can be inferred from the state (e.g., by looking for the first argument in 'carrying' or 'at' facts that isn't a spanner or nut).

    Heuristic Initialization:
    - Parses static 'link' facts to build a directed graph of locations.
    - Identifies all locations mentioned in static facts, initial state, and goals.
    - Computes all-pairs shortest paths between all identified locations using BFS
      on the directed graph. Stores these distances. Unreachable locations have infinite distance.
    - Identifies the set of goal nuts from the task goals.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the man's current location and the man object's name from the state. If the man or his location cannot be determined, return infinity.
    2. Identify all nuts that are in the goal state but are currently loose. These are the 'loose goal nuts'.
    3. If there are no loose goal nuts, the heuristic is 0 (goal state).
    4. Initialize the heuristic value `h` with the number of loose goal nuts (representing the `tighten_nut` action cost for each).
    5. Identify the locations of all loose goal nuts. If any loose goal nut's location is unknown, return infinity.
    6. Identify usable spanners the man is currently carrying.
    7. Identify usable spanners at locations and their locations.
    8. Calculate how many additional usable spanners the man needs to pick up from locations to tighten all loose goal nuts (`needed_from_locs`). This is the maximum of 0 and (number of loose goal nuts - number of usable spanners carried).
    9. Add `needed_from_locs` to `h` (representing the `pickup_spanner` action cost for each).
    10. Identify the locations from which these additional spanners will be picked up. Select the `needed_from_locs` usable spanners at locations that are *reachable* from the man's current location and have the minimum shortest path distance from the man's current location. If there aren't enough reachable usable spanners available at locations, the problem is likely unsolvable, return infinity.
    11. Define the set of 'target locations' that the man must visit. This set includes the locations of all loose goal nuts and the selected spanner pickup locations.
    12. Calculate the movement cost using a greedy approach:
        - Start at the man's current location.
        - Maintain a set of unvisited target locations.
        - If the man is already at a target location, remove it from the unvisited set.
        - While there are unvisited target locations:
            - Find the unvisited target location that is nearest (using precomputed shortest path distances) to the current location.
            - If no reachable unvisited target is found (minimum distance is infinity), return infinity (unsolvable).
            - Add the distance to this nearest location to `h`.
            - Move to this nearest location (update current location).
            - Mark the location as visited by removing it from the unvisited set.
    13. Return the final heuristic value `h`.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # 1. Build the directed graph from link facts and collect all locations
        self.graph = {}
        all_locations_in_problem = set()

        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                all_locations_in_problem.add(loc1)
                all_locations_in_problem.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                self.graph[loc1].append(loc2)

        # Add locations mentioned in initial state and goals
        # Assuming task.initial_state is available in the Task object
        if hasattr(task, 'initial_state'):
             for fact in task.initial_state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and len(parts) == 3:
                       all_locations_in_problem.add(parts[2])

        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'at' and len(parts) == 3:
                  all_locations_in_problem.add(parts[2])

        self.locations = list(all_locations_in_problem)

        # Ensure all locations are keys in graph, even if they have no outgoing links
        for loc in self.locations:
             if loc not in self.graph:
                 self.graph[loc] = []

        # 2. Compute all-pairs shortest paths using BFS on the directed graph
        self.distances = {}
        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_node, dist = q.popleft()

                if current_node in self.graph:
                    for neighbor in self.graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_node, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

        # For any pair not found by BFS, the distance is infinity
        for l1 in self.locations:
            for l2 in self.locations:
                if (l1, l2) not in self.distances:
                    self.distances[(l1, l2)] = float('inf')

        # 3. Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Identify object types if possible (fragile inference)
        # This is needed to reliably find the man object name in __call__
        self.spanner_names = set()
        self.nut_names = set()
        # Try to infer from initial state if available
        if hasattr(task, 'initial_state'):
             for fact in task.initial_state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and len(parts) == 3:
                       obj = parts[1]
                       # Infer type based on name pattern (fragile)
                       if obj.startswith('spanner'): self.spanner_names.add(obj)
                       elif obj.startswith('nut'): self.nut_names.add(obj)

        # Fallback: Infer goal nuts (already done)
        self.nut_names.update(self.goal_nuts)


    def get_distance(self, loc1, loc2):
         """Helper to get precomputed distance, handling unknown locations."""
         # Ensure locations are in our known set before lookup
         if loc1 not in self.locations or loc2 not in self.locations:
             # This indicates an object is at a location not seen in init/static/goals
             # Treat as unreachable for heuristic purposes
             return float('inf')
         return self.distances[(loc1, loc2)]


    def __call__(self, node):
        state = node.state

        # 1. Identify man's current location and name
        man_obj = None
        man_location = None

        # Refined man object inference: Find the single object that is carried
        # or is at a location and is not a known spanner or nut.
        potential_men = set()
        current_spanners = set()
        current_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if loc in self.locations: # Only consider facts with known locations
                    potential_men.add(obj)
                    # Update known spanners/nuts from state
                    if obj.startswith('spanner'): current_spanners.add(obj)
                    elif obj.startswith('nut'): current_nuts.add(obj)
            elif parts[0] == 'carrying' and len(parts) == 3:
                m, s = parts[1], parts[2]
                potential_men.add(m) # Object being carried is very likely the man
                current_spanners.add(s) # Object being carried is a spanner

        # The man is the object in potential_men that is not a spanner or nut
        # Use inferred names from init + names found in current state
        all_spanners = self.spanner_names | current_spanners
        all_nuts = self.nut_names | current_nuts
        man_candidates = potential_men - all_spanners - all_nuts

        if len(man_candidates) == 1:
            man_obj = list(man_candidates)[0]
        elif len(man_candidates) > 1:
             # Multiple candidates, or none? Fallback or error.
             # Assume the first object in an 'at' fact that isn't a spanner/nut is the man.
             man_obj = None
             for fact in state:
                  parts = get_parts(fact)
                  if parts[0] == 'at' and len(parts) == 3:
                       obj = parts[1]
                       if obj not in all_spanners and obj not in all_nuts:
                            man_obj = obj
                            break # Found a likely man object name

        if man_obj is None:
             # Could not identify the man object name reliably.
             # This state might be malformed or requires better object type info.
             return float('inf') # Indicate unsolvability or invalid state

        # Find the location of the identified man object
        for fact in state:
            if match(fact, "at", man_obj, "*"):
                man_location = get_parts(fact)[2]
                break

        if man_location is None or man_location not in self.locations:
             # Man object found, but not at a known location? Invalid state.
             return float('inf')


        # 2. Identify loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if f'(loose {n})' in state}

        # 3. If no loose goal nuts, goal is reached for these nuts
        if not loose_goal_nuts:
            return 0

        # 4. Initialize heuristic
        h = len(loose_goal_nuts) # Cost for tighten_nut actions

        # 5. Identify locations of loose goal nuts
        nut_locations = {} # {nut_name: location_name}
        for nut in loose_goal_nuts:
            found_loc = False
            for fact in state:
                if match(fact, "at", nut, "*"):
                    loc = get_parts(fact)[2]
                    if loc in self.locations:
                        nut_locations[nut] = loc
                        found_loc = True
                        break
                    # else: nut is at an unknown location, treat as unreachable?
            if not found_loc or nut_locations[nut] not in self.locations:
                 # Loose goal nut exists but isn't at a known location? Invalid state or unreachable nut.
                 return float('inf')

        # 6. Identify usable spanners carried by the man
        usable_spanners_carried = {get_parts(fact)[2] for fact in state if match(fact, "carrying", man_obj, "*") and f'(usable {get_parts(fact)[2]})' in state}

        # 7. Identify usable spanners at locations
        usable_spanners_at_locs = {} # {spanner_name: location_name}
        for fact in state:
            # Check if it's an 'at' fact for an object that looks like a spanner
            if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 obj_name = parts[1]
                 loc = parts[2]
                 if obj_name.startswith('spanner') and loc in self.locations: # Basic inference + known location
                     if f'(usable {obj_name})' in state:
                         usable_spanners_at_locs[obj_name] = loc

        # 8. Calculate needed spanners from locations
        needed_from_locs = max(0, len(loose_goal_nuts) - len(usable_spanners_carried))

        # 9. Add pickup cost
        h += needed_from_locs

        # 10. Identify spanner pickup locations
        spanner_pickup_locations = set() # Set of locations to visit for pickup
        if needed_from_locs > 0:
            # Sort usable spanners at locations by distance from man's current location
            available_spanners_list = list(usable_spanners_at_locs.items()) # [(spanner, loc), ...]
            # Filter out spanners at locations unreachable from the man
            reachable_available_spanners = [(s, loc) for s, loc in available_spanners_list if self.get_distance(man_location, loc) != float('inf')]

            if len(reachable_available_spanners) < needed_from_locs:
                 # Not enough reachable usable spanners to meet the need
                 return float('inf')

            # Sort the reachable ones by distance
            reachable_available_spanners.sort(key=lambda item: self.get_distance(man_location, item[1]))

            # Take the locations of the nearest 'needed_from_locs' spanners
            for i in range(needed_from_locs):
                 spanner_pickup_locations.add(reachable_available_spanners[i][1])


        # 11. Define target locations
        target_locations = set(nut_locations.values()) | spanner_pickup_locations

        # 12. Calculate movement cost using greedy TSP
        current_loc = man_location
        unvisited_targets = set(target_locations)

        # If man is already at a target location, remove it from unvisited
        if current_loc in unvisited_targets:
             unvisited_targets.remove(current_loc)

        while unvisited_targets:
            # Find nearest unvisited target
            nearest_loc = None
            min_dist = float('inf')

            for target_loc in unvisited_targets:
                dist = self.get_distance(current_loc, target_loc)
                if dist < min_dist:
                    min_dist = dist
                    nearest_loc = target_loc

            if nearest_loc is None or min_dist == float('inf'):
                # Cannot reach any remaining target location from the current location
                return float('inf') # Indicate unsolvability

            h += min_dist
            current_loc = nearest_loc
            unvisited_targets.remove(nearest_loc)

        # 13. Return heuristic value
        return h
