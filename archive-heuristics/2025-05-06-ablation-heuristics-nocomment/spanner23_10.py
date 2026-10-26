# from heuristics.heuristic_base import Heuristic # Uncomment in actual environment
from fnmatch import fnmatch
from collections import deque
import math # For infinity

# Helper functions (can be defined inside or outside the class)
def get_parts(fact):
    """Removes parentheses and splits a PDDL fact string into parts."""
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if the parts of a fact match the given arguments (with fnmatch support)."""
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the Heuristic base class structure if not provided externally
# This is just for self-containment during development/testing.
# In the actual environment, it would be imported.
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#         pass # Placeholder
#
#     def __call__(self, node):
#         pass # Placeholder


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        Estimates the cost to reach the goal state (all goal nuts tightened)
        by summing up:
        1. The number of loose goal nuts (representing tighten actions).
        2. The number of spanners that need to be picked up from locations
           (representing pickup actions).
        3. The minimum distance from the man's current location to any
           location containing a loose goal nut (representing initial movement
           towards a task).
        4. The minimum distance from the man's current location to any
           location containing an available usable spanner, if spanners
           need to be picked up (representing initial movement towards a resource).

    Assumptions:
        - There is exactly one man object.
        - Nuts are static (do not change location).
        - Spanners are consumed (become not usable) after one tighten action.
        - The man can carry multiple spanners simultaneously.
        - The location graph defined by 'link' predicates is undirected.
        - The problem is solvable (enough usable spanners exist initially or can be acquired).
        - Object types (man, spanner, nut, location) can be inferred from
          predicate usage in the initial state and static facts.

    Heuristic Initialization:
        - Parses static facts to build the location graph based on 'link' predicates.
        - Computes all-pairs shortest paths between locations using BFS and stores them.
        - Identifies the man object, nut objects, spanner objects, and location objects
          by examining initial state and static facts.
        - Stores the fixed locations of nuts.
        - Identifies the set of goal nuts from the task's goal state.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state from the search node.
        2. Identify the man's current location by finding the '(at man_obj loc)' fact.
        3. Identify spanners currently carried by the man by finding '(carrying man_obj spanner)' facts.
        4. Identify spanners that are currently usable by finding '(usable spanner)' facts.
        5. Identify nuts that are currently loose by finding '(loose nut)' facts.
        6. Determine the set of goal nuts that are currently loose. If this set is empty, the goal is reached, return 0.
        7. Initialize heuristic value `h` to 0.
        8. Add the number of loose goal nuts to `h`. This accounts for the 'tighten_nut' action needed for each.
        9. Determine the number of usable spanners the man is currently carrying.
        10. Calculate how many additional usable spanners are needed from locations: `num_spanners_needed_from_locs = max(0, num_loose_goal_nuts - num_carried_usable_spanners)`.
        11. Add `num_spanners_needed_from_locs` to `h`. This accounts for the 'pickup_spanner' actions needed.
        12. Find the locations of all loose goal nuts.
        13. Calculate the minimum distance from the man's current location to any of the loose nut locations. Add this distance to `h`. This accounts for the travel to the first nut location.
        14. If `num_spanners_needed_from_locs > 0`:
            a. Find all usable spanners that are currently at locations (not carried).
            b. If the number of such spanners is less than `num_spanners_needed_from_locs`, the state is likely unsolvable; return infinity.
            c. Find the locations of these available usable spanners.
            d. Calculate the minimum distance from the man's current location to any of these available spanner locations. Add this distance to `h`. This accounts for the travel to the first spanner location needed for pickup.
        15. Return the calculated value of `h`.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # --- Parse static information ---
        self.locations = set()
        self.link_graph = {} # Adjacency list for locations
        self.nut_locations = {} # Map nut -> initial location
        self.spanner_objects = set()
        self.nut_objects = set()
        self.man_object = None

        # Collect objects and locations from initial state and static facts
        all_objects_mentioned = set()
        all_locations_mentioned = set()

        for fact in self.initial_state | self.static:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'link':
                loc1, loc2 = args
                self.link_graph.setdefault(loc1, set()).add(loc2)
                self.link_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional
                all_locations_mentioned.add(loc1)
                all_locations_mentioned.add(loc2)
            elif predicate == 'at':
                obj, loc = args
                all_objects_mentioned.add(obj)
                all_locations_mentioned.add(loc)
            elif predicate == 'usable':
                spanner = args[0]
                self.spanner_objects.add(spanner)
                all_objects_mentioned.add(spanner)
            elif predicate in ['loose', 'tightened']:
                nut = args[0]
                self.nut_objects.add(nut)
                all_objects_mentioned.add(nut)
            elif predicate == 'carrying':
                 man = args[0]
                 all_objects_mentioned.add(man)


        # Infer locations from all mentioned locations
        self.locations = all_locations_mentioned

        # Infer man object (the object in 'at' that is not a spanner or nut, or the one in 'carrying')
        potential_men = set()
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj not in self.spanner_objects and obj not in self.nut_objects:
                     potential_men.add(obj)
             elif match(fact, "carrying", "*", "*"):
                  man, spanner = get_parts(fact)[1:]
                  potential_men.add(man)

        # If there's only one potential man, assume it's the man object
        if len(potential_men) == 1:
            self.man_object = potential_men.pop()
        else:
             # Fallback: Assume the first object in initial state 'at' fact that isn't a spanner/nut
             # This might be necessary if the man is not initially carrying anything
             for fact in self.initial_state:
                 if match(fact, "at", "*", "*"):
                     obj, loc = get_parts(fact)[1:]
                     if obj not in self.spanner_objects and obj not in self.nut_objects:
                         self.man_object = obj
                         break


        # Store initial nut locations (nuts are static)
        for fact in self.initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in self.nut_objects:
                    self.nut_locations[obj] = loc

        # --- Compute shortest paths ---
        self.distances = {}
        # Handle case with no locations or links
        if not self.locations:
             # If there are goal nuts but no locations, it's unsolvable unless nuts are at man's implicit start loc
             # But domain requires locations. Return early, __call__ will handle inf distances.
             return

        for start_loc in self.locations:
            self.distances[(start_loc, start_loc)] = 0
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()

                if current_loc in self.link_graph:
                    for neighbor in self.link_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = dist + 1
                            queue.append((neighbor, dist + 1))

        # --- Identify goal nuts ---
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                self.goal_nuts.add(nut)

    def __call__(self, node):
        state = node.state

        # --- Extract state information ---
        man_loc = None
        carried_spanners = set()
        current_usable_spanners = set()
        current_loose_nuts = set()
        available_spanners_at_locs_map = {} # spanner -> location (includes usable/non-usable)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and args[0] == self.man_object:
                man_loc = args[1]
            elif predicate == 'carrying' and args[0] == self.man_object:
                carried_spanners.add(args[1])
            elif predicate == 'usable':
                current_usable_spanners.add(args[0])
            elif predicate == 'loose':
                current_loose_nuts.add(args[0])
            elif predicate == 'at' and args[0] in self.spanner_objects:
                 # This spanner is at a location, not carried
                 available_spanners_at_locs_map[args[0]] = args[1]


        # --- Heuristic Calculation ---
        loose_goal_nuts = {n for n in self.goal_nuts if n in current_loose_nuts}
        num_loose_goal_nuts = len(loose_goal_nuts)

        # If all goal nuts are tightened, heuristic is 0
        if num_loose_goal_nuts == 0:
            return 0

        h = 0

        # 1. Cost for tighten actions
        h += num_loose_goal_nuts

        # Count usable spanners the man is carrying
        carried_usable_spanners = carried_spanners.intersection(current_usable_spanners)

        # 2. Cost for pickup actions needed
        num_spanners_needed_from_locs = max(0, num_loose_goal_nuts - len(carried_usable_spanners))
        h += num_spanners_needed_from_locs

        # Filter available spanners at locations to only include usable ones
        available_usable_spanners_at_locs_map = {
            s: loc for s, loc in available_spanners_at_locs_map.items()
            if s in current_usable_spanners
        }

        # Check if enough usable spanners exist in total
        total_usable_spanners_available = len(carried_usable_spanners) + len(available_usable_spanners_at_locs_map)
        if total_usable_spanners_available < num_loose_goal_nuts:
             # Problem is unsolvable from this state
             return math.inf # Use math.inf for infinity

        # 3. Movement cost to reach the first nut location
        loose_nut_locs = {self.nut_locations[n] for n in loose_goal_nuts}
        min_dist_man_to_nut = math.inf

        # Ensure man_loc is a known location before calculating distance
        if man_loc is not None and man_loc in self.locations:
            for loc in loose_nut_locs:
                if (man_loc, loc) in self.distances:
                     min_dist_man_to_nut = min(min_dist_man_to_nut, self.distances[(man_loc, loc)])
                # else: loc is unreachable from man_loc, state is unsolvable.
        # If man_loc is None or not in locations, or no nut location is reachable, return inf
        if min_dist_man_to_nut == math.inf:
             return math.inf # Unreachable goal nuts

        h += min_dist_man_to_nut

        # 4. Movement cost to reach the first spanner location, if pickups are needed
        if num_spanners_needed_from_locs > 0:
            available_spanner_locs = set(available_usable_spanners_at_locs_map.values())
            min_dist_man_to_spanner = math.inf

            # Ensure man_loc is a known location before calculating distance
            if man_loc is not None and man_loc in self.locations:
                for loc in available_spanner_locs:
                     if (man_loc, loc) in self.distances:
                          min_dist_man_to_spanner = min(min_dist_man_to_spanner, self.distances[(man_loc, loc)])
                # else: No available spanner location is reachable from man_loc.
            # If man_loc is None or not in locations, or no spanner location is reachable, return inf
            if min_dist_man_to_spanner == math.inf:
                 return math.inf # Unreachable spanners needed

            h += min_dist_man_to_spanner

        return h
