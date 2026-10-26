from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys
from collections import deque

def get_parts(fact):
    """Helper to parse PDDL fact string into parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to match fact parts with pattern args."""
    parts = get_parts(fact)
    # Ensure we don't try to match more args than parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        Estimates the cost to reach the goal by summing the costs for each
        nut that still needs to be tightened. The cost for each loose goal nut
        includes:
        1. The cost of the 'tighten_nut' action (1).
        2. The cost of a 'pickup_spanner' action for each spanner needed but
           not currently carried by the man.
        3. The minimum travel cost for the man to reach any location that is
           either a loose goal nut's location or a location with an available
           usable spanner (if pickups are needed).

    Assumptions:
        - The problem is solvable. This implies there are enough usable spanners
          in the initial state to tighten all goal nuts, and the locations are
          connected such that the man can reach all necessary places.
        - The man object name starts with 'bob'.
        - Spanner object names start with 'spanner'.
        - Nut object names start with 'nut'.
        - The graph of locations defined by 'link' facts is connected for all
          relevant locations (man's initial location, spanner locations, nut locations).

    Heuristic Initialization:
        - Parses static facts from the task definition.
        - Identifies the man object based on naming convention.
        - Identifies all spanner and nut objects based on naming convention.
        - Builds a graph of locations based on 'link' facts and initial 'at' facts.
        - Computes all-pairs shortest paths between locations using BFS.
        - Stores the set of nuts that need to be tightened (goal nuts).

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if all goal nuts are already tightened. If yes, return 0.
        2. Identify the man's current location from the state. If location is unknown, return infinity.
        3. Identify which usable spanners the man is currently carrying from the state.
        4. Identify which usable spanners are available at specific locations (not carried) from the state.
        5. Identify the set of loose nuts that are also goal nuts, and their current locations from the state.
        6. Count the number of loose goal nuts (`num_loose_goal_nuts`).
        7. Count the number of usable spanners the man is carrying (`num_carried_usable`).
        8. Calculate the number of additional spanners the man needs to pick up (`needed_pickups = max(0, num_loose_goal_nuts - num_carried_usable)`).
        9. Initialize the heuristic value (`h`) with `num_loose_goal_nuts` (representing the 'tighten_nut' actions).
        10. Add `needed_pickups` to `h` (representing the 'pickup_spanner' actions).
        11. Determine the set of target locations the man needs to reach:
            - Include the locations of all loose goal nuts.
            - If `needed_pickups > 0`, also include the locations of available usable spanners.
        12. If there are target locations:
            - Calculate the minimum shortest path distance from the man's current location to any of these target locations using the precomputed distances.
            - Add this minimum distance to `h`. This estimates the initial travel cost to get to a useful location.
        13. If there are no target locations but `num_loose_goal_nuts > 0`, it implies a state from which the goal is likely unreachable (e.g., loose goal nuts have no location, or no usable spanners available when needed and none carried). Return infinity.
        14. Return the calculated heuristic value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Identify man object (assuming object name starts with 'bob')
        self.man = None
        for fact in task.initial_state:
             parts = get_parts(fact)
             for part in parts:
                  if part.startswith('bob'):
                       self.man = part
                       break
             if self.man: break
        if not self.man:
             # Fallback if 'bob' isn't in initial state facts (unlikely for man)
             # Assume 'bob' as per example if not found otherwise
             self.man = 'bob' # Strong assumption based on example

        # Identify spanners (assuming object name starts with 'spanner')
        self.all_spanners = set()
        for fact in static_facts:
             parts = get_parts(fact)
             for part in parts:
                  if part.startswith('spanner'):
                       self.all_spanners.add(part)
        for fact in task.initial_state:
             parts = get_parts(fact)
             for part in parts:
                  if part.startswith('spanner'):
                       self.all_spanners.add(part)

        # Identify nuts (assuming object name starts with 'nut')
        self.all_nuts = set()
        for fact in static_facts:
             parts = get_parts(fact)
             for part in parts:
                  if part.startswith('nut'):
                       self.all_nuts.add(part)
        for fact in task.initial_state:
             parts = get_parts(fact)
             for part in parts:
                  if part.startswith('nut'):
                       self.all_nuts.add(part)
        for goal in self.goals:
             parts = get_parts(goal)
             for part in parts:
                  if part.startswith('nut'):
                       self.all_nuts.add(part)


        # 2. Identify all locations
        self.locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
            elif match(fact, "at", "*", "*"): # Static objects' initial locations
                 _, obj, loc = get_parts(fact)
                 self.locations.add(loc)
        for fact in task.initial_state: # Man's initial location
             if match(fact, "at", "*", "*"):
                  _, obj, loc = get_parts(fact)
                  self.locations.add(loc)

        # 3. Build location graph
        self.graph = {loc: set() for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.graph[l1].add(l2)
                self.graph[l2].add(l1) # Links are bidirectional

        # 4. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                (curr, d) = queue.popleft()
                self.dist[start_node][curr] = d
                for neighbor in self.graph.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

        # 5. Store goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}


    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        # The goal is reached if all goal nuts are tightened.
        all_goal_nuts_tightened = True
        for goal_nut in self.goal_nuts:
            if f'(tightened {goal_nut})' not in state:
                all_goal_nuts_tightened = False
                break
        if all_goal_nuts_tightened:
            return 0

        # 2. Identify man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man's location is unknown, should not happen in valid states
             return sys.maxsize # Indicate unreachable

        # 3. Identify carried usable spanners
        carried_usable_spanners = set()
        for fact in state:
             if match(fact, "carrying", self.man, "*"):
                  spanner = get_parts(fact)[2]
                  if spanner in self.all_spanners and f'(usable {spanner})' in state:
                       carried_usable_spanners.add(spanner)

        num_carried_usable = len(carried_usable_spanners)

        # 4. Identify available usable spanners at locations
        available_usable_spanners_locs = set()
        spanner_locations = {}
        for fact in state:
             if match(fact, "at", "*", "*"):
                  obj, loc = get_parts(fact)[1:]
                  if obj in self.all_spanners:
                       spanner_locations[obj] = loc

        for spanner, loc in spanner_locations.items():
             if f'(usable {spanner})' in state and f'(carrying {self.man} {spanner})' not in state:
                  available_usable_spanners_locs.add(loc)


        # 5. Identify loose goal nuts and their locations
        loose_goal_nuts = set()
        loose_nut_locs = set()
        nut_locations = {}
        for fact in state:
             if match(fact, "at", "*", "*"):
                  obj, loc = get_parts(fact)[1:]
                  if obj in self.all_nuts:
                       nut_locations[obj] = loc

        for goal_nut in self.goal_nuts:
             if f'(loose {goal_nut})' in state:
                  loose_goal_nuts.add(goal_nut)
                  if goal_nut in nut_locations:
                       loose_nut_locs.add(nut_locations[goal_nut])
                  # else: loose goal nut has no location? Should not happen in valid states.

        num_loose_goal_nuts = len(loose_goal_nuts)

        # If num_loose_goal_nuts is 0, we should have returned 0 already.
        if num_loose_goal_nuts == 0:
             return 0

        # 6-10. Calculate base heuristic (tighten + pickup)
        h = num_loose_goal_nuts # Cost for tighten actions

        needed_pickups = max(0, num_loose_goal_nuts - num_carried_usable)
        h += needed_pickups # Cost for pickup actions

        # 11-13. Calculate travel cost
        locations_to_reach = set(loose_nut_locs)
        if needed_pickups > 0:
            locations_to_reach.update(available_usable_spanners_locs)

        min_dist_to_target = sys.maxsize # Use sys.maxsize for integer infinity

        if not locations_to_reach:
             # This case implies num_loose_goal_nuts > 0 but no loose goal nuts have locations
             # and either needed_pickups is 0 or there are no available usable spanners with locations.
             # This suggests an unsolvable state or parsing issue.
             # In a solvable problem derived from initial state, loose goal nuts should have locations.
             # If needed_pickups > 0, there should be available usable spanners with locations
             # if the problem is solvable and enough spanners exist initially.
             # Returning a large value indicates this path is likely bad/unsolvable.
             return sys.maxsize

        # Calculate min distance from man_loc to any location in locations_to_reach
        if man_loc in self.dist:
            for target_loc in locations_to_reach:
                if target_loc in self.dist[man_loc]:
                    min_dist_to_target = min(min_dist_to_target, self.dist[man_loc][target_loc])

        if min_dist_to_target == sys.maxsize:
             # Man cannot reach any required location. Unsolvable from here.
             return sys.maxsize

        h += min_dist_to_target

        # 14. Return heuristic value
        return h
