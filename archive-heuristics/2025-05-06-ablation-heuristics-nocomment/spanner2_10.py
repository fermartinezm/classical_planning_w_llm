import math
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def bfs(adj, start_node):
    """Computes shortest path distances from start_node using BFS."""
    distances = {node: math.inf for node in adj}
    if start_node in distances: # Handle case where start_node might not be in adj (e.g., isolated location)
        distances[start_node] = 0
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            if current_node in adj: # Check if current_node has neighbors
                for neighbor in adj[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(adj, all_nodes):
    """Computes shortest path distances between all pairs of nodes."""
    all_dist = {}
    # Ensure all potential nodes (locations) are considered, even if isolated
    for node in all_nodes:
        all_dist[node] = bfs(adj, node)
    return all_dist


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    This heuristic estimates the cost to reach the goal by simulating a greedy
    sequence of actions. The goal is to tighten all specified nuts. Each nut
    requires the man to be at the nut's location with a usable spanner, followed
    by a tighten action. The heuristic estimates the cost by iteratively
    selecting the next task for the man: if he needs a spanner, he goes to the
    nearest available usable spanner; otherwise, he goes to the nearest remaining
    loose goal nut. The cost accumulated is the sum of walk actions (shortest
    path distances) and pickup/tighten actions (cost 1 each). This heuristic
    is not admissible but aims to be informative for greedy best-first search.

    Assumptions:
    - Nuts do not move from their initial locations. Their locations are determined
      from the initial state.
    - Spanners become unusable after one tighten action.
    - If the man is carrying a spanner, it is assumed to be usable for the purpose
    of the next tighten action in the heuristic calculation. The heuristic does
    not track the specific usability status of carried spanners beyond this
    optimistic assumption.
    - The graph of locations connected by 'link' predicates is undirected.
    - The problem is solvable (enough usable spanners exist and locations are reachable).
    If not, the heuristic returns infinity.

    Heuristic Initialization:
    - Parses static facts to build the location graph based on 'link' predicates.
    - Collects all relevant locations from 'link' facts and initial state 'at' facts.
    - Computes all-pairs shortest path distances between these locations using BFS.
    - Parses goal facts to identify the set of goal nuts.
    - Parses initial state facts to identify the man's name and the initial locations
    of all goal nuts (assuming nut locations are static).

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set of goal nuts that are currently 'loose' in the state. If this set is empty, the goal is reached, and the heuristic is 0.
    2. Identify the man's current location by finding the 'at' fact involving the man object. If the man object name was not identified during initialization or his location is not found, return infinity.
    3. Determine if the man is currently carrying any spanner by finding the 'carrying' fact involving the man object. Assume this carried spanner is usable for the next tighten action if one is needed.
    4. Identify the set of usable spanners that are currently 'at' a location (available for pickup) by finding 'at' facts for spanners and 'usable' facts for spanners.
    5. Check if the total number of usable spanners (carried + available at locations) is sufficient for the number of loose goal nuts. If not, return infinity, as the problem is unsolvable from this state.
    6. Initialize the heuristic value `h` to 0.
    7. Set the man's `current_location` to his actual location in the state. If this location is not part of the precomputed distance graph, return infinity.
    8. Set a flag `currently_carrying_usable` based on whether the man was carrying a spanner in the state (using the optimistic assumption).
    9. Create working copies of the sets of `remaining_loose_nuts` and `remaining_available_spanners`.
    10. Start a loop that continues as long as there are `remaining_loose_nuts`.
    11. Inside the loop:
        a. If `currently_carrying_usable` is true:
           i. Find the nut in `remaining_loose_nuts` whose location is nearest to the `current_location` using the precomputed distances.
           ii. If the nearest nut's location is unreachable (distance is infinity), return infinity.
           iii. Add the shortest distance found to `h` (cost of walking to the nut).
           iv. Add 1 to `h` (cost of the tighten action).
           v. Update `current_location` to the location of the nut just processed.
           vi. Set `currently_carrying_usable` to false, as the spanner is now considered used.
           vii. Remove the processed nut from `remaining_loose_nuts`.
        b. If `currently_carrying_usable` is false:
           i. Find the spanner in `remaining_available_spanners` whose location is nearest to the `current_location` using the precomputed distances.
           ii. If no available spanner is found (should be caught by total count check, but safety first) or the nearest spanner's location is unreachable (distance is infinity), return infinity.
           iii. Add the shortest distance found to `h` (cost of walking to the spanner).
           iv. Add 1 to `h` (cost of the pickup action).
           v. Update `current_location` to the location of the spanner just picked up.
           vi. Remove the processed spanner from `remaining_available_spanners`.
           vii. Set `currently_carrying_usable` to true, as the man is now carrying a spanner.
    12. Once the loop finishes (all loose goal nuts are processed), return the final accumulated value of `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Precompute location graph and distances
        self.locations = set()
        adj = {}

        # Collect locations from link facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                adj.setdefault(loc1, []).append(loc2)
                adj.setdefault(loc2, []).append(loc1) # Assuming links are bidirectional

        # Collect locations from initial state 'at' facts
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                self.locations.add(parts[2])

        self.dist = compute_all_pairs_shortest_paths(adj, self.locations)

        # 2. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])

        # 3. Identify initial nut locations and man name
        self.nut_locations = {}
        initial_spanners = set()
        initial_nuts = set()

        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj.startswith('nut'):
                     self.nut_locations[obj] = loc
                     initial_nuts.add(obj)
                 elif obj.startswith('spanner'):
                     initial_spanners.add(obj)

        # Identify man name: Assume the first object in an 'at' fact in initial state
        # that is not a nut or spanner is the man. This is a heuristic guess.
        self.man_name = None
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and not parts[1].startswith('nut') and not parts[1].startswith('spanner'):
                 self.man_name = parts[1]
                 break

        # If man name couldn't be identified, the heuristic cannot function.
        # We handle this by returning infinity in __call__.


    def __call__(self, node):
        state = node.state

        # Check if man name was successfully identified in __init__
        if self.man_name is None:
            # Cannot compute heuristic without man name
            return math.inf

        # 1. Identify loose goal nuts in current state
        current_loose_goal_nuts = set()
        for nut_name in self.goal_nuts:
            if f'(loose {nut_name})' in state:
                 current_loose_goal_nuts.add(nut_name)

        # If all goal nuts are tightened, heuristic is 0
        if not current_loose_goal_nuts:
            return 0

        # 2. Identify man's current location
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1] == self.man_name:
                 man_location = parts[2]
                 break

        # Man must always be somewhere in a valid state.
        if man_location is None:
             return math.inf # Error state? Unreachable man?

        # Check if man's current location is in the computed distances graph
        if man_location not in self.dist:
             # Man is in a location not connected to the graph. Unsolvable.
             return math.inf

        # 3. Determine if man is carrying a usable spanner
        carrying_spanner_name = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'carrying' and len(parts) == 3 and parts[1] == self.man_name:
                 carrying_spanner_name = parts[2]
                 break

        # Optimistic assumption: if carrying any spanner, it's usable for the next tighten.
        man_has_usable_spanner = (carrying_spanner_name is not None)

        # 4. Identify available usable spanners at locations
        available_usable_spanners = {} # {spanner_name: location}
        spanners_at_location = {} # {spanner_name: location}
        usable_spanners_set = set() # {spanner_name}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('spanner'):
                 spanners_at_location[parts[1]] = parts[2]
            elif parts[0] == 'usable' and len(parts) == 2 and parts[1].startswith('spanner'):
                 usable_spanners_set.add(parts[1])

        for spanner_name in usable_spanners_set:
             if spanner_name in spanners_at_location:
                 available_usable_spanners[spanner_name] = spanners_at_location[spanner_name]

        # 5. Check solvability based on spanner count
        num_spanners_needed = len(current_loose_goal_nuts)
        num_usable_spanners_available_total = len(available_usable_spanners) + (1 if man_has_usable_spanner else 0)

        if num_usable_spanners_available_total < num_spanners_needed:
             return math.inf # Unsolvable

        # 6. Initialize heuristic
        h = 0
        current_location = man_location
        currently_carrying_usable = man_has_usable_spanner
        remaining_loose_nuts = set(current_loose_goal_nuts)
        remaining_available_spanners = dict(available_usable_spanners) # Copy

        # 7. Greedy simulation loop
        while remaining_loose_nuts:
            if currently_carrying_usable:
                # Go to nearest remaining loose nut
                nearest_nut = None
                min_dist = math.inf
                target_loc = None

                for nut_name in remaining_loose_nuts:
                    nut_loc = self.nut_locations.get(nut_name) # Get precomputed location
                    if nut_loc is None:
                         # Goal nut location not found in initial state - problem definition issue?
                         return math.inf # Cannot proceed

                    # Check if current_location and target_loc are in the computed distances
                    if current_location not in self.dist or nut_loc not in self.dist.get(current_location, {}):
                         # Cannot reach this nut location from current location
                         return math.inf # Unsolvable path

                    d = self.dist[current_location][nut_loc]
                    if d < min_dist:
                        min_dist = d
                        nearest_nut = nut_name
                        target_loc = nut_loc

                # If min_dist is still infinity, no remaining nut is reachable
                if min_dist == math.inf:
                     return math.inf # Unsolvable

                h += min_dist # Walk cost
                h += 1 # Tighten cost
                current_location = target_loc
                remaining_loose_nuts.remove(nearest_nut)
                currently_carrying_usable = False # Spanner used

            else: # Man is not carrying a usable spanner
                # Go to nearest remaining available usable spanner
                nearest_spanner_name = None
                min_dist = math.inf
                target_loc = None

                for spanner_name, spanner_loc in remaining_available_spanners.items():
                    # Check if current_location and target_loc are in the computed distances
                    if current_location not in self.dist or spanner_loc not in self.dist.get(current_location, {}):
                         # Cannot reach this spanner location from current location
                         return math.inf # Unsolvable path

                    d = self.dist[current_location][spanner_loc]
                    if d < min_dist:
                        min_dist = d
                        nearest_spanner_name = spanner_name
                        target_loc = spanner_loc

                # If min_dist is still infinity, no remaining available spanner is reachable
                if min_dist == math.inf:
                     return math.inf # Unsolvable (caught by total count check earlier, but safety)

                h += min_dist # Walk cost
                h += 1 # Pickup cost
                current_location = target_loc
                del remaining_available_spanners[nearest_spanner_name]
                currently_carrying_usable = True # Now carrying a usable spanner

        # 8. Return accumulated heuristic value
        return h
