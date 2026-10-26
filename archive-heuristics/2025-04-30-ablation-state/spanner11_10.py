# Import necessary modules
import collections # For deque used in BFS
import fnmatch
import math # For float('inf')

# Assuming Heuristic base class is available from the planning framework
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass

# Helper function (from example)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It uses a greedy strategy: if the man is carrying a usable spanner, he moves to the closest loose goal nut and tightens it. If he is not carrying a usable spanner, he moves to the closest location with a usable spanner and picks it up. The estimated cost is the sum of walk actions (based on shortest path distances) and pickup/tighten actions.

    # Assumptions:
    - There is exactly one man.
    - Links between locations are bidirectional.
    - Nut locations are static and provided in the initial state.
    - Spanner usability is a one-time use per spanner (consumed by tighten_nut).
    - Solvable problems have enough usable spanners available initially (carried or at locations) to tighten all goal nuts.
    - The goal is always to tighten a specific set of nuts.
    - The location graph is connected for all relevant locations (man start, nut locations, spanner locations).

    # Heuristic Initialization
    The heuristic initializes by:
    1. Inferring the names of the man, spanners, nuts, and locations from the initial state, goal, and static facts.
    2. Building a graph of locations based on the static `link` facts.
    3. Computing all-pairs shortest path distances between locations using Breadth-First Search (BFS).
    4. Identifying the set of goal nuts and their static locations from the task's goals and initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Extract the man's current location, the set of usable spanners the man is carrying, the set of usable spanners at locations, and the set of goal nuts that are still loose along with their static locations.
    2. If no goal nuts are loose, the heuristic is 0 (goal state).
    3. Initialize the estimated cost to 0.
    4. Initialize the current man location, the count of spanners carried, a list of locations of loose goal nuts, and a list of locations with usable spanners.
    5. Start a loop that continues as long as there are loose goal nuts remaining:
        a. Check if the man is currently carrying any usable spanners (`num_spanners_carried > 0`).
        b. If yes:
            i. Find the location of the loose goal nut that is closest to the man's current location using the precomputed distances.
            ii. Add the shortest distance to this location to the total cost (representing walk actions).
            iii. Add 1 to the total cost (representing the `tighten_nut` action).
            iv. Update the man's current location to the location of the tightened nut.
            v. Decrement the count of spanners carried (as one is used).
            vi. Remove one instance of the tightened nut's location from the list of loose goal nut locations to visit.
        c. If no (man needs a spanner):
            i. Find the location with a usable spanner that is closest to the man's current location using the precomputed distances.
            ii. If no usable spanners are available at any location, return infinity (representing an unsolvable state from here).
            iii. Add the shortest distance to this spanner location to the total cost (representing walk actions).
            iv. Add 1 to the total cost (representing the `pickup_spanner` action).
            v. Update the man's current location to the spanner location.
            vi. Increment the count of spanners carried.
            vii. Remove one instance of the picked-up spanner's location from the list of available spanner locations.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information and computing distances."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Infer object names by type ---
        # This is a necessary step as the Task object doesn't provide object types directly.
        # We infer types based on predicates objects appear in within the initial state, goals, and static facts.
        all_potential_objects = set()
        facts_set = set(initial_state) | set(self.goals) | set(static_facts) # Use sets for faster lookup

        for fact in facts_set:
            parts = get_parts(fact)
            all_potential_objects.update(parts[1:]) # Add all objects mentioned

        self.man_name = None
        self.spanner_names = set()
        self.nut_names = set()
        self.location_names = set()

        for obj in all_potential_objects:
            is_man = any(get_parts(f)[0] == 'carrying' and len(get_parts(f)) > 1 and get_parts(f)[1] == obj for f in facts_set)
            is_spanner = any((get_parts(f)[0] == 'usable' and len(get_parts(f)) > 1 and get_parts(f)[1] == obj) or (get_parts(f)[0] == 'carrying' and len(get_parts(f)) > 2 and get_parts(f)[2] == obj) for f in facts_set)
            is_nut = any(get_parts(f)[0] in ['loose', 'tightened'] and len(get_parts(f)) > 1 and get_parts(f)[1] == obj for f in facts_set)
            is_location = any((get_parts(f)[0] == 'at' and len(get_parts(f)) > 2 and get_parts(f)[2] == obj) or (get_parts(f)[0] == 'link' and (len(get_parts(f)) > 1 and get_parts(f)[1] == obj or (len(get_parts(f)) > 2 and get_parts(f)[2] == obj))) for f in facts_set)

            if is_man:
                self.man_name = obj # Assumes exactly one man
            if is_spanner:
                self.spanner_names.add(obj)
            if is_nut:
                self.nut_names.add(obj)
            if is_location:
                self.location_names.add(obj)

        # --- Build location graph and compute distances ---
        self.location_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) > 2:
                l1, l2 = parts[1], parts[2]
                # Ensure locations are in the inferred set (handle potential inference errors gracefully)
                if l1 in self.location_names and l2 in self.location_names:
                    self.location_graph.setdefault(l1, set()).add(l2)
                    self.location_graph.setdefault(l2, set()).add(l1) # Links are bidirectional
                # else: Ignore links involving objects not identified as locations

        self.dist = {}
        # Compute distances only for locations we identified
        for start_node in self.location_names:
            self.dist[start_node] = {}
            # Perform BFS from start_node
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                (curr, d) = queue.popleft()
                self.dist[start_node][curr] = d
                if curr in self.location_graph:
                    for neighbor in self.location_graph[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))

        # --- Identify goal nuts and their static locations ---
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened' and len(parts) > 1:
                self.goal_nuts.add(parts[1])

        self.nut_locations = {} # {nut: location}
        # Nut locations are static, find them in the initial state
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) > 2 and parts[1] in self.nut_names:
                self.nut_locations[parts[1]] = parts[2]

        # Basic check: ensure all goal nuts have a known static location
        # This check is not strictly necessary for the heuristic logic itself,
        # but helps identify potential issues with the problem definition or inference.
        # We proceed assuming valid problems.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract current state information ---
        current_man_loc = None
        current_spanners_carried_usable = []
        current_spanners_at_loc_usable = {} # {location: [spanner1, ...], ...}
        current_loose_goal_nuts = {} # {nut: location}

        # Use a set for faster lookup of facts in the current state
        state_set = set(state)

        for fact in state_set:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) > 2:
                obj, loc = parts[1], parts[2]
                if obj == self.man_name:
                    current_man_loc = loc
                elif obj in self.spanner_names:
                    if f"(usable {obj})" in state_set:
                        current_spanners_at_loc_usable.setdefault(loc, []).append(obj)
                # Nut locations are static, already in self.nut_locations
            elif parts[0] == 'carrying' and len(parts) > 2:
                m, s = parts[1], parts[2]
                if m == self.man_name and s in self.spanner_names:
                   if f"(usable {s})" in state_set:
                     current_spanners_carried_usable.append(s)
            elif parts[0] == 'loose' and len(parts) > 1:
                nut = parts[1]
                if nut in self.goal_nuts:
                    # Ensure nut has a known static location before adding to loose goals
                    if nut in self.nut_locations:
                         current_loose_goal_nuts[nut] = self.nut_locations[nut]
                    # else: Ignore goal nut if its location is unknown (problem definition issue)


        # If all goal nuts are tightened, heuristic is 0.
        if not current_loose_goal_nuts:
            return 0

        # --- Heuristic Calculation ---
        cost = 0
        num_spanners_carried = len(current_spanners_carried_usable)
        # Create a mutable list of locations for loose goal nuts
        # Use a list because multiple nuts might be at the same location, and we need to visit that location once per nut.
        loose_goal_nut_locations_list = list(current_loose_goal_nuts.values())
        # Create a mutable list of locations with usable spanners
        # Use a list because multiple usable spanners might be at the same location.
        available_spanner_locations_list = [loc for loc, spanners in current_spanners_at_loc_usable.items() for s in spanners]

        # Greedy loop: while there are loose goal nuts, decide whether to go for a nut or a spanner.
        # Decision: If man has a spanner, go to the closest nut. If not, go to the closest spanner.
        while loose_goal_nut_locations_list:
            if num_spanners_carried > 0:
                # Man has a spanner, go tighten the closest loose goal nut
                closest_nut_loc = None
                min_dist_to_nut = math.inf

                # Find the closest loose goal nut location
                for nut_loc in loose_goal_nut_locations_list:
                    # Ensure locations are in the distance map (should be if inference and graph building were successful)
                    if current_man_loc in self.dist and nut_loc in self.dist[current_man_loc]:
                         dist = self.dist[current_man_loc][nut_loc]
                         if dist < min_dist_to_nut:
                             min_dist_to_nut = dist
                             closest_nut_loc = nut_loc
                    # else: If a location is not in the distance map, it's unreachable.
                    # This implies an unsolvable state or a problem definition issue (disconnected graph).
                    # Return infinity in this case.

                # If min_dist_to_nut is still inf, it means either loose_goal_nut_locations_list was empty (handled before loop)
                # or none of the nut locations are reachable from the current man location.
                if min_dist_to_nut == math.inf:
                     return math.inf # Unreachable goal

                cost += min_dist_to_nut # Walk action cost
                cost += 1 # tighten_nut action cost
                current_man_loc = closest_nut_loc
                num_spanners_carried -= 1
                loose_goal_nut_locations_list.remove(closest_nut_loc) # Remove one instance of this location

            else: # Man needs a spanner
                # Go pick up the closest usable spanner
                closest_spanner_loc = None
                min_dist_to_spanner = math.inf

                # Find the closest usable spanner location
                for spanner_loc in available_spanner_locations_list:
                     # Ensure locations are in the distance map
                     if current_man_loc in self.dist and spanner_loc in self.dist[current_man_loc]:
                         dist = self.dist[current_man_loc][spanner_loc]
                         if dist < min_dist_to_spanner:
                             min_dist_to_spanner = dist
                             closest_spanner_loc = spanner_loc
                     # else: If a location is not in the distance map, it's unreachable.
                     # Return infinity.

                # If min_dist_to_spanner is still inf, it means no usable spanners are available at locations
                # and the man is not carrying one. If there are still loose nuts, the problem is unsolvable.
                if min_dist_to_spanner == math.inf:
                    # This state is a dead end for a solvable problem. Return infinity.
                    return math.inf

                # Add cost to reach the spanner and pick it up
                cost += min_dist_to_spanner # Walk action cost
                cost += 1 # pickup_spanner action cost
                current_man_loc = closest_spanner_loc
                num_spanners_carried += 1
                available_spanner_locations_list.remove(closest_spanner_loc) # Remove one instance of this location

        return cost
