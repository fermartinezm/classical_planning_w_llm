from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If running standalone without the planner framework, uncomment the following dummy class
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         return 0 # Placeholder


# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_distances(start_location, locations, links):
    """
    Computes shortest path distances from a start_location to all other locations
    using BFS on the link graph.

    Args:
        start_location: The starting location name.
        locations: A set of all location names in the graph.
        links: A dictionary representing the adjacency list {loc: [neighbor1, neighbor2, ...]}.

    Returns:
        A dictionary mapping location names to their shortest distance from start_location.
        Returns float('inf') for unreachable locations.
    """
    distances = {loc: float('inf') for loc in locations}
    if start_location in distances: # Ensure start_location is one of the known locations
        distances[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            if current_loc in links: # Ensure the location has outgoing links
                for neighbor in links[current_loc]:
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)

    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all required nuts.
    It uses a greedy approach: if the man is carrying a usable spanner, he goes to the closest nut.
    If not, he finds the best pair of (available usable spanner on ground, remaining loose nut)
    that minimizes the total travel cost to the spanner, pickup, travel to the nut, and tighten.
    The estimated cost for this pair is added, and the process repeats from the nut's location.

    # Assumptions:
    - The man can carry at most one spanner at a time (inferred from singular predicate).
    - A spanner becomes unusable after tightening one nut.
    - There are enough usable spanners available (initially or on the ground) to tighten all required nuts in solvable problems.
    - The link predicate is bidirectional.
    - The man's name can be inferred from 'carrying' facts or is 'bob' as a fallback.
    - Nuts in the goal state that are not currently 'tightened' are assumed to be 'loose' and need tightening.

    # Heuristic Initialization
    - Extracts the set of all locations and the graph of links between them from static facts.
    - Extracts the set of nuts that must be in the 'tightened' state in the goal.
    - Collects all nut and spanner names mentioned in initial state, goals, and static facts.
    - Attempts to identify the man's name.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location. If not found, return infinity (unless goal is met).
    2. Identify all nuts that are in the goal state and are not currently 'tightened'. Assume these are 'loose' and need tightening. Store their locations. If no such nuts, return 0.
    3. Identify all usable spanners that are currently 'at' a location (on the ground) and store their locations.
    4. Check if the man is currently 'carrying' a spanner, and if that spanner is 'usable'.
    5. Compute shortest path distances from the man's current location and all usable spanner locations on the ground to all other relevant locations (locations of nuts, spanners, and linked locations).
    6. Initialize the total heuristic cost `h` to 0.
    7. Create mutable lists of the remaining goal nuts and available usable spanners on the ground.
    8. Enter a loop that continues as long as there are remaining goal nuts:
        a. If the man is currently carrying a usable spanner:
           i. Find the remaining loose nut closest to the man's current location using computed distances.
           ii. If no reachable nut, return infinity.
           iii. Add the distance to this nut plus 1 (for the tighten action) to `h`.
           iv. Update the man's current location to the nut's location.
           v. The carried spanner is now considered used/unusable for future steps in the heuristic calculation. Set the 'currently carrying usable' flag to False.
        b. If the man is not currently carrying a usable spanner:
           i. Find the pair of (available usable spanner on the ground, remaining loose nut) that minimizes the total travel distance: distance from man to spanner + distance from spanner to nut.
           ii. If no usable spanners are available on the ground, return infinity.
           iii. If no reachable spanner-nut pair, return infinity.
           iv. Add the minimum total travel distance found, plus 1 (for pickup) and 1 (for tighten), to `h`.
           v. Update the man's current location to the nut's location.
           vi. Remove the selected spanner from the list of available usable spanners on the ground.
           vii. Remove the selected nut from the list of remaining goal nuts.
           viii. The man picked up and used a spanner, so he is not carrying a usable spanner for the next iteration. Set the 'currently carrying usable' flag to False.
    9. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal nuts (those that must be tightened)
        self.goal_tightened_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened":
                if len(parts) > 1:
                    self.goal_tightened_nuts.add(parts[1])

        self.locations = set()
        self.links = {} # Adjacency list
        self.all_nuts = set()
        self.all_spanners = set()
        self.man_name = None # Will try to identify man name

        # Collect all locations and build links graph from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "link":
                if len(parts) > 2:
                    loc1, loc2 = parts[1], parts[2]
                    self.locations.add(loc1)
                    self.locations.add(loc2)
                    self.links.setdefault(loc1, []).append(loc2)
                    self.links.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Collect all nuts and spanners from initial state and goals
        # And collect all locations mentioned anywhere
        facts_to_parse = set(task.initial_state) | set(task.goals) | set(self.static)
        for fact in facts_to_parse:
             parts = get_parts(fact)
             if parts[0] in ["loose", "tightened"]:
                 if len(parts) > 1: self.all_nuts.add(parts[1])
             elif parts[0] == "usable":
                 if len(parts) > 1: self.all_spanners.add(parts[1])
             elif parts[0] == "carrying":
                 if len(parts) > 2:
                     # The first argument of 'carrying' is the man
                     self.man_name = parts[1]
                     # The second argument is a spanner
                     self.all_spanners.add(parts[2])
             elif parts[0] == "at":
                 if len(parts) > 2:
                     # The second argument is a location
                     self.locations.add(parts[2])
                     # The first argument is a locatable object (man, nut, spanner)
                     # We collect the object name, will identify man later if needed.
                     # obj_name = parts[1]

        # If man_name wasn't found via 'carrying', try to infer it from 'at' facts in initial state.
        # Assume the man is the single object of type 'man'. We infer this by finding
        # the object in an 'at' fact that is not a known nut or spanner.
        if self.man_name is None:
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 if parts[0] == "at":
                     obj_name = parts[1]
                     # If this object is not a known nut and not a known spanner, assume it's the man.
                     if obj_name not in self.all_nuts and obj_name not in self.all_spanners:
                         self.man_name = obj_name
                         break

        # Fallback: Assume man is 'bob' if still not identified (very fragile, relies on example)
        if self.man_name is None:
             self.man_name = 'bob' # HACK: Assuming man is named 'bob'

        # Ensure all locations mentioned in links are in the set (already done)
        # Ensure locations from state/goal are in the set (already done)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of strings)

        # 1. Identify the man's current location.
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break

        # If man_loc is None, the man is not at any location. This state is likely invalid
        # or represents an unsolvable situation unless the goal is already met.
        if man_loc is None:
             if self.is_goal(state): return 0
             # Man is not at any location and goal is not met. Unsolvable from here.
             return float('inf')


        # 2. Identify loose nuts that are in the goal state and their locations.
        goal_nuts = set() # Names of nuts that are loose AND in the goal
        nut_loc = {}      # Map nut name to its location
        tightened_nuts_in_state = set() # Names of nuts that are tightened in state

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "tightened":
                 if len(parts) > 1:
                     tightened_nuts_in_state.add(parts[1])
            elif parts[0] == "at":
                 if len(parts) > 2 and parts[1] in self.all_nuts: # Check if the object is a known nut
                      nut_loc[parts[1]] = parts[2]


        # Nuts needing tightening are those in the goal that are NOT tightened in the current state.
        # We assume nuts in the goal that are not tightened are implicitly loose (if they exist in the state).
        # Let's refine: A nut needs tightening if it's in the goal AND it's explicitly 'loose' in the state.
        # If a nut is in the goal but neither 'loose' nor 'tightened' in the state, its status is unknown/problematic.
        # Given the examples, nuts are either loose or tightened.
        # Let's stick to: nuts needing tightening are those in the goal AND not tightened in state.
        nuts_needing_tightening = self.goal_tightened_nuts - tightened_nuts_in_state

        # From these, keep only those whose location is known in the current state.
        # If a nut needing tightening has no location in the state, it's unreachable.
        goal_nuts = {n for n in nuts_needing_tightening if n in nut_loc}

        # If any nut needing tightening is not in nut_loc, it's unreachable.
        if len(goal_nuts) < len(nuts_needing_tightening):
             return float('inf')


        # If all goal nuts are already tightened, heuristic is 0.
        if not goal_nuts:
            return 0

        # 3. Identify usable spanners on the ground and the carried spanner.
        usable_spanners_on_ground = set() # Names of usable spanners on the ground
        spanner_loc_on_ground = {}      # Map spanner name to its location
        spanner_carried = None          # Name of spanner carried by man (at most one)
        spanner_carried_usable = False  # Is the carried spanner usable?

        usable_spanners_in_state = set() # Names of usable spanners in state

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "usable":
                 if len(parts) > 1:
                     usable_spanners_in_state.add(parts[1])
            elif parts[0] == "carrying":
                 if len(parts) > 2 and parts[1] == self.man_name: # Check if man is carrying
                     spanner_carried = parts[2]
            elif parts[0] == "at":
                 if len(parts) > 2 and parts[1] in self.all_spanners: # Check if object is a known spanner
                      # If this spanner is usable and at a location, it's on the ground
                      if parts[1] in usable_spanners_in_state:
                           usable_spanners_on_ground.add(parts[1])
                           spanner_loc_on_ground[parts[1]] = parts[2]

        # Check if the carried spanner (if any) is usable
        if spanner_carried is not None and spanner_carried in usable_spanners_in_state:
             spanner_carried_usable = True

        # Check if there are enough usable spanners (carried + on ground) for the remaining nuts.
        # If not, the state is unsolvable.
        if len(usable_spanners_on_ground) + (1 if spanner_carried_usable else 0) < len(goal_nuts):
             return float('inf')


        # 4. Compute shortest path distances from relevant locations.
        # Relevant locations are man_loc, all spanner_loc_on_ground, all nut_loc for goal nuts.
        relevant_locations_in_state = {man_loc} | set(nut_loc.values()) | set(spanner_loc_on_ground.values())
        # Ensure these locations are part of the known locations graph
        all_possible_locations = self.locations | relevant_locations_in_state

        # Compute BFS from current_man_loc and all spanner_loc_on_ground.
        distances = {}
        locations_to_bfs_from = {man_loc} | set(spanner_loc_on_ground.values())
        for start_loc in locations_to_bfs_from:
             distances[start_loc] = bfs_distances(start_loc, all_possible_locations, self.links)

        def get_dist(l1, l2):
             if l1 not in distances or l2 not in distances[l1]:
                 # This means l1 was not a location we ran BFS from, or l2 is unreachable.
                 # If l1 == l2, dist is 0. Otherwise, unreachable.
                 if l1 == l2: return 0
                 return float('inf')
             return distances[l1][l2]


        # 5. Implement the greedy loop.
        h = 0
        remaining_goal_nuts = list(goal_nuts) # Use list for easy removal
        # Need to track usable spanners that haven't been assigned to a nut yet *in this heuristic calculation*.
        remaining_usable_spanners_on_ground_names = list(usable_spanners_on_ground) # Use list for easy removal
        current_man_loc = man_loc
        currently_carrying_usable = spanner_carried_usable

        while remaining_goal_nuts:
            if currently_carrying_usable:
                # Man has a usable spanner, go to closest remaining nut
                min_dist = float('inf')
                best_nut_name = None

                for nut_name in remaining_goal_nuts:
                    target_loc = nut_loc[nut_name]
                    d = get_dist(current_man_loc, target_loc)
                    if d == float('inf'):
                         # This nut is unreachable from current man location. Problem unsolvable.
                         return float('inf')

                    if d < min_dist:
                        min_dist = d
                        best_nut_name = nut_name

                # If min_dist is still inf, no remaining nut is reachable.
                if min_dist == float('inf'):
                     return float('inf')

                # Cost: walk to nut + tighten
                h += min_dist + 1
                current_man_loc = nut_loc[best_nut_name]
                remaining_goal_nuts.remove(best_nut_name)
                currently_carrying_usable = False # Spanner used

            else: # Man needs a spanner
                min_cost = float('inf')
                best_spanner_name = None
                best_nut_name = None

                available_spanners_to_consider = []
                for s_name in remaining_usable_spanners_on_ground_names:
                    # Ensure the spanner location is known (it should be if it's in usable_spanners_on_ground)
                    if s_name in spanner_loc_on_ground:
                         available_spanners_to_consider.append((s_name, spanner_loc_on_ground[s_name]))

                if not available_spanners_to_consider:
                    # No usable spanners left on the ground and not carrying one.
                    # Problem unsolvable from here unless goal is met (checked at loop start).
                    return float('inf')

                for spanner_name, spanner_location in available_spanners_to_consider:
                    for nut_name in remaining_goal_nuts:
                        nut_location = nut_loc[nut_name]

                        # Cost = travel to spanner + pickup + travel to nut + tighten
                        dist_to_spanner = get_dist(current_man_loc, spanner_location)
                        dist_spanner_to_nut = get_dist(spanner_location, nut_location)

                        if dist_to_spanner == float('inf') or dist_spanner_to_nut == float('inf'):
                             # This spanner-nut path is unreachable. Skip this pair.
                             continue

                        current_cost = dist_to_spanner + 1 + dist_spanner_to_nut + 1 # pickup + tighten

                        if current_cost < min_cost:
                            min_cost = current_cost
                            best_spanner_name = spanner_name
                            best_nut_name = nut_name

                # After checking all pairs, if min_cost is still inf, no reachable spanner-nut pair exists.
                if min_cost == float('inf'):
                     return float('inf')

                h += min_cost
                current_man_loc = nut_loc[best_nut_name]
                remaining_usable_spanners_on_ground_names.remove(best_spanner_name)
                remaining_goal_nuts.remove(best_nut_name)
                currently_carrying_usable = False # Man picked up a spanner, but it's used after this step.

        return h

    def is_goal(self, state):
        """Check if the state is a goal state."""
        # Assuming goal is a set of facts
        return self.goals <= state

