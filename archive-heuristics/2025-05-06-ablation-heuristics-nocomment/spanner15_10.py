import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Removes parentheses and splits a PDDL fact string into parts."""
    # Ensure fact is a string and has expected format
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Handle unexpected format, maybe log a warning or raise error
        # For this domain, facts are expected to be '(predicate arg1 arg2 ...)'
        return []
    return fact[1:-1].split()

# Helper function for pattern matching facts
def match(fact, *args):
    """Checks if a fact matches a pattern of parts."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    return len(parts) >= len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to compute shortest paths using BFS
def bfs_shortest_paths(locations, links):
    """
    Computes shortest path distances between all pairs of locations
    using BFS starting from each location.

    Args:
        locations: A set of all location names.
        links: A set of (loc1, loc2) tuples representing bidirectional links.

    Returns:
        A dictionary distances[loc1][loc2] = shortest_path_distance.
        Returns float('inf') if locations are unreachable.
    """
    adj = collections.defaultdict(set)
    for l1, l2 in links:
        adj[l1].add(l2)
        adj[l2].add(l1)

    distances = {}
    all_locations_list = list(locations) # Use a list for consistent iteration order

    for start_node in all_locations_list:
        distances[start_node] = {}
        queue = collections.deque([(start_node, 0)])
        visited = {start_node}
        distances[start_node][start_node] = 0 # Distance to self is 0

        while queue:
            current_node, dist = queue.popleft()

            # Check if current_node is in adj to avoid errors for isolated locations
            if current_node in adj:
                for neighbor in adj[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        # Mark unreachable locations with infinity
        for loc in all_locations_list:
             if loc not in distances[start_node]:
                 distances[start_node][loc] = float('inf')

    return distances


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        Estimates the cost to reach the goal (tighten all specified nuts)
        by summing the minimum number of tighten actions required and the
        estimated cost to get the man and a usable spanner to the location
        of the first nut that needs tightening. The cost to get the man and
        spanner ready for the first nut depends on whether the man is already
        carrying a usable spanner. If so, it's the travel cost to the closest
        loose nut. If not, it's the travel cost to the closest usable spanner,
        plus the pickup cost, plus the travel cost from the spanner location
        to the closest loose nut location from there.

    Assumptions:
        - The problem is solvable (i.e., there are enough usable spanners
          initially to tighten all goal nuts, and all locations are connected
          or relevant locations are reachable). The heuristic returns infinity
          if it detects an obvious unsolvable state (e.g., no usable spanners
          available when needed, or required locations are unreachable).
        - Nuts do not move.
        - Spanners do not move unless carried by the man.
        - Links between locations are bidirectional.
        - There is exactly one man object.

    Heuristic Initialization:
        - Parses static 'link' facts to build the location graph.
        - Computes all-pairs shortest paths between locations using BFS.
        - Identifies all nuts and their fixed locations from initial state/static facts.
        - Identifies the man object.
        - Identifies the set of goal nuts.

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify the set of loose nuts that are part of the goal. If this set is empty, the goal is reached, and the heuristic is 0.
        2. Identify the man's current location from the state.
        3. Identify the set of usable spanners currently available (either carried by the man or on the ground with the 'usable' predicate).
        4. If there are loose nuts but no usable spanners available at all, the problem is unsolvable from this state; return infinity.
        5. The base heuristic cost is the number of loose nuts, as each requires at least one 'tighten_nut' action.
        6. Add the estimated cost to get the man and a usable spanner ready to tighten the *first* nut. This cost depends on whether the man is currently carrying a usable spanner:
            a. If the man is carrying a usable spanner:
               - Find the loose nut location that is closest to the man's current location using precomputed shortest paths.
               - If no loose nut location is reachable, return infinity.
               - Add the shortest path distance from the man's location to this closest loose nut location to the cost.
            b. If the man is not carrying a usable spanner:
               - Find the usable spanner on the ground that is closest to the man's current location using precomputed shortest paths. If no usable spanner is reachable, return infinity. Let this location be L_S_closest.
               - Add the shortest path distance to L_S_closest plus 1 (for the pickup action) to the cost.
               - From L_S_closest, find the loose nut location that is closest using precomputed shortest paths. If no loose nut location is reachable from L_S_closest, return infinity. Let this location be L_N_closest_from_spanner.
               - Add the shortest path distance from L_S_closest to L_N_closest_from_spanner to the cost.
        7. Return the total computed cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Build location graph and compute shortest paths
        locations = set()
        links = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                links.add((loc1, loc2))
                locations.add(loc1)
                locations.add(loc2)

        # Add locations mentioned in initial state or goals if not already in links
        for fact in initial_state | self.goals:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 locations.add(loc)

        self.distances = bfs_shortest_paths(locations, links)

        # 2. Identify nuts and their locations (nuts are locatable and their location is static)
        self.nut_locations = {}
        all_objects = set()
        # Collect all objects from initial state and goals that appear as parameters
        for fact in initial_state | self.goals:
             parts = get_parts(fact)
             # Add all parts except the predicate name itself
             if len(parts) > 1:
                 all_objects.update(parts[1:])

        # Find nuts (objects appearing in loose/tightened) and their locations (from initial/static 'at')
        nut_objects = {obj for obj in all_objects if any(match(f, "loose", obj) or match(f, "tightened", obj) for f in initial_state | self.goals)}

        for nut in nut_objects:
            for fact in initial_state | static_facts:
                if match(fact, "at", nut, "*"):
                    _, _, loc = get_parts(fact)
                    self.nut_locations[nut] = loc
                    break
            # If a nut location isn't found, it's an issue with the problem definition
            # We assume valid problems where all nut locations are specified initially/statically.


        # 3. Identify the man object (assuming there's only one man)
        self.man_name = None
        # A man is locatable and can carry things. Look for 'carrying' or 'at' + not spanner/nut
        locatable_objects_initial = {get_parts(fact)[1] for fact in initial_state if match(fact, "at", "*", "*")}
        spanner_objects_initial = {get_parts(fact)[1] for fact in initial_state if match(fact, "usable", "*")} # Usable implies spanner

        for obj_name in locatable_objects_initial:
             if obj_name not in nut_objects and obj_name not in spanner_objects_initial:
                  self.man_name = obj_name
                  break # Found the man

        if self.man_name is None:
             # Fallback: If no object fits the above, check 'carrying' predicate directly
             for fact in initial_state:
                  if match(fact, "carrying", "*", "*"):
                       self.man_name = get_parts(fact)[1]
                       break
        
        if self.man_name is None:
             # This indicates a problem parsing the man object.
             print("Error: Could not identify the man object.")
             # In a real planner, this might raise an error or make the task unsolvable.
             # For heuristic computation, we might return infinity if we can't proceed.
             # Assuming valid input for now.
             pass


        # 4. Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

    def __call__(self, node):
        state = node.state

        # 1. Identify loose nuts that are goals
        loose_goal_nuts = {nut for nut in self.goal_nuts if f'(loose {nut})' in state}

        if not loose_goal_nuts:
            return 0 # Goal reached

        # 2. Identify man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
             # Man's location should always be known in a valid state
             # This suggests an invalid state representation or domain issue
             print(f"Error: Man {self.man_name} location not found in state.")
             return float('inf') # Cannot compute heuristic

        # 3. Identify usable spanners (carried or on ground)
        usable_spanners_ground = {} # {spanner_name: location}
        usable_spanner_carried = None # spanner_name

        # Find all usable spanners in the current state
        usable_spanners_in_state = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        # Determine if usable spanner is carried or on ground
        for spanner in usable_spanners_in_state:
            is_carried = False
            for fact in state:
                if match(fact, "carrying", self.man_name, spanner):
                    usable_spanner_carried = spanner
                    is_carried = True
                    break
            if not is_carried:
                 # If not carried, it must be on the ground at some location
                 for fact in state:
                     if match(fact, "at", spanner, "*"):
                         usable_spanners_ground[spanner] = get_parts(fact)[2]
                         break
                 # If a usable spanner is neither carried nor at a location, it's an inconsistency.
                 # Assuming valid states, this won't happen.

        # 4. Check solvability based on spanners
        if not usable_spanners_ground and usable_spanner_carried is None:
             # No usable spanners left, but nuts still need tightening
             return float('inf')

        # 5. Base cost: number of loose goal nuts
        cost = len(loose_goal_nuts)

        # 6. Cost to get man and spanner ready for the first nut
        loose_nut_locations = {self.nut_locations[nut] for nut in loose_goal_nuts if nut in self.nut_locations} # Ensure nut location is known

        if not loose_nut_locations:
             # Should not happen if loose_goal_nuts is not empty and nut_locations were found in init
             # unless a goal nut's location wasn't in initial/static facts, which is a problem definition issue.
             return float('inf')


        if usable_spanner_carried is not None:
            # Man is carrying a usable spanner, just need to get to the closest nut location
            min_dist_to_nut_loc = float('inf')
            if man_location in self.distances:
                for nut_loc in loose_nut_locations:
                    if nut_loc in self.distances[man_location]:
                        min_dist_to_nut_loc = min(min_dist_to_nut_loc, self.distances[man_location][nut_loc])

            if min_dist_to_nut_loc == float('inf'):
                 return float('inf') # Cannot reach any loose nut location

            cost += min_dist_to_nut_loc

        else:
            # Man is not carrying a usable spanner, needs to get one first
            min_dist_to_spanner = float('inf')
            closest_spanner_loc = None

            # Find the closest usable spanner on the ground
            if man_location in self.distances:
                for spanner_loc in usable_spanners_ground.values():
                     if spanner_loc in self.distances[man_location]:
                         dist = self.distances[man_location][spanner_loc]
                         if dist < min_dist_to_spanner:
                             min_dist_to_spanner = dist
                             closest_spanner_loc = spanner_loc

            if min_dist_to_spanner == float('inf'):
                 return float('inf') # Cannot reach any usable spanner on the ground

            # Cost to get to the closest spanner and pick it up
            cost += min_dist_to_spanner + 1 # +1 for pickup action

            # From the spanner location, find the closest loose nut location
            min_dist_spanner_to_nut = float('inf')
            if closest_spanner_loc in self.distances:
                for nut_loc in loose_nut_locations:
                     if nut_loc in self.distances[closest_spanner_loc]:
                         min_dist_spanner_to_nut = min(min_dist_spanner_to_nut, self.distances[closest_spanner_loc][nut_loc])

            if min_dist_spanner_to_nut == float('inf'):
                 return float('inf') # Cannot reach any loose nut location from the closest spanner

            cost += min_dist_spanner_to_nut # Cost to walk from spanner location to closest nut location

        return cost
