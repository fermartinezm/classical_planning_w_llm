from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It sums the estimated costs for:
    1. Tightening each loose goal nut.
    2. Picking up necessary usable spanners from the ground.
    3. Moving the man to the locations relevant for tightening nuts and acquiring spanners.

    # Assumptions
    - Nuts stay in their initial locations.
    - Spanners are consumed (become not usable) after one use.
    - The man is the only agent.
    - The location graph is connected (or relevant parts are connected).
    - Object types (man, nut, spanner, location) can be inferred from initial state/goal facts based on predicate usage.

    # Heuristic Initialization
    - Builds the location graph from `link` facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the man, nuts, spanners, and goal nuts by inspecting predicates in the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal nuts that are currently `loose`. Let this count be `N_loose_goals`.
    2. If `N_loose_goals` is 0, the heuristic is 0 (goal state).
    3. Initialize heuristic `h = N_loose_goals` (representing the `tighten_nut` actions).
    4. Determine the man's current location.
    5. Count the number of `usable` spanners the man is currently `carrying`.
    6. Calculate the number of additional usable spanners the man needs to pick up from the ground: `spanners_needed_from_ground = max(0, N_loose_goals - carried_usable_count)`.
    7. Add `spanners_needed_from_ground` to `h` (representing the `pickup_spanner` actions).
    8. Identify the locations of all loose goal nuts.
    9. If `spanners_needed_from_ground > 0`, identify the locations of all usable spanners on the ground. Find the `spanners_needed_from_ground` closest usable ground spanner locations to the man's current location.
    10. Define the set of "required locations" the man must visit. This set includes all loose nut locations and the selected closest usable ground spanner locations (if spanners are needed).
    11. Calculate the sum of distances from the man's current location to each location in the set of required locations. Add this sum to `h` (representing a simplified, overestimated travel cost).
    12. Return the total `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and computing distances.
        """
        self.goals = task.goals
        self.static = task.static

        self.location_graph = {}
        self.man_names = set() # Use set to handle potential multiple mentions, pick one later
        self.nut_names = set()
        self.spanner_names = set()
        self.location_names = set()
        self.goal_nuts = set()

        # Infer object types and build location graph from initial state, static facts, and goal facts
        all_facts = task.initial_state | task.static | task.goals
        for fact in all_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'link':
                l1, l2 = args
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1)
                self.location_names.add(l1)
                self.location_names.add(l2)
            elif predicate == 'at':
                 obj, loc = args
                 self.location_names.add(loc)
                 # Type inference will happen below based on roles in predicates
            elif predicate == 'carrying':
                 man_obj, spanner_obj = args
                 self.man_names.add(man_obj)
                 self.spanner_names.add(spanner_obj)
            elif predicate == 'usable':
                 spanner_obj = args[0]
                 self.spanner_names.add(spanner_obj)
            elif predicate == 'tightened' or predicate == 'loose':
                 nut_obj = args[0]
                 self.nut_names.add(nut_obj)

        # Assume the single man object is the one found
        self.man_name = list(self.man_names)[0] if self.man_names else None
        # Note: If there are multiple men or no man mentioned in relevant predicates, this might fail.
        # Standard spanner domain instances usually have one man.

        # Ensure all locations mentioned are in the graph keys for BFS
        for loc in self.location_names:
             self.location_graph.setdefault(loc, [])

        # Store goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'tightened'}

        # Compute all-pairs shortest paths
        self.distances = self._compute_distances()

    def _compute_distances(self):
        """Compute shortest path distances between all pairs of locations using BFS."""
        distances = {}
        locations = list(self.location_graph.keys())

        for start_loc in locations:
            distances[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()
                distances[start_loc][current_loc] = dist

                for neighbor in self.location_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Ensure all known locations are in the distance map, even if isolated
        for loc in self.location_names:
             if loc not in distances:
                 distances[loc] = {loc: 0} # Distance to self is 0
             for other_loc in self.location_names:
                 if other_loc not in distances[loc]:
                      distances[loc][other_loc] = float('inf') # Unreachable

        return distances

    def get_distance(self, loc1, loc2):
        """Get shortest distance between two locations."""
        if loc1 is None or loc2 is None:
            return float('inf') # Cannot compute distance if location is unknown
        return self.distances.get(loc1, {}).get(loc2, float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {N for N in self.goal_nuts if f'(loose {N})' in state}
        N_loose_goals = len(loose_goal_nuts)

        # 2. Goal state check
        if N_loose_goals == 0:
            return 0

        # Initialize heuristic
        h = 0

        # Cost for tighten actions (1 per loose goal nut)
        h += N_loose_goals

        # Extract current state information
        man_loc = None
        carried_spanners = set()
        usable_spanners = set()
        nut_locations = {}
        ground_spanner_locations = {} # Spanners not carried by man

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at':
                obj, loc = args
                if obj == self.man_name:
                    man_loc = loc
                elif obj in self.nut_names:
                    nut_locations[obj] = loc
                elif obj in self.spanner_names:
                    ground_spanner_locations[obj] = loc # It's on the ground if 'at'
            elif predicate == 'carrying':
                m, s = args
                if m == self.man_name and s in self.spanner_names:
                    carried_spanners.add(s)
            elif predicate == 'usable':
                s = args[0]
                if s in self.spanner_names:
                    usable_spanners.add(s)

        # Ensure man_loc is found (should be in initial state)
        if man_loc is None:
             # This state is likely unreachable or invalid based on domain
             return float('inf')

        # 5. Count carried usable spanners
        carried_usable_count = len([s for s in carried_spanners if s in usable_spanners])

        # 6. Calculate spanners needed from ground
        spanners_needed_from_ground = max(0, N_loose_goals - carried_usable_count)

        # 7. Cost for pickup actions
        h += spanners_needed_from_ground

        # 8. Identify loose nut locations
        loose_nut_locations = {nut_locations.get(N) for N in loose_goal_nuts if nut_locations.get(N) is not None}

        # 9. Identify usable ground spanner locations and find closest ones if needed
        required_locations = set(loose_nut_locations) # Man must visit nut locations

        if spanners_needed_from_ground > 0:
            usable_ground_spanners_available = [s for s in usable_spanners if s not in carried_spanners]
            usable_ground_spanner_locs_available = {ground_spanner_locations.get(s) for s in usable_ground_spanners_available if ground_spanner_locations.get(s) is not None}

            # Find the spanners_needed_from_ground closest usable ground spanner locations
            dist_loc_pairs = sorted([(self.get_distance(man_loc, loc), loc) for loc in usable_ground_spanner_locs_available])

            # Add the locations of the closest spanners needed to the required locations
            for i in range(min(spanners_needed_from_ground, len(dist_loc_pairs))):
                required_locations.add(dist_loc_pairs[i][1])

        # Remove None locations if any (e.g., nut location not found, should not happen for goal nuts)
        required_locations.discard(None)

        # 11. Calculate movement cost: Sum of distances from man_loc to all required locations.
        # This is an overestimate of total travel but simple and correlates with complexity.
        movement_cost = 0
        for loc in required_locations:
             dist = self.get_distance(man_loc, loc)
             if dist == float('inf'):
                  # Cannot reach a required location - problem likely unsolvable
                  return float('inf')
             movement_cost += dist

        h += movement_cost

        # 12. Return total heuristic value
        return h
