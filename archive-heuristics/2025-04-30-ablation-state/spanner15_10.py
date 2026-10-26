from collections import deque
import fnmatch

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If the base class is not provided externally, uncomment this dummy definition:
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        raise NotImplementedError


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
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts
    that are part of the goal. It calculates the estimated cost for the first nut
    based on the man's current location and available usable spanners, and adds
    a fixed cost for each subsequent nut.

    # Assumptions
    - The man can only carry one spanner at a time.
    - Each usable spanner can tighten exactly one nut.
    - Links between locations are bidirectional for walking.
    - The problem is solvable (enough usable spanners exist and locations are connected).
    - The man object is the one appearing as the first argument in any 'carrying' fact,
      or defaults to 'bob' if no 'carrying' fact exists initially.

    # Heuristic Initialization
    - Identifies all location objects from static 'link' facts and initial state 'at' facts.
    - Precomputes all-pairs shortest path distances between all identified locations
      based on the 'link' predicates using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to identify:
       - The man's current location (`ManLoc`).
       - The set of currently loose nuts (`LooseNuts`).
       - The locations of all nuts (`NutLocations`).
       - The set of currently usable spanners (`UsableSpanners`).
       - The locations of all spanners (`SpannerLocations`).
       - Whether the man is carrying a spanner, and if so, which one (`ManCarryingSpanner`).
    2. Filter the set of loose nuts to include only those that are part of the goal (`CurrentLooseGoalNuts`).
    3. If `CurrentLooseGoalNuts` is empty, the heuristic is 0 (goal state reached for relevant nuts).
    4. Check if the man is currently carrying a usable spanner (`ManHasUsableCarried`).
    5. Estimate the cost to tighten the *first* loose goal nut:
       - If `ManHasUsableCarried` is True: The cost is the minimum shortest path distance
         from `ManLoc` to any location of a loose goal nut, plus 1 action for the
         `tighten_nut` action.
       - If `ManHasUsableCarried` is False: The cost is the minimum over all pairs of
         (usable spanner S at L_S, loose goal nut N at L_N) of the sum of distances
         `dist(ManLoc, L_S) + dist(L_S, L_N)`, plus 1 action for `pickup_spanner`
         and 1 action for `tighten_nut`. This requires finding the best spanner-nut pair
         for the first sequence of actions.
    6. Estimate the cost for the *remaining* loose goal nuts: After the first nut is
       tightened, the man is at that nut's location and needs to tighten the rest.
       Each remaining nut requires getting a *new* usable spanner and performing the
       tightening sequence. This sequence involves walking to a spanner, picking it up,
       walking to the nut, and tightening it. A fixed cost of 4 actions
       (representing walk_to_spanner + pickup + walk_to_nut + tighten) is added for
       each remaining loose goal nut. This is a simplification and assumes a constant
       average cost for subsequent nuts.
    7. The total heuristic value is the sum of the estimated cost for the first nut
       and the estimated cost for the remaining nuts. Return infinity if any required
       location is unreachable.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing all-pairs shortest paths
        and identifying locations.
        """
        self.goals = task.goals
        self.static = task.static

        self.location_links = {}
        self.locations = set()

        # Extract locations and links from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.location_links.setdefault(loc1, set()).add(loc2)
                self.location_links.setdefault(loc2, set()).add(loc1) # Links are bidirectional
            # Locations can also appear in initial 'at' facts or goal 'at' facts
            # Although spanner goals are 'tightened', not 'at'.

        # Ensure all locations mentioned in initial state 'at' facts are included
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 self.locations.add(parts[2])

        # Convert locations set to a list for consistent indexing if needed, or just iterate
        self.locations = list(self.locations)

        # Compute all-pairs shortest paths using BFS from each location
        self.all_dist = {}
        for start_loc in self.locations:
            self.all_dist[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """Performs BFS from a start_node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Start node is not a known location, cannot compute distances within this graph
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has any links defined
            if current_node in self.location_links:
                for neighbor in self.location_links[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Retrieves the precomputed distance between two locations."""
        if loc1 not in self.all_dist or loc2 not in self.all_dist.get(loc1, {}):
             # This means loc1 or loc2 was not included in the initial locations set,
             # or loc2 is unreachable from loc1 within the graph we built.
             return float('inf')
        return self.all_dist[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Parse state facts
        current_locations = {}
        loose_nuts = set()
        tightened_nuts = set()
        usable_spanners = set()
        man_carrying_spanner = None
        man_object = None # Need to identify the man object

        # First pass to identify the man and basic facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'carrying':
                man_object = parts[1] # Assume the carrier is the man
                man_carrying_spanner = parts[2]
            elif parts[0] == 'loose':
                loose_nuts.add(parts[1])
            elif parts[0] == 'tightened':
                tightened_nuts.add(parts[1])
            elif parts[0] == 'usable':
                usable_spanners.add(parts[1])

        # If man not found via 'carrying', assume 'bob' from examples
        # This is a potential point of failure if the man object name is different
        # and there's no 'carrying' fact in the initial state.
        if man_object is None:
             # Fallback assumption based on example instances
             man_object = 'bob'

        # Second pass to get locations
        man_location = None
        nut_locations = {}
        spanner_locations = {}

        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                  obj, loc = parts[1], parts[2]
                  current_locations[obj] = loc
                  if obj == man_object:
                       man_location = loc
                  # Infer type based on sets populated in the first pass
                  if obj in loose_nuts or obj in tightened_nuts: # It's a nut
                       nut_locations[obj] = loc
                  elif obj in usable_spanners or obj == man_carrying_spanner: # It's a spanner (usable or carried)
                       spanner_locations[obj] = loc

        # If man_location is None, the state is likely invalid or unreachable
        if man_location is None:
             return float('inf')


        # 2. Filter loose nuts to only include those that are goals
        goal_nuts = {get_parts(g)[1] for g in self.goals if match(g, "(tightened *)")}
        current_loose_goal_nuts = loose_nuts.intersection(goal_nuts)

        # 3. If no loose goal nuts, return 0
        if not current_loose_goal_nuts:
            return 0

        # Check if man is carrying a *usable* spanner
        man_has_usable_carried = (man_carrying_spanner is not None and man_carrying_spanner in usable_spanners)

        # Get usable spanners that are currently on the ground
        usable_spanners_on_ground = {s for s in usable_spanners if s in spanner_locations}

        # 4. Estimate cost for the first nut
        h = 0
        nuts_to_tighten_count = len(current_loose_goal_nuts)

        if man_has_usable_carried:
            # Man uses the carried spanner for one nut.
            # Cost = walk from ManLoc to closest loose goal nut + tighten.
            min_dist_to_closest_nut = float('inf')
            closest_nut = None
            for nut in current_loose_goal_nuts:
                loc_n = nut_locations.get(nut)
                if loc_n is not None:
                    dist = self.get_distance(man_location, loc_n)
                    if dist != float('inf') and dist < min_dist_to_closest_nut:
                        min_dist_to_closest_nut = dist
                        closest_nut = nut

            if closest_nut is None or min_dist_to_closest_nut == float('inf'):
                 # Should not happen in a solvable state if nuts are reachable
                 return float('inf') # Unreachable nuts

            h += min_dist_to_closest_nut + 1 # walk + tighten
            nuts_to_tighten_count -= 1

        else: # Man is not carrying a usable spanner
            # Cost = get a usable spanner + go to a loose goal nut + tighten.
            # Find the minimum cost sequence: ManLoc -> L_S -> L_N -> Tighten
            min_cost_first_nut_sequence = float('inf')

            if not usable_spanners_on_ground:
                 # No usable spanners available on the ground and not carrying one.
                 # If there are loose nuts, this state is likely unsolvable.
                 if nuts_to_tighten_count > 0:
                      return float('inf')
                 else:
                      # This case should have been caught by the initial check for loose nuts.
                      # Returning 0 here indicates no work needed.
                      return 0

            for nut in current_loose_goal_nuts:
                loc_n = nut_locations.get(nut)
                if loc_n is None: continue # Nut location unknown

                for spanner in usable_spanners_on_ground:
                    loc_s = spanner_locations.get(spanner)
                    if loc_s is None: continue # Spanner location unknown (should be in spanner_locations if on ground)

                    # Cost = dist(ManLoc, L_S) + 1 (pickup) + dist(L_S, L_N) + 1 (tighten)
                    dist_man_s = self.get_distance(man_location, loc_s)
                    dist_s_n = self.get_distance(loc_s, loc_n)

                    if dist_man_s != float('inf') and dist_s_n != float('inf'):
                        cost = dist_man_s + 1 + dist_s_n + 1
                        min_cost_first_nut_sequence = min(min_cost_first_nut_sequence, cost)

            if min_cost_first_nut_sequence == float('inf'):
                 # No reachable usable spanner or no reachable loose nut
                 return float('inf')

            h += min_cost_first_nut_sequence
            nuts_to_tighten_count -= 1


        # 5. Estimate cost for remaining nuts
        # Each remaining nut requires getting a *new* usable spanner and tightening.
        # This involves: walk to spanner + pickup + walk to nut + tighten.
        # Approximate cost per remaining nut: 4 actions.
        # This is a simplification; the actual cost depends on the man's location
        # after tightening the previous nut and the locations of remaining spanners/nuts.
        h += nuts_to_tighten_count * 4

        return h
