# All necessary imports
import collections

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner planning domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing three components:
    1. The minimum number of 'tighten_nut' actions required, which is equal to the number of loose nuts that are part of the goal.
    2. The minimum number of 'pickup_spanner' actions required, which is calculated as the number of loose goal nuts minus the number of usable spanners the man is currently carrying (minimum of 0).
    3. An estimate of the 'walk' actions needed to get the man to the first location where a necessary action (either picking up a spanner if needed, or tightening a nut) can be performed. This is estimated as the shortest distance from the man's current location to the closest relevant location (closest usable spanner location if pickups are needed, otherwise the closest loose goal nut location).

    Assumptions:
    - There is exactly one man object in the domain, identifiable from the initial state or static facts.
    - Links between locations defined by the 'link' predicate are bidirectional.
    - The goal is always expressed as a conjunction of 'tightened' predicates for a specific set of nuts.
    - A usable spanner is consumed (becomes unusable) after being used in one 'tighten_nut' action.
    - The man can carry multiple spanners simultaneously.
    - All locations, nuts, and spanners relevant to achieving the goal are either reachable from the man's current location via the location graph or their inaccessibility correctly contributes to an infinite heuristic value.

    Heuristic Initialization:
    During initialization, the heuristic processes the planning task definition. It extracts the set of nuts that must be tightened to satisfy the goal. It builds a graph representing the connections between locations based on the static 'link' facts, assuming bidirectionality. It also identifies the man object, the set of all nuts, and the set of all spanners present in the initial state and static facts, which helps in parsing state information efficiently during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Check if the state satisfies the goal conditions (all goal nuts are tightened). If so, the heuristic value is 0.
    2. Determine the man's current location from the state facts. If the man's location is not found, the state is likely invalid or unsolvable, and the heuristic returns infinity.
    3. Identify the set of nuts that are part of the goal but are currently loose in the state. This is the set of nuts that still need to be tightened. Let the count be `num_nuts_to_tighten`.
    4. If `num_nuts_to_tighten` is 0, the goal is reached (already checked in step 1), return 0.
    5. Count the number of usable spanners the man is currently carrying.
    6. Calculate the minimum number of additional spanners the man needs to pick up to tighten all remaining loose goal nuts. This is `max(0, num_nuts_to_tighten - carried_usable_count)`. Let this be `num_pickups_needed`.
    7. Perform a Breadth-First Search (BFS) starting from the man's current location to compute the shortest distance to all other reachable locations in the location graph.
    8. Identify the locations of all usable spanners that are currently *not* carried by the man and are reachable according to the BFS.
    9. Identify the locations of all remaining loose goal nuts that are reachable according to the BFS.
    10. Calculate the minimum distance from the man's current location to any reachable location containing a usable spanner not carried by the man (`min_dist_to_spanner_loc`). If `num_pickups_needed > 0` but no such spanner is reachable, the problem is unsolvable from this state, return infinity.
    11. Calculate the minimum distance from the man's current location to any reachable location containing a remaining loose goal nut (`min_dist_to_nut_loc`). If `num_nuts_to_tighten > 0` but no such nut is reachable, the problem is unsolvable from this state, return infinity.
    12. Estimate walk cost.
        - If `num_pickups_needed > 0`, the man's first objective is likely to get a spanner. The walk cost is estimated as `min_dist_to_spanner_loc`.
        - If `num_pickups_needed == 0` (meaning the man is already carrying enough usable spanners), the man's first objective is to get to a nut. The walk cost is estimated as `min_dist_to_nut_loc`.
    13. The total heuristic value is the sum of `num_nuts_to_tighten` (for tighten actions), `num_pickups_needed` (for pickup actions), and the estimated `walk_cost`.
    """
    def __init__(self, task):
        # Store goal nuts
        self.goal_nuts = {fact.split()[1][:-1] for fact in task.goals if fact.startswith('(tightened ')}

        # Build location graph and identify objects from initial state and static facts
        self.location_graph = collections.defaultdict(set)
        self.locations = set()
        self.all_nuts = set()
        self.all_spanners = set()
        self.man = None

        all_relevant_facts = set(task.initial_state) | set(task.static)

        # First pass: Identify object types and locations
        locatable_objects = set()
        for fact in all_relevant_facts:
            parts = fact.split()
            pred = parts[0][1:] # Remove '('
            if pred == 'link':
                if len(parts) > 2:
                    loc1 = parts[1]
                    loc2 = parts[2][:-1] # Remove ')'
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1) # Assuming links are bidirectional
                    self.locations.add(loc1)
                    self.locations.add(loc2)
            elif pred == 'loose' or pred == 'tightened':
                if len(parts) > 1: # Ensure there's an argument
                    self.all_nuts.add(parts[1][:-1])
            elif pred == 'usable':
                 if len(parts) > 1: # Ensure there's an argument
                    self.all_spanners.add(parts[1][:-1])
            elif pred == 'carrying':
                 if len(parts) > 2: # Ensure there are two arguments
                     # The first argument of carrying is the man
                     self.man = parts[1]
                     self.all_spanners.add(parts[2][:-1]) # The second argument is a spanner
            elif pred == 'at':
                 if len(parts) > 2: # Ensure there are two arguments
                     locatable_objects.add(parts[1])
                     self.locations.add(parts[2][:-1])

        # If man wasn't identified by 'carrying', try to infer from 'at' facts
        if self.man is None:
             # Assume the single locatable object that is not a nut or spanner is the man
             potential_men = locatable_objects - self.all_nuts - self.all_spanners
             if len(potential_men) == 1:
                 self.man = list(potential_men)[0]
             else:
                 # Fallback: Assume the first object in initial 'at' that isn't a known nut/spanner
                 # This is less robust but might work for typical instances
                 for fact in task.initial_state:
                     if fact.startswith('(at '):
                         parts = fact.split()
                         if len(parts) > 2:
                             obj = parts[1]
                             if obj not in self.all_nuts and obj not in self.all_spanners:
                                 self.man = obj
                                 break

        if self.man is None:
             # Still couldn't identify the man. This is an issue for the heuristic.
             # It will likely return infinity or cause errors if the man object is needed.
             print("Warning: Could not identify the man object during heuristic initialization.")


    def bfs(self, start_node):
        """
        Performs BFS to find shortest distances from start_node to all reachable locations.

        Args:
            start_node: The starting location.

        Returns:
            A dictionary mapping reachable locations to their distance from start_node.
            Returns an empty dictionary if start_node is not a known location in the graph.
        """
        if start_node not in self.locations:
             # Start node is not a known location, cannot perform BFS
             return {}

        dist = {start_node: 0}
        queue = collections.deque([start_node])
        visited = {start_node}

        while queue:
            u = queue.popleft()
            # Check if u is in graph keys, although it should be if in self.locations
            if u in self.location_graph:
                for v in self.location_graph[u]:
                    if v not in visited:
                        visited.add(v)
                        dist[v] = dist[u] + 1
                        queue.append(v)
        return dist

    def __call__(self, state):
        # 1. Check if the goal is reached. If yes, the heuristic is 0.
        current_tightened_nuts = {fact.split()[1][:-1] for fact in state if fact.startswith('(tightened ')}
        if self.goal_nuts.issubset(current_tightened_nuts):
             return 0

        # 2. Identify the man's current location.
        man_loc = None
        if self.man: # Ensure man object was identified
            for fact in state:
                if fact.startswith('(at '):
                    parts = fact.split()
                    if len(parts) > 2:
                        obj = parts[1]
                        if obj == self.man:
                            man_loc = parts[2][:-1]
                            break

        if man_loc is None:
             # Man's location is unknown - problem state is likely invalid or unsolvable
             return float('inf')

        # 3. Identify loose goal nuts in the current state.
        current_loose_nuts = {fact.split()[1][:-1] for fact in state if fact.startswith('(loose ')}
        remaining_goal_nuts = self.goal_nuts.intersection(current_loose_nuts)

        # 4. If no loose goal nuts remain, the goal is effectively reached (all goal nuts are tightened).
        if not remaining_goal_nuts:
            return 0

        # 5. Count usable spanners currently carried by the man.
        carried_spanners = {fact.split()[2][:-1] for fact in state if fact.startswith('(carrying ')}
        usable_spanners = {fact.split()[1][:-1] for fact in state if fact.startswith('(usable ')}
        carried_usable_count = len(carried_spanners.intersection(usable_spanners))

        # 6. Calculate the minimum number of additional spanners that need to be picked up.
        num_nuts_to_tighten = len(remaining_goal_nuts)
        num_pickups_needed = max(0, num_nuts_to_tighten - carried_usable_count)

        # 7. Compute shortest distances from the man's current location to all reachable locations.
        dist_map = self.bfs(man_loc)

        # 8. Find locations of usable spanners that are not currently carried by the man.
        usable_spanners_at_locs_set = set()
        for fact in state:
            if fact.startswith('(at '):
                parts = fact.split()
                if len(parts) > 2:
                    obj = parts[1]
                    loc = parts[2][:-1]
                    # Check if it's a spanner, usable, not carried, and at a reachable location
                    if obj in self.all_spanners and obj in usable_spanners and obj not in carried_spanners and loc in dist_map:
                         usable_spanners_at_locs_set.add(loc)

        # 9. Find locations of the remaining loose goal nuts.
        remaining_nut_locs_set = set()
        for fact in state:
            if fact.startswith('(at '):
                parts = fact.split()
                if len(parts) > 2:
                    obj = parts[1]
                    loc = parts[2][:-1]
                    # Check if it's a remaining goal nut and at a reachable location
                    if obj in remaining_goal_nuts and loc in dist_map:
                         remaining_nut_locs_set.add(loc)

        # 10. Calculate minimum distance to a usable spanner location (if pickups are needed).
        min_dist_to_spanner_loc = float('inf')
        if num_pickups_needed > 0:
            if not usable_spanners_at_locs_set:
                # Need to pick up spanners, but no usable spanners are available anywhere reachable.
                # Problem is unsolvable from this state.
                return float('inf')
            min_dist_to_spanner_loc = min(dist_map[loc] for loc in usable_spanners_at_locs_set)

        # 11. Calculate minimum distance to a remaining loose goal nut location.
        min_dist_to_nut_loc = float('inf')
        if remaining_nut_locs_set: # Should be true if num_nuts_to_tighten > 0 and nuts are located
             min_dist_to_nut_loc = min(dist_map[loc] for loc in remaining_nut_locs_set)
        elif num_nuts_to_tighten > 0:
             # Loose goal nuts exist but are not at any reachable location.
             # Problem is unsolvable from this state.
             return float('inf')


        # 12. Estimate walk cost.
        #    This is the distance to the first location where a necessary action (pickup or tighten) can occur.
        walk_cost = 0
        if num_pickups_needed > 0:
            # Need to pick up a spanner first.
            walk_cost = min_dist_to_spanner_loc
        elif num_nuts_to_tighten > 0:
            # Don't need to pick up a spanner immediately (have enough carried), go to a nut.
            walk_cost = min_dist_to_nut_loc
        # If num_nuts_to_tighten == 0, walk_cost remains 0 (goal reached).

        # 13. Calculate the total heuristic value.
        #    Sum of tighten actions, pickup actions, and estimated initial walk cost.
        heuristic_value = num_nuts_to_tighten + num_pickups_needed + walk_cost

        return heuristic_value
