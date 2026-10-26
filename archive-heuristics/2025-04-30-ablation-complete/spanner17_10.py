from fnmatch import fnmatch
from collections import deque
import sys

# Assume Heuristic base class exists and has __init__(self, task) and __call__(self, node)
# If running standalone, you might need a dummy base class definition
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual one is not provided
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class not found. Using dummy base class.", file=sys.stderr)
    class Heuristic:
        def __init__(self, task):
            self.task = task
            pass
        def __call__(self, node):
            pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Not a valid PDDL fact string format we expect
         return [] # Return empty list for malformed facts

    # Remove parentheses and split by whitespace
    # Handle cases like '(at obj (loc))' - split might be tricky if locations have spaces,
    # but PDDL standard object names are symbols without spaces.
    return fact[1:-1].split()

def build_location_graph(static_facts):
    """Builds an adjacency list representation of the location graph from link facts."""
    graph = {}
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'link':
            # Ensure fact has correct number of parts for 'link'
            if len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                graph.setdefault(loc1, []).append(loc2)
                graph.setdefault(loc2, []).append(loc1) # Links are bidirectional
            # else: ignore malformed link fact
    return graph

def bfs(start_loc, graph):
    """Performs BFS from start_loc to find distances to all reachable locations."""
    distances = {loc: float('inf') for loc in graph}

    # If start_loc is not in the graph keys, it means it has no outgoing links
    # from the perspective of the 'link' facts processed.
    # Distance to itself is 0, to others remains inf.
    if start_loc not in graph:
        # We still need to include start_loc in the distances map
        distances[start_loc] = 0
        return distances # Return map with start_loc at 0 and others inf

    distances[start_loc] = 0
    queue = deque([(start_loc, 0)])
    visited = {start_loc}

    while queue:
        current_loc, current_dist = queue.popleft() # Use deque for efficient popleft

        # Check if current_loc has neighbors in the graph (it should, based on the check above)
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = current_dist + 1
                    queue.append((neighbor, current_dist + 1))

    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the required number of actions (tighten, pickup, and initial walk)
    to tighten all goal nuts. It counts the number of tightening actions needed, the number
    of spanner pickups required, and the estimated walking cost to reach the first necessary
    location (either a spanner or a nut). This heuristic accounts for the possibility of
    carrying multiple spanners.

    # Assumptions
    - Nuts that are goals and are currently loose need to be tightened.
    - Nuts do not change location.
    - The man can carry multiple spanners simultaneously.
    - Spanners are consumed (become not usable) after tightening one nut.
    - Links between locations are bidirectional.
    - For a solvable problem, there are enough usable spanners available (carried or on ground)
      to tighten all goal nuts.
    - All relevant objects (man, nuts, spanners) have an 'at' predicate specifying their location
      unless they are being carried.
    - All locations mentioned in 'at' or 'link' predicates are part of the location graph,
      even if some are isolated.

    # Heuristic Initialization
    - Extracts the set of nuts that are specified in the goal condition.
    - Parses the static `link` facts to build a graph representation of the locations.
    - Identifies all unique locations mentioned in the graph.
    - Precomputes shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify the set of goal nuts that are currently in a `loose` state. These are the nuts that still need to be tightened. Let this count be `N_loose_goals`.
    2.  If `N_loose_goals` is 0, the goal is reached, so the heuristic is 0.
    3.  Find the man's current location.
    4.  Count the number of `usable` spanners the man is currently carrying (`N_usable_carried`).
    5.  Identify the locations of all `usable` spanners that are currently on the ground (not being carried).
    6.  Calculate the total number of usable spanners available in the world (`N_usable_total = N_usable_carried + N_usable_available_on_ground`).
    7.  Check if `N_usable_total` is less than `N_loose_goals`. If so, the problem is unsolvable from this state, return infinity.
    8.  Calculate the number of additional usable spanners the man needs to pick up from the ground. This is `N_loose_goals` minus `N_usable_carried`, minimum 0. Let this be `N_spanners_to_pickup`.
    9.  Initialize the heuristic value: `h = N_loose_goals` (for the `tighten_nut` actions) + `N_spanners_to_pickup` (for the `pickup_spanner` actions).
    10. Estimate the walking cost to start the sequence of tasks:
        -   Identify the set of "required locations". This set includes the locations of all loose goal nuts and, if `N_spanners_to_pickup > 0`, the locations of all usable spanners currently on the ground.
        -   Find the minimum distance from the man's current location to any location in the set of required locations. Add this minimum distance to `h`.
        -   If no required locations are reachable from the man's current location (and required locations exist), return infinity.
    11. Return the calculated heuristic value `h`. This heuristic provides an estimate based on the total required actions (tighten, pickup) and the initial travel cost to reach the vicinity of the tasks. It does not model the complex sequence of movements and pickups for subsequent nuts after the first interaction.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts and precomputing distances.
        """
        self.goals = task.goals
        self.static = task.static

        # Identify the names of the nuts that need to be tightened in the goal
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g) and get_parts(g)[0] == 'tightened'}

        # Build the location graph from static link facts
        self.location_graph = build_location_graph(self.static)

        # Collect all unique locations mentioned in the graph and static facts
        all_locations = set(self.location_graph.keys())
        for fact in self.static:
             parts = get_parts(fact)
             if parts and parts[0] == 'link' and len(parts) == 3:
                  all_locations.add(parts[1])
                  all_locations.add(parts[2])
             # We might also need locations mentioned in initial 'at' facts,
             # but the BFS will only find paths within the linked graph.
             # For simplicity, assume all relevant locations are part of the linked graph.

        self.all_locations = list(all_locations)

        # Precompute shortest path distances from every location to every other location
        self.dist_maps = {loc: bfs(loc, self.location_graph) for loc in self.all_locations}

    def get_distance(self, start_loc, end_loc):
        """
        Retrieves the precomputed shortest distance between two locations.
        Returns 0 if start_loc == end_loc.
        Returns float('inf') if start_loc is not in the precomputed map or if end_loc is unreachable.
        """
        if start_loc == end_loc:
             # Distance to self is 0 if the location is known/exists in our graph context
             # If start_loc isn't in self.all_locations, it's an unknown location, maybe unreachable?
             # Assuming all relevant locations are in self.all_locations.
             return 0

        # Check if the start location is in our precomputed map
        if start_loc not in self.dist_maps:
            # This location might be isolated or not mentioned in links processed.
            # Distance to any other location is infinity unless it's the same location (handled above).
            return float('inf')

        # Retrieve distance from the precomputed map. Returns inf if end_loc is not found or unreachable.
        return self.dist_maps[start_loc].get(end_loc, float('inf'))


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Identify loose goal nuts and their locations
        loose_goal_nuts_in_state = {n for n in self.goal_nuts if f'(loose {n})' in state}
        N_loose_goals = len(loose_goal_nuts_in_state)

        # 2. If no loose goal nuts, the goal is reached
        if N_loose_goals == 0:
            return 0

        # Find locations of loose goal nuts
        loose_goal_nut_locations = set()
        for nut in loose_goal_nuts_in_state:
             # Find location of this nut in the current state
             nut_loc = None
             for fact in state:
                  parts = get_parts(fact)
                  if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] == nut:
                       nut_loc = parts[2]
                       break
             if nut_loc:
                  loose_goal_nut_locations.add(nut_loc)
             # else: nut location not found in state? Problematic state, maybe inf?
             # Assuming nuts always have an 'at' predicate in state if they exist and are relevant.


        # 3. Find man's current location
        man_loc = None
        man_name = None # Need man's name to check 'carrying' predicate

        # Find the man object and his location
        # Assuming the man is the only 'locatable' object that is not a spanner or nut,
        # or the object involved in a 'carrying' predicate.
        carried_spanners = set()
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'carrying' and len(parts) == 3:
                  man_name = parts[1]
                  carried_spanners.add(parts[2]) # Man can carry multiple spanners
             # Find man's location - check for 'at' predicate for the man object
             if man_name and parts and parts[0] == 'at' and len(parts) == 3 and parts[1] == man_name:
                  man_loc = parts[2]
                  # Found location, can stop searching for location if man_name is known

        # If man_name wasn't found via 'carrying', try finding the object at a location
        # that isn't a spanner or nut. This is less robust but might work if 'carrying' is absent.
        if man_name is None:
             for fact in state:
                  parts = get_parts(fact)
                  if parts and parts[0] == 'at' and len(parts) == 3 and not parts[1].startswith('spanner') and not parts[1].startswith('nut'):
                       man_name = parts[1]
                       man_loc = parts[2]
                       # Now that man_name is found, re-iterate state to find all carried spanners
                       # (This is slightly inefficient but ensures we get all if man_name is found late)
                       carried_spanners = set() # Reset in case it was partially filled
                       for f2 in state:
                            p2 = get_parts(f2)
                            if p2 and p2[0] == 'carrying' and len(p2) == 3 and p2[1] == man_name:
                                 carried_spanners.add(p2[2])
                       break # Found man's location and carried items, can stop

        # If man_loc is still None, something is wrong with the state representation
        if man_loc is None:
             # Cannot compute heuristic without man's location
             return float('inf') # Or handle as an error state

        # Count usable spanners carried
        N_usable_carried = 0
        for spanner_name in carried_spanners:
             if f'(usable {spanner_name})' in state:
                  N_usable_carried += 1

        # 4. Identify usable spanners not being carried and their locations
        available_usable_spanner_locations = set()
        N_usable_available_on_ground = 0
        for fact in state:
             parts = get_parts(fact)
             # Check for (at spannerX locationY)
             if parts and parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('spanner'):
                  spanner_name = parts[1]
                  spanner_loc = parts[2]
                  # Check if this spanner is usable
                  if f'(usable {spanner_name})' in state:
                       # Check if this spanner is NOT being carried by the man
                       if spanner_name not in carried_spanners:
                            available_usable_spanner_locations.add(spanner_loc)
                            N_usable_available_on_ground += 1

        # 5. Calculate total usable spanners and check sufficiency
        N_usable_total = N_usable_carried + N_usable_available_on_ground
        if N_usable_total < N_loose_goals:
             # Not enough usable spanners exist in the world to tighten all goal nuts
             return float('inf')

        # 6. Calculate number of spanners man needs to pick up
        N_spanners_to_pickup = max(0, N_loose_goals - N_usable_carried)

        # 7. Initialize heuristic value
        heuristic = N_loose_goals  # Cost for tighten_nut actions
        heuristic += N_spanners_to_pickup # Cost for pickup_spanner actions

        # 8. Estimate walking cost to start the sequence of tasks
        walking_cost_to_first_task = 0

        if N_loose_goals > 0: # Only add walk cost if there's work to do
            required_locations = set(loose_goal_nut_locations)
            if N_spanners_to_pickup > 0:
                 # If man needs to pick up spanners, the locations of available usable spanners are also required
                 required_locations.update(available_usable_spanner_locations)

            min_dist_to_required = float('inf')

            # Ensure man's location is one we can calculate distance from
            if man_loc in self.dist_maps:
                for loc in required_locations:
                    dist = self.get_distance(man_loc, loc)
                    min_dist_to_required = min(min_dist_to_required, dist)

            if min_dist_to_required == float('inf') and required_locations:
                 # Man cannot reach any required location (nuts or spanners needed for pickup)
                 return float('inf')

            walking_cost_to_first_task = min_dist_to_required

        heuristic += walking_cost_to_first_task

        # 9. Return total heuristic
        return heuristic

