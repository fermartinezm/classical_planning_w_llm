# Assuming Heuristic base class is available, otherwise remove inheritance
# from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential empty fact string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, unless args has wildcards
    # A simple zip and all check is sufficient for basic matching
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# class spannerHeuristic(Heuristic): # Use this if inheriting
class spannerHeuristic: # Use this if not inheriting from a specific base
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the number of untightened nuts, the cost for Bob to reach the nearest
    untightened nut, and the cost for Bob to acquire a usable spanner if he doesn't
    already have one.

    # Assumptions
    - Bob must visit each nut's location to tighten it.
    - Bob must be carrying a usable spanner to tighten a nut.
    - Movement between linked locations costs 1 action.
    - Picking up or putting down a spanner costs 1 action.
    - Links between locations are bidirectional.
    - There is always a path between any two locations in the graph (or unreachable targets result in infinite cost).
    - There is always a usable spanner available somewhere if needed (or needing one when none are available results in infinite cost).

    # Heuristic Initialization
    - Extracts the goal nuts from the task's goal conditions.
    - Builds a graph representing the locations and links from static facts to calculate distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all nuts that need to be tightened according to the goal.
    2. Count how many of these goal nuts are *not* currently tightened in the state (`NumUntightened`).
    3. If `NumUntightened` is 0, the heuristic is 0 (goal state).
    4. Initialize the heuristic cost with `NumUntightened` (representing the tighten actions).
    5. Find Bob's current location.
    6. Find the locations of all untightened goal nuts.
    7. Calculate the shortest path distance from Bob's current location to the nearest untightened nut location (`min_dist_to_any_nut`) using BFS on the location graph. Add this distance to the cost.
    8. Determine if Bob is currently carrying a usable spanner.
    9. If Bob is *not* carrying a usable spanner:
       a. Initialize spanner acquisition cost to 0.
       b. Check if Bob is carrying *any* spanner (even a non-usable one). If yes, add 1 to the spanner acquisition cost (for the putdown action).
       c. Find the locations of all usable spanners.
       d. Calculate the shortest path distance from Bob's current location to the nearest usable spanner location (`min_dist_to_spanner`) using BFS.
       e. Add `min_dist_to_spanner + 1` (for the pickup action) to the spanner acquisition cost.
       f. Add the total spanner acquisition cost to the heuristic cost.
    10. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal nuts and building the location graph."""
        # Extract goal nuts from goal conditions like (tightened nut1)
        self.goal_nuts = {get_parts(goal)[1] for goal in task.goals if match(goal, "tightened", "*")}
        # Build the location graph from static facts like (link loc1 loc2)
        self.location_graph = self._build_location_graph(task.static)

    def _build_location_graph(self, static_facts):
        """Build an adjacency list representation of the location graph from link facts."""
        graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) == 3: # Ensure parts is not empty and has correct structure
                loc1, loc2 = parts[1], parts[2]
                graph.setdefault(loc1, set()).add(loc2)
                graph.setdefault(loc2, set()).add(loc1) # Assuming bidirectional links
        return graph

    def _bfs_min_distance_to_set(self, start_loc, target_locations):
        """
        Calculate shortest path distance from start_loc to the nearest location
        in target_locations using BFS. Returns float('inf') if no target is reachable.
        """
        if not target_locations:
            return float('inf') # No targets to reach

        # Check if start_loc is one of the targets
        if start_loc in target_locations:
            return 0

        # If start_loc is not a node in the graph and not a target, it's isolated
        # and cannot reach any target in the graph.
        if start_loc not in self.location_graph and start_loc not in target_locations:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}
        min_dist = float('inf')

        while queue:
            current_loc, dist = queue.popleft()

            # Check neighbors only if current_loc is in the graph
            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if neighbor in target_locations:
                        min_dist = min(min_dist, dist + 1)
                    # Optimization: If we found a path to a target, we don't need to explore further from this node
                    # if the current distance already exceeds the best found distance.
                    # However, BFS guarantees shortest path, so just checking visited is enough.
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            # If current_loc is not in graph, it has no neighbors to explore

        return min_dist # Will be inf if no target is reachable

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state

        # 1. Identify untightened goal nuts
        untightened_nuts = {nut for nut in self.goal_nuts if f"(tightened {nut})" not in state}

        # 2. If NumUntightened is 0, heuristic is 0
        num_untightened = len(untightened_nuts)
        if num_untightened == 0:
            return 0

        # 3. Initialize cost
        cost = num_untightened # Cost for the tighten actions

        # Parse state to find locations and spanner info
        bob_location = None
        obj_locations = {} # Map object -> location
        usable_spanners_set = set()
        bob_carrying_spanner = None # The spanner Bob is carrying, if any

        for fact in state:
             parts = get_parts(fact)
             if not parts: continue # Skip malformed facts

             if parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 obj_locations[obj] = loc
                 if obj == "bob":
                     bob_location = loc
             elif parts[0] == "usable" and len(parts) == 2:
                 spanner = parts[1]
                 usable_spanners_set.add(spanner)
             elif parts[0] == "carrying" and len(parts) == 3 and parts[1] == "bob":
                 bob_carrying_spanner = parts[2]

        # Ensure we found Bob's location
        if bob_location is None:
             # Bob's location is unknown, cannot proceed
             return float('inf')

        # 6. Calculate min_dist_to_any_nut
        # Get locations for untightened nuts that are currently 'at' a location
        target_nut_locations = {obj_locations[nut] for nut in untightened_nuts if nut in obj_locations}

        # If there are untightened nuts but none have a known location in the state,
        # or if Bob's location is not connected to any nut location, the state might be unsolvable.
        # The BFS handles reachability. If target_nut_locations is empty but num_untightened > 0,
        # it means untightened nuts are not 'at' any location, which is unexpected for this domain.
        # Let's assume untightened nuts are always 'at' a location.
        # If target_nut_locations is empty, min_dist_to_any_nut will be inf.

        min_dist_to_any_nut = self._bfs_min_distance_to_set(bob_location, target_nut_locations)

        # If any target nut location is unreachable from Bob, the state is likely unsolvable
        if min_dist_to_any_nut == float('inf'):
             return float('inf')

        cost += min_dist_to_any_nut

        # 8. Determine if Bob needs a usable spanner
        bob_has_usable_spanner = (bob_carrying_spanner is not None) and (bob_carrying_spanner in usable_spanners_set)

        # 9. If Bob does not have a usable spanner
        if not bob_has_usable_spanner:
            spanner_acquisition_cost = 0
            # If Bob is carrying a broken spanner, he needs to put it down
            if bob_carrying_spanner is not None: # He is carrying something, but it's not usable
                 spanner_acquisition_cost += 1 # Cost to put down the broken spanner

            # Find the nearest usable spanner
            # Get locations for usable spanners that are currently 'at' a location
            target_spanner_locations = {obj_locations[spanner] for spanner in usable_spanners_set if spanner in obj_locations}

            # If there are no usable spanners anywhere (or none with 'at' facts), the state is unsolvable
            if not target_spanner_locations:
                 return float('inf')

            min_dist_to_spanner = self._bfs_min_distance_to_set(bob_location, target_spanner_locations)

            # If no usable spanner is reachable from Bob, the state is unsolvable
            if min_dist_to_spanner == float('inf'):
                 return float('inf')

            spanner_acquisition_cost += min_dist_to_spanner + 1 # Cost to move to spanner + pickup
            cost += spanner_acquisition_cost

        # 10. Return total cost
        return cost
