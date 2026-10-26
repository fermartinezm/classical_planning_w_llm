from collections import deque
from fnmatch import fnmatch
# Assuming heuristics.heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running in the planner environment
# This is just for standalone testing or linting, the actual planner provides it.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or return empty list
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args for a strict match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_location_graph(static_facts):
    """Builds an adjacency list graph from 'link' facts."""
    graph = {}
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'link' and len(parts) == 3:
            loc1, loc2 = parts[1], parts[2]
            graph.setdefault(loc1, set()).add(loc2)
            graph.setdefault(loc2, set()).add(loc1)
    return graph

def bfs_distances(start_loc, graph):
    """Computes shortest path distances from start_loc to all reachable locations."""
    distances = {start_loc: 0}
    queue = deque([start_loc])
    visited = {start_loc}

    # If start_loc is not in the graph, it's an isolated location.
    # BFS will only find distance 0 to itself. This is correct.

    while queue:
        current_loc = queue.popleft()
        # Only process if current_loc is in the graph (i.e., has links)
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts
    specified in the goal. It considers the cost of tightening each nut, the cost
    of acquiring a spanner if needed, and the cost of movement to reach the nut locations.

    # Assumptions
    - Bob (the man) is the only agent.
    - Bob can only carry one spanner at a time.
    - All nuts needing tightening are initially loose.
    - All usable spanners remain usable.
    - The locations form a graph defined by 'link' facts. Unlinked locations are isolated.
    - The heuristic uses a simplified movement cost: sum of distances from the starting
      point of the nut-tightening tour to each nut location.

    # Heuristic Initialization
    - Extract the goal conditions to identify which nuts need tightening.
    - Build the location graph from static 'link' facts.
    - Pre-compute shortest path distances between all pairs of *linked* locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify Bob's current location. If unknown, return infinity.
    2. Check if Bob is currently carrying a spanner.
    3. Identify all nuts that are loose in the current state AND are required to be tightened in the goal state. Find their current locations. If a location is unknown, return infinity. Let this set of nuts be `LooseGoalNuts` and their locations `NutLocations`.
    4. If `LooseGoalNuts` is empty, the heuristic is 0 (all required nuts are tightened).
    5. Calculate the cost to acquire a spanner if Bob is not carrying one and needs one (`spanner_cost`):
       - If Bob is carrying a spanner: `spanner_cost = 0`.
       - If Bob is not carrying a spanner: Find the locations of all usable spanners in the current state. Compute the minimum distance from Bob's current location to any usable spanner location. `spanner_cost = min_distance + 1` (for the pick-up action). If no usable spanners are reachable, return infinity.
    6. Determine Bob's effective starting location for the nut-tightening tour (`TourStartLoc`):
       - If `spanner_cost > 0` (meaning Bob had to go get a spanner), `TourStartLoc` is the location where he picks up the nearest spanner.
       - If `spanner_cost == 0` (meaning Bob already had a spanner or didn't need one), `TourStartLoc` is Bob's current location.
    7. Calculate the movement cost (`movement_cost`):
       - Sum the shortest distances from `TourStartLoc` to each location in `NutLocations`. If any nut location is unreachable from `TourStartLoc`, return infinity. This is an overestimate of the actual travel required but is easy to compute.
    8. Calculate the tightening cost (`tighten_cost`):
       - This is simply the number of nuts in `LooseGoalNuts`, as each requires one 'tighten' action. `tighten_cost = len(LooseGoalNuts)`.
    9. The total heuristic value is `spanner_cost + movement_cost + tighten_cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions, e.g., frozenset({'(tightened nut1)', '(tightened nut2)'})
        static_facts = task.static # Static facts, e.g., frozenset({'(link shed location1)', ...})

        # Identify which nuts need to be tightened in the goal
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Build the location graph from static 'link' facts
        self.location_graph = build_location_graph(static_facts)

        # Pre-compute all-pairs shortest paths between all *linked* locations
        self.all_pairs_distances = {}
        linked_locations = set(self.location_graph.keys()) # Get all locations that have at least one link
        for start_loc in linked_locations:
             self.all_pairs_distances[start_loc] = bfs_distances(start_loc, self.location_graph)

    def get_distance(self, loc1, loc2):
        """Get shortest distance between two locations. Returns infinity if unreachable."""
        if loc1 == loc2:
            return 0
        # Check if both locations are part of the linked graph structure we computed distances for
        # and if a path exists between them.
        if loc1 in self.all_pairs_distances and loc2 in self.all_pairs_distances.get(loc1, {}):
             return self.all_pairs_distances[loc1][loc2]
        else:
             # If either location is not in the graph or they are in different
             # disconnected components, they are unreachable from each other
             # via the defined links.
             return float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify Bob's current location.
        bob_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_location = get_parts(fact)[2]
                break
        if bob_location is None:
             # Bob's location must be known in a valid state.
             return float('inf') # Should not happen in practice for valid states

        # 2. Check if Bob is currently carrying a spanner.
        bob_carrying_spanner = False
        # We don't need the specific spanner Bob is carrying for the heuristic logic
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                bob_carrying_spanner = True
                break

        # 3. Identify loose nuts needing tightening and their locations.
        loose_goal_nuts_info = {} # Map nut -> location
        current_nut_locations = {} # Map nut -> location for all nuts in state
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                 # Assuming objects starting with 'nut' are nuts
                 if obj.startswith('nut'):
                     current_nut_locations[obj] = loc

        loose_goal_nuts = set()
        for nut in self.goal_nuts:
             # Check if the nut is loose in the current state
             if f"(loose {nut})" in state:
                  loose_goal_nuts.add(nut)
                  # Store its current location
                  if nut in current_nut_locations:
                      loose_goal_nuts_info[nut] = current_nut_locations[nut]
                  else:
                      # Nut location unknown? Problematic state. Assume unsolvable.
                      return float('inf') # Should not happen in practice for valid states


        # 4. If LooseGoalNuts is empty, the heuristic is 0.
        if not loose_goal_nuts:
            return 0

        # 5. Calculate the cost to acquire a spanner if needed.
        spanner_cost = 0
        tour_start_location = bob_location # Default start location

        if not bob_carrying_spanner:
            # Find locations of usable spanners in the current state
            usable_spanner_locations = set()
            for fact in state:
                if match(fact, "usable", "*"):
                    spanner = get_parts(fact)[1]
                    # Find where this spanner is located in the current state
                    for at_fact in state:
                        if match(at_fact, "at", spanner, "*"):
                            usable_spanner_locations.add(get_parts(at_fact)[2])
                            break # Found location for this spanner

            if not usable_spanner_locations:
                # No usable spanners available in the state. Problem unsolvable.
                return float('inf')

            # Find the minimum distance from Bob's current location to any usable spanner location
            min_dist_to_spanner = float('inf')
            nearest_spanner_loc = None
            for spanner_loc in usable_spanner_locations:
                 dist = self.get_distance(bob_location, spanner_loc)
                 if dist < min_dist_to_spanner:
                     min_dist_to_spanner = dist
                     nearest_spanner_loc = spanner_loc

            if min_dist_to_spanner == float('inf'):
                 # Cannot reach any usable spanner from Bob's current location. Problem unsolvable.
                 return float('inf')

            spanner_cost = min_dist_to_spanner + 1 # +1 for the pick-up action
            tour_start_location = nearest_spanner_loc # Bob starts the nut-tightening tour from where he gets the spanner

        # 6. & 7. Nut locations and TourStartLoc determined above.

        # 7. Calculate the movement cost to visit all nut locations.
        nut_locations = list(loose_goal_nuts_info.values()) # Get list of locations of loose goal nuts
        movement_cost = 0
        if nut_locations:
             # Simple approximation: Sum of distances from tour_start_location to each nut location
             for nut_loc in nut_locations:
                  dist = self.get_distance(tour_start_location, nut_loc)
                  if dist == float('inf'):
                       # Cannot reach a nut location from the tour start location. Problem unsolvable.
                       return float('inf')
                  movement_cost += dist

        # 8. Calculate the tightening cost.
        tighten_cost = len(loose_goal_nuts)

        # 9. Total heuristic value.
        total_cost = spanner_cost + movement_cost + tighten_cost

        return total_cost
