from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove surrounding parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the number of nuts remaining to be tightened, the cost to acquire
    a spanner if Bob is not carrying one, and the cost to move Bob (with or without
    a spanner) to the first location where a goal nut can be tightened or a spanner
    can be acquired.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Bob needs at least one spanner to tighten a nut.
    - Links between locations are bidirectional.
    - Usable spanners are available somewhere if Bob needs one (otherwise, the state is likely unsolvable).
    - The heuristic estimates the cost to reach the *first* useful location (spanner or nut) and doesn't explicitly model the path visiting all nuts, making it non-admissible.

    # Heuristic Initialization
    - Extract the set of nuts that need to be tightened from the goal conditions.
    - Build a graph representing the locations and the links between them.
    - Precompute shortest path distances between all pairs of locations using BFS. This allows efficient lookup of movement costs during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal nuts that are currently `loose`. If none are loose, the heuristic is 0.
    2. If there are loose goal nuts, the base heuristic value is the count of these loose nuts (representing the minimum number of `tighten` actions required).
    3. Determine Bob's current location.
    4. Check if Bob is currently `carrying` any spanner.
    5. Find the locations of all `usable` spanners in the current state.
    6. Find the locations of all `loose` goal nuts in the current state.
    7. Calculate the minimum cost to get Bob to the *first* location where he can perform a useful action (either picking up a spanner if needed, or tightening a nut).
       - If Bob is already carrying a spanner: The cost is the shortest distance from Bob's current location to the nearest location containing a loose goal nut.
       - If Bob is *not* carrying a spanner: He must first acquire one. The cost is estimated as the minimum cost over all usable spanner locations (`LocS`) and all loose goal nut locations (`LocN`) of: (shortest distance from Bob's current location to `LocS`) + 1 (for the `pickup` action) + (shortest distance from `LocS` to `LocN`). This estimates the cost to get a spanner and bring it to the first nut location.
    8. If the calculated minimum cost to reach the first useful location is infinite (meaning no path exists to any spanner or any loose nut), the state is likely unsolvable, and the heuristic returns infinity.
    9. Add the calculated minimum cost from step 7 to the base heuristic value from step 2.
    10. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal nuts
        self.goal_nuts = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == "tightened" and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # Build location graph from static links and collect all mentioned locations
        self.location_graph = {}
        all_locations = set()

        # Add locations from link facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Add locations from initial state and goal state 'at' predicates
        for fact in task.initial_state | self.goals:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                  obj, loc = parts[1], parts[2]
                  all_locations.add(loc)

        # Compute all-pairs shortest paths using BFS from each location
        self.distances = {}
        for start_loc in all_locations:
             self.distances[start_loc] = self._bfs(start_loc, all_locations)


    def _bfs(self, start_loc, all_locations):
        """Performs BFS from start_loc to find distances to all other locations."""
        dist = {loc: float('inf') for loc in all_locations}
        # Handle case where start_loc is not in the graph (isolated)
        if start_loc not in all_locations:
             return dist # All distances remain infinity

        dist[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            u = queue.popleft()

            # If u is not in the graph (e.g., an isolated location only mentioned in 'at'), skip its neighbors
            # It should be in all_locations, but might not have any links in location_graph
            if u not in self.location_graph:
                 continue

            for v in self.location_graph[u]:
                if dist[v] == float('inf'):
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse state to find relevant facts efficiently
        bob_location = None
        bob_carrying_spanner = False
        usable_spanners = set() # Names of usable spanners
        object_locations = {} # {object_name: location}
        loose_nuts_in_state = set() # Names of loose nuts

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                object_locations[obj] = loc
                if obj == "bob":
                    bob_location = loc
            elif parts[0] == "carrying" and len(parts) == 2 and parts[1] == "bob":
                 # Check if Bob is carrying *any* object
                 bob_carrying_spanner = True # Assuming the only thing Bob carries is a spanner
            elif parts[0] == "usable" and len(parts) == 2:
                 usable_spanners.add(parts[1])
            elif parts[0] == "loose" and len(parts) == 2:
                 loose_nuts_in_state.add(parts[1])


        # 1. Identify loose goal nuts and their locations
        loose_goal_nuts = loose_nuts_in_state.intersection(self.goal_nuts)
        loose_nut_locations = {nut: object_locations.get(nut) for nut in loose_goal_nuts if nut in object_locations}
        unique_loose_nut_locations = set(loose_nut_locations.values())

        # Locations of usable spanners
        usable_spanner_locations = {object_locations.get(spanner) for spanner in usable_spanners if spanner in object_locations}
        usable_spanner_locations.discard(None) # Remove None if any spanner location wasn't found


        # 2. Base heuristic: number of loose goal nuts
        h = len(loose_goal_nuts)

        # If no loose goal nuts, goal is reached for these nuts
        if h == 0:
            return 0

        # Handle case where Bob's location is unknown (shouldn't happen in valid states)
        if bob_location is None:
             # This state is likely invalid or unsolvable
             return float('inf')

        # 7. Calculate minimum cost to reach the first useful location with a spanner
        min_cost_to_reach_first_nut_with_spanner = float('inf')

        # Ensure Bob's current location is in the distance map keys
        if bob_location not in self.distances:
             # Bob is in an isolated location not connected to the graph
             return float('inf')


        if bob_carrying_spanner:
            # Bob has a spanner, just needs to reach the nearest loose nut location
            if unique_loose_nut_locations:
                 min_dist_to_nut = float('inf')
                 for nut_loc in unique_loose_nut_locations:
                      # Ensure nut_loc is a valid key in distances from bob_location
                      if nut_loc in self.distances[bob_location]:
                           min_dist_to_nut = min(min_dist_to_nut, self.distances[bob_location][nut_loc])

                 min_cost_to_reach_first_nut_with_spanner = min_dist_to_nut
            # else: No loose nuts, but h > 0? This case should not happen if h = len(loose_goal_nuts) > 0

        else: # Bob needs a spanner
            # He needs to go to a spanner, pick it up, then go to a nut
            # Estimate cost as: dist(Bob->SpannerLoc) + 1 (pickup) + dist(SpannerLoc->NutLoc)
            # Minimize over all usable spanner locations and all loose nut locations
            if not usable_spanner_locations:
                 # Cannot get a spanner, likely unsolvable
                 return float('inf')

            if not unique_loose_nut_locations:
                 # No loose nuts, but h > 0? This case should not happen if h = len(loose_goal_nuts) > 0
                 pass # Should not reach here if h > 0 and unique_loose_nut_locations is empty

            for spanner_loc in usable_spanner_locations:
                 # Ensure spanner_loc is a valid key in distances
                 if spanner_loc not in self.distances:
                      continue # Cannot reach from/to this spanner location

                 # Distance from Bob to spanner location
                 dist_B_S = self.distances[bob_location].get(spanner_loc, float('inf'))

                 if dist_B_S == float('inf'):
                      continue # Cannot reach this spanner

                 # Find minimum distance from this spanner location to any loose nut location
                 min_dist_S_N = float('inf')
                 for nut_loc in unique_loose_nut_locations:
                      # Ensure nut_loc is a valid key in distances from spanner_loc
                      if spanner_loc in self.distances and nut_loc in self.distances[spanner_loc]:
                           min_dist_S_N = min(min_dist_S_N, self.distances[spanner_loc][nut_loc])

                 if min_dist_S_N != float('inf'):
                      # Cost to go Bob->S, pickup, S->N
                      cost = dist_B_S + 1 + min_dist_S_N
                      min_cost_to_reach_first_nut_with_spanner = min(min_cost_to_reach_first_nut_with_spanner, cost)


        # 8. Handle unsolvable case (cannot reach any spanner or nut)
        if min_cost_to_reach_first_nut_with_spanner == float('inf'):
             # This can happen if Bob is isolated, or all spanners/nuts are isolated,
             # or no usable spanners exist when needed and Bob doesn't have one,
             # or no path exists between Bob/spanners and nuts.
             return float('inf')


        # 9. Add movement/spanner cost to base cost
        h += min_cost_to_reach_first_nut_with_spanner

        # 10. Return total heuristic value
        return h
