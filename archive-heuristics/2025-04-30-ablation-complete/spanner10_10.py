from fnmatch import fnmatch
from collections import deque

# Assume Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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

    Summary
    This heuristic estimates the number of actions (walk, pickup_spanner, tighten_nut)
    required to tighten all loose nuts. It uses a greedy simulation approach,
    prioritizing the closest required action (tightening a nut if a usable spanner
    is carried, or picking up a spanner if not). Shortest path distances between
    locations are precomputed using BFS.

    Assumptions
    - There is only one man. The heuristic attempts to identify the man object
      from the state facts (e.g., involved in 'carrying').
    - The goal is to tighten all nuts that are initially loose.
    - Spanners, once used for tightening, become unusable.
    - The graph of locations connected by 'link' predicates is static and connected
      (or at least all relevant locations are in the same connected component).
    - A man can carry multiple spanners.

    Heuristic Initialization
    - Build the graph of locations based on 'link' predicates found in static facts.
    - Compute all-pairs shortest paths between locations using BFS. Store these
      distances in a dictionary `self.dist`.

    Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, return 0.
    2. Identify the man object's name by looking for facts like '(carrying ?m ?s)'.
    3. Identify the man's current location from '(at ?m ?l)' fact.
    4. Identify all nuts that are currently 'loose' and determine their fixed locations
       from '(at ?n ?l)' facts in the state.
    5. Identify all spanners the man is currently 'carrying' from '(carrying ?m ?s)' facts.
    6. Identify all spanners that are currently 'usable' from '(usable ?s)' facts.
    7. Determine the set of usable spanners the man is carrying.
    8. Determine the set of usable spanners that are currently on the ground (not carried)
       and their locations from '(at ?s ?l)' facts.
    9. Count the number of loose nuts (k), carried usable spanners (c), and usable ground spanners (g).
    10. If k > c + g, the problem is likely unsolvable in this domain (spanners don't become usable again),
        return infinity (`float('inf')`).
    11. Initialize heuristic value h = 0.
    12. Create mutable sets of locations for loose nuts that still need tightening and
        locations of usable ground spanners that can be picked up.
    13. Start a loop that continues as long as there are loose nut locations remaining:
        a. If the man is carrying at least one usable spanner (c > 0):
           - Find the loose nut location L_n from the remaining set that is closest to the man's current location using precomputed distances.
           - Add the distance from the man's location to L_n to h (cost of walking).
           - Update the man's current location to L_n.
           - Add 1 to h (cost of tighten_nut action).
           - Decrement the count of carried usable spanners (c).
           - Remove L_n from the set of remaining loose nut locations.
        b. If the man is not carrying any usable spanner (c == 0):
           - Find the usable ground spanner location L_s from the available set that is closest to the man's current location using precomputed distances.
           - Add the distance from the man's location to L_s to h (cost of walking).
           - Update the man's current location to L_s.
           - Add 1 to h (cost of pickup_spanner action).
           - Increment the count of carried usable spanners (c).
           - Remove L_s from the set of available usable ground spanner locations.
    14. Return the final heuristic value h.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and computing
        all-pairs shortest paths.
        """
        self.task = task
        self.goals = task.goals # Keep goals to check if state is goal state

        # Build the location graph from 'link' facts
        self.location_graph = {}
        locations = set()
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        self.locations = list(locations) # Store list of all locations

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """Perform BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.location_graph: # Check if node has neighbors
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Get the precomputed shortest distance between two locations."""
        if loc1 not in self.dist or loc2 not in self.dist[loc1]:
             # This handles cases where a location might not be in the graph (e.g., if static facts are incomplete)
             # or if loc2 is unreachable from loc1.
             return float('inf')
        return self.dist[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        # Identify objects and their properties from the state
        man_name = None
        nuts = set()
        spanners = set()
        nut_locations = {}
        spanner_locations = {}
        carried_spanners = set()
        usable_spanners = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at":
                obj, loc = parts[1:]
                # We need to differentiate object types.
                # A robust way is to parse PDDL types, but given the examples,
                # we can infer based on predicates they appear in or naming conventions.
                # Let's find the man by the 'carrying' predicate.
                # Nuts are involved in 'loose'/'tightened'. Spanners in 'usable'.
                if obj.startswith('nut'): # Assume nuts start with 'nut'
                    nuts.add(obj)
                    nut_locations[obj] = loc
                elif obj.startswith('spanner'): # Assume spanners start with 'spanner'
                    spanners.add(obj)
                    spanner_locations[obj] = loc
                # Man's location is found later once man_name is known

            elif predicate == "loose":
                nuts.add(parts[1])
            elif predicate == "tightened":
                nuts.add(parts[1])
            elif predicate == "usable":
                spanners.add(parts[1])
                usable_spanners.add(parts[1])
            elif predicate == "carrying":
                # The first argument is the man, second is the spanner
                man_name = parts[1] # Identify man name
                spanners.add(parts[2])
                carried_spanners.add(parts[2])

        # 2. Identify man's current location
        man_loc = None
        if man_name:
            for fact in state:
                if match(fact, "at", man_name, "*"):
                    man_loc = get_parts(fact)[2]
                    break

        if man_loc is None:
             # Man's location not found? Should not happen in valid states.
             return float('inf') # Problem state

        # 4. Identify loose nuts and their locations
        loose_nuts = {n for n in nuts if f'(loose {n})' in state}
        loose_nuts_locs = {nut_locations[n] for n in loose_nuts if n in nut_locations}

        # 7. Determine carried usable spanners
        carried_usable_spanners = carried_spanners.intersection(usable_spanners)
        c = len(carried_usable_spanners)

        # 8. Identify usable ground spanners and their locations
        # Ground spanners are spanners that are usable and not carried.
        ground_usable_spanners_locs = {
            spanner_locations[s] for s in spanners
            if s in usable_spanners and s not in carried_spanners and s in spanner_locations
        }

        # 9. Count k, c, g
        k = len(loose_nuts)
        g_locs = set(ground_usable_spanners_locs) # Mutable copy

        # 10. Check solvability
        if c + len(g_locs) < k:
             return float('inf') # Unsolvable

        # 11. Initialize heuristic value
        h = 0

        # 12. Mutable sets for simulation
        remaining_loose_nut_locs = set(loose_nuts_locs)
        available_ground_spanner_locs = set(g_locs)

        # 13. Greedy simulation loop
        current_man_loc = man_loc
        current_carried_usable = c

        while remaining_loose_nut_locs:
            if current_carried_usable > 0:
                # Man has spanner, go to closest loose nut location
                closest_nut_loc = None
                min_dist = float('inf')
                for loc in remaining_loose_nut_locs:
                    dist = self.get_distance(current_man_loc, loc)
                    if dist < min_dist:
                        min_dist = dist
                        closest_nut_loc = loc

                if closest_nut_loc is None or min_dist == float('inf'):
                     # Should not happen if solvable and graph is connected, but handle defensively
                     return float('inf') # Cannot reach any remaining nut

                h += min_dist # Walk cost
                current_man_loc = closest_nut_loc # Update man's location
                h += 1 # Tighten cost
                current_carried_usable -= 1 # Spanner used
                remaining_loose_nut_locs.remove(closest_nut_loc) # Nut tightened

            else:
                # Man needs spanner, go to closest usable ground spanner location
                closest_spanner_loc = None
                min_dist = float('inf')
                for loc in available_ground_spanner_locs:
                    dist = self.get_distance(current_man_loc, loc)
                    if dist < min_dist:
                        min_dist = dist
                        closest_spanner_loc = loc

                if closest_spanner_loc is None or min_dist == float('inf'):
                     # Should not happen if solvable and spanners exist, but handle defensively
                     return float('inf') # Cannot reach any usable ground spanner

                h += min_dist # Walk cost
                current_man_loc = closest_spanner_loc # Update man's location
                h += 1 # Pickup cost
                current_carried_usable += 1 # Now carrying one spanner
                available_ground_spanner_locs.remove(closest_spanner_loc) # Spanner picked up

        # 14. Return heuristic value
        return h

