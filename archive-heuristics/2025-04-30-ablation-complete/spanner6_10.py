from fnmatch import fnmatch
from collections import deque

# Assuming heuristics.heuristic_base exists and provides a Heuristic base class
# If not, a minimal base class like the following would be needed:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

from heuristics.heuristic_base import Heuristic


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
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It counts the required tighten and pickup actions and adds an estimate for the initial travel cost.

    # Assumptions:
    - Nuts are static (their location does not change).
    - Spanners become unusable after one tighten action.
    - The man can carry multiple spanners.
    - The location graph defined by 'link' predicates is static and bidirectional.
    - The problem is solvable (i.e., enough usable spanners exist and locations are reachable). If not solvable from the current state, the heuristic returns infinity.
    - There is exactly one man object in the domain.

    # Heuristic Initialization
    - Build the location graph from 'link' predicates.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify the goal nuts from the task goals.
    - Find the static locations of the goal nuts from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the man's current location.
    2. Identify which nuts are currently loose among the goal nuts and find their locations (nut locations are precomputed as they are static).
    3. Identify which spanners are currently usable (either carried or at a location).
    4. Count the number of loose goal nuts (`N_loose`). If `N_loose` is 0, the goal is reached, return 0.
    5. Count the number of usable spanners the man is currently carrying (`C_carried_usable`).
    6. Calculate the number of additional spanners needed from the ground: `S_needed_from_ground = max(0, N_loose - C_carried_usable)`.
    7. Identify usable spanners currently located on the ground and find their current locations from the state.
    8. Calculate the estimated cost of the "first trip":
       - If `S_needed_from_ground > 0`: The man needs to get a spanner first. Find the usable spanner on the ground that is nearest to the man's current location. Calculate the cost to walk to this spanner's location, plus the cost of the pickup action (1), plus the cost to walk from the spanner's location to the nearest loose nut location. If any required location is unreachable, the cost is infinity.
       - If `S_needed_from_ground == 0`: The man has enough spanners. Calculate the cost to walk from the man's current location to the nearest loose nut location. If the location is unreachable, the cost is infinity.
    9. The total heuristic value is the sum of:
       - `N_loose` (for the tighten actions).
       - `S_needed_from_ground` (for the pickup actions).
       - The estimated cost of the first trip.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Need initial state to find initial nut locations

        # Build the location graph from 'link' predicates
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1) # Links are bidirectional
                locations.add(l1)
                locations.add(l2)

        self.all_locations = list(locations) # Store as list for consistent indexing if needed, though dict lookup is fine

        # Compute all-pairs shortest path distances using BFS
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Identify the goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])

        # Find the static locations of the goal nuts from the initial state
        self.nut_locations = {}
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and parts[1] in self.goal_nuts:
                 self.nut_locations[parts[1]] = parts[2]

        # Identify the man's name (assuming only one man and he is in the initial state)
        self.man_name = None
        for fact in initial_state:
            parts = get_parts(fact)
            # A man is a locatable that is not a nut or spanner
            if parts[0] == 'at' and parts[1] not in self.nut_locations and not parts[1].startswith('spanner'):
                 self.man_name = parts[1]
                 break
        # Fallback: If not found by type assumption, look for 'carrying' predicate in initial state
        if self.man_name is None:
             for fact in initial_state:
                 parts = get_parts(fact)
                 if parts[0] == 'carrying':
                     self.man_name = parts[1]
                     break

        if self.man_name is None:
             # This case indicates a problem with parsing or domain definition
             print("Warning: Could not identify the man object.")


    def _bfs(self, start_node):
        """
        Performs Breadth-First Search to find shortest distances from start_node
        to all other reachable nodes in the location graph.
        """
        distances = {node: float('inf') for node in self.all_locations}
        if start_node not in self.all_locations:
             # Start node is not in the graph, no paths possible
             return distances

        distances[start_node] = 0
        queue = deque([(start_node, 0)]) # Use deque for efficient pop(0)
        visited = {start_node}

        while queue:
            (current_node, dist) = queue.popleft() # Use popleft() for BFS

            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
        return distances

    def get_distance(self, loc1, loc2):
        """
        Returns the precomputed shortest distance between two locations.
        Returns float('inf') if loc1 or loc2 are not in the precomputed distances
        (meaning they are not in the location graph) or if loc2 is unreachable from loc1.
        """
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # This indicates an issue or unreachable location
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify the man's current location.
        man_location = None
        if self.man_name:
            for fact in state:
                if match(fact, "at", self.man_name, "*"):
                    man_location = get_parts(fact)[2]
                    break

        if man_location is None:
             # Man's location unknown in this state, should not happen in valid states
             return float('inf')

        # 2. Identify loose goal nuts and their locations.
        loose_goal_nuts = {n for n in self.goal_nuts if f'(loose {n})' in state}
        # Nut locations are static, retrieved from self.nut_locations

        # 3. Identify usable spanners in the current state.
        usable_spanners_in_state = {s for fact in state if match(fact, "usable", s)}

        # 4. Count loose goal nuts.
        n_loose = len(loose_goal_nuts)
        if n_loose == 0:
            return 0  # Goal reached

        # 5. Count carried usable spanners.
        carried_spanners = {s for fact in state if self.man_name and match(fact, "carrying", self.man_name, s)}
        carried_usable_spanners = usable_spanners_in_state.intersection(carried_spanners)
        c_carried_usable = len(carried_usable_spanners)

        # 6. Calculate additional spanners needed from the ground.
        s_needed_from_ground = max(0, n_loose - c_carried_usable)

        # 7. Identify usable spanners on the ground and their locations.
        available_usable_spanners_on_ground = {
            s for s in usable_spanners_in_state
            if s not in carried_spanners # Check if not carried
            # Check if it's at a location present in our graph
            and any(match(fact, "at", s, loc) for fact in state if loc in self.all_locations)
        }
        available_usable_spanner_locs = {
             loc for s in available_usable_spanners_on_ground
             for fact in state if match(fact, "at", s, loc)
        }

        # 8. Calculate the estimated cost of the "first trip".
        first_trip_cost = 0
        target_nut_locs = {self.nut_locations[n] for n in loose_goal_nuts if n in self.nut_locations}

        if s_needed_from_ground > 0:
            # Man needs to get a spanner first
            if not available_usable_spanner_locs:
                return float('inf') # Needed spanner but none available on ground or reachable

            min_dist_to_spanner = float('inf')
            nearest_spanner_loc = None
            for sloc in available_usable_spanner_locs:
                dist = self.get_distance(man_location, sloc)
                if dist < min_dist_to_spanner:
                    min_dist_to_spanner = dist
                    nearest_spanner_loc = sloc

            if min_dist_to_spanner == float('inf'):
                 return float('inf') # Cannot reach any usable spanner

            # Cost to get spanner: walk + pickup
            first_trip_cost += min_dist_to_spanner + 1

            # Now need to go from spanner location to nearest nut location
            if not target_nut_locs:
                 # Should not happen if n_loose > 0, but defensive check
                 min_dist_spanner_to_nut = 0
            else:
                min_dist_spanner_to_nut = float('inf')
                for nloc in target_nut_locs:
                    dist = self.get_distance(nearest_spanner_loc, nloc)
                    min_dist_spanner_to_nut = min(min_dist_spanner_to_nut, dist)

                if min_dist_spanner_to_nut == float('inf'):
                     return float('inf') # Cannot reach any nut from spanner location

            first_trip_cost += min_dist_spanner_to_nut

        else: # s_needed_from_ground == 0, man has enough spanners
            if not target_nut_locs:
                 # Should not happen if n_loose > 0, but defensive check
                 first_trip_cost = 0
            else:
                min_dist_to_nut = float('inf')
                for nloc in target_nut_locs:
                    dist = self.get_distance(man_location, nloc)
                    min_dist_to_nut = min(min_dist_to_nut, dist)

                if min_dist_to_nut == float('inf'):
                     return float('inf') # Cannot reach any nut

                first_trip_cost = min_dist_to_nut

        # 9. Total heuristic value
        # Ensure costs are finite before summing
        if first_trip_cost == float('inf'):
             return float('inf')

        total_cost = n_loose + s_needed_from_ground + first_trip_cost

        return total_cost
