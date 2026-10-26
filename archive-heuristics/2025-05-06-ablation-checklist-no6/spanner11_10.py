from heuristics.heuristic_base import Heuristic
from collections import deque
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    This heuristic estimates the cost to tighten all required nuts by simulating
    a greedy strategy: the man alternates between picking up the closest available
    usable spanner and moving to the closest remaining loose goal nut, performing
    the necessary actions (walk, pickup, tighten).

    The heuristic precomputes shortest path distances between all locations.
    It returns infinity if the problem is unsolvable (not enough usable spanners).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and precomputing
        shortest path distances between locations.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Precompute location graph and shortest paths
        self.location_graph = {}
        all_locations = set()

        # Collect all locations mentioned in links, initial state, and goals
        all_facts = task.initial_state | task.goals | self.static_facts
        for fact in all_facts:
             if match(fact, "link", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                    _, loc1, loc2 = parts[1], parts[2]
                    all_locations.add(loc1)
                    all_locations.add(loc2)
             elif match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                    _, obj, loc = parts[1], parts[2]
                    all_locations.add(loc)

        # Build graph adjacency list from link facts
        self.location_graph = {loc: [] for loc in all_locations}
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, loc1, loc2 = parts[1], parts[2]
                    # Ensure locations are in our collected set (should be)
                    if loc1 in self.location_graph and loc2 in self.location_graph:
                         self.location_graph[loc1].append(loc2)
                         self.location_graph[loc2].append(loc1)

        # Compute all-pairs shortest paths using BFS
        self.all_pairs_dist = self._compute_all_pairs_shortest_paths(self.location_graph)

        # Store goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                parts = get_parts(goal)
                if len(parts) > 1:
                    _, nut = parts[1]
                    self.goal_nuts.add(nut)

    def _bfs(self, graph, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in graph}
        if start_node not in graph:
             # Start node is not in the graph keys (e.g., an isolated location)
             # If it's a known location, distance to itself is 0.
             if start_node in distances:
                 distances[start_node] = 0
             return distances # Cannot reach other nodes

        distances[start_node] = 0
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            # current_node is guaranteed to be in graph keys here
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def _compute_all_pairs_shortest_paths(self, graph):
        """Computes shortest path distances between all pairs of locations."""
        all_pairs_dist = {}
        # Iterate over all known locations, which are the keys in the graph dictionary
        for start_node in graph.keys():
            all_pairs_dist[start_node] = self._bfs(graph, start_node)
        return all_pairs_dist

    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance, returns infinity if unreachable or unknown."""
        if loc1 in self.all_pairs_dist and loc2 in self.all_pairs_dist[loc1]:
            return self.all_pairs_dist[loc1][loc2]
        return float('inf')

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.

        Estimates the cost by simulating a greedy sequence of actions:
        - If the man has a spanner, go to the closest loose goal nut and tighten it.
        - If the man does not have a spanner, go to the closest available usable spanner and pick it up.
        """
        state = node.state

        # 1. Extract relevant state information
        man_loc = None
        man_carrying_spanner = None
        man_carrying_usable_spanner = False
        spanners_on_ground = {} # {spanner_obj: location}
        usable_spanners_on_ground = {} # {spanner_obj: location}
        nuts_status = {} # {nut_obj: status (loose/tightened)}
        nuts_location = {} # {nut_obj: location}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # Assuming object types based on name patterns
                    if 'man' in obj:
                         man_loc = loc
                    elif 'spanner' in obj:
                         spanners_on_ground[obj] = loc
                    elif 'nut' in obj:
                         nuts_location[obj] = loc
            elif predicate == "carrying":
                 if len(parts) == 3:
                    m, s = parts[1], parts[2]
                    if 'man' in m and 'spanner' in s:
                        man_carrying_spanner = s
            elif predicate == "usable":
                 if len(parts) == 2:
                    s = parts[1]
                    if 'spanner' in s:
                        # Check if this usable spanner is the one the man is carrying
                        if man_carrying_spanner == s:
                            man_carrying_usable_spanner = True
                        # Check if this usable spanner is on the ground
                        if s in spanners_on_ground:
                            usable_spanners_on_ground[s] = spanners_on_ground[s]
            elif predicate == "loose":
                 if len(parts) == 2:
                    n = parts[1]
                    if 'nut' in n:
                        nuts_status[n] = 'loose'
            elif predicate == "tightened":
                 if len(parts) == 2:
                    n = parts[1]
                    if 'nut' in n:
                        nuts_status[n] = 'tightened'

        # 2. Identify loose goal nuts
        loose_goal_nuts = [n for n in self.goal_nuts if nuts_status.get(n) == 'loose']

        # 3. Check if goal is reached
        if not loose_goal_nuts:
            return 0

        # 4. Check solvability (enough usable spanners in total)
        total_usable_spanners_count = 0
        for fact in state:
             if match(fact, "usable", "*"):
                 total_usable_spanners_count += 1

        if total_usable_spanners_count < len(loose_goal_nuts):
             return float('inf') # Not enough usable spanners exist to tighten all required nuts

        # Handle case where man_loc is unknown (should not happen in valid states)
        if man_loc is None:
             return float('inf')

        # 5. Run greedy simulation
        h = 0
        curr_loc = man_loc
        nuts_remaining = set(loose_goal_nuts)
        # Available spanners are those on the ground that are usable
        spanners_available = {s: l for s, l in usable_spanners_on_ground.items()}
        man_has_spanner = man_carrying_usable_spanner

        while nuts_remaining:
            if man_has_spanner:
                # Man has spanner, needs to go to a nut
                min_dist = float('inf')
                next_nut = None
                next_nut_loc = None

                # Find the closest remaining loose goal nut
                for nut in nuts_remaining:
                    loc_n = nuts_location.get(nut)
                    if loc_n is None:
                         # Location of a required nut is unknown, unsolvable
                         return float('inf')
                    dist = self.get_distance(curr_loc, loc_n)
                    if dist < min_dist:
                        min_dist = dist
                        next_nut = nut
                        next_nut_loc = loc_n

                if next_nut is None or min_dist == float('inf'):
                     # Cannot reach any remaining nut location
                     return float('inf')

                h += min_dist # Walk to nut
                curr_loc = next_nut_loc
                h += 1 # Tighten nut
                nuts_remaining.remove(next_nut)
                man_has_spanner = False # Spanner is used

            else:
                # Man needs a spanner, go pick one up
                min_dist = float('inf')
                next_spanner_obj = None
                next_spanner_loc = None

                # Find the closest available usable spanner location
                for s_obj, s_loc in spanners_available.items():
                    dist = self.get_distance(curr_loc, s_loc)
                    if dist < min_dist:
                        min_dist = dist
                        next_spanner_obj = s_obj
                        next_spanner_loc = s_loc

                if next_spanner_obj is None or min_dist == float('inf'):
                    # No more usable spanners available or reachable
                    return float('inf')

                h += min_dist # Walk to spanner
                curr_loc = next_spanner_loc
                h += 1 # Pickup spanner
                del spanners_available[next_spanner_obj] # Spanner is now carried/used
                man_has_spanner = True

        return h
