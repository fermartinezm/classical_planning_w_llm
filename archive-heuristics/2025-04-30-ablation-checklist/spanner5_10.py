from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If the Heuristic base class is not provided, you might need a simple placeholder:
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs BFS from a start node to find distances to all reachable nodes.

    Args:
        graph: Adjacency list representation of the graph {node: [neighbors]}.
        start_node: The node to start BFS from.

    Returns:
        A dictionary mapping reachable nodes to their distance from the start_node.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all goal nuts by summing:
    1. The number of loose goal nuts (for tighten actions).
    2. The number of usable spanners needed from the ground (for pickup actions).
    3. The sum of shortest path distances from the man's current location
       to each unique location that contains either a loose goal nut or
       one of the required closest usable spanners on the ground.

    This heuristic is not admissible but aims to guide a greedy search efficiently.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and identifying static info.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        self.nut_locations = {}
        self.locations = set()
        graph = {}
        self.man_name = None

        # Combine initial state and static facts for parsing initial configuration
        initial_facts = set(self.initial_state).union(set(self.static))

        # Identify object types based on predicates in initial facts
        man_candidates = set()
        nut_candidates = set()
        spanner_candidates = set()

        for fact in initial_facts:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]

             if predicate == "carrying" and len(parts) == 3:
                  man_candidates.add(parts[1])
                  spanner_candidates.add(parts[2])
             elif predicate == "tightened" and len(parts) == 2:
                  nut_candidates.add(parts[1])
             elif predicate == "loose" and len(parts) == 2:
                  nut_candidates.add(parts[1])
             elif predicate == "usable" and len(parts) == 2:
                  spanner_candidates.add(parts[1])
             elif predicate == "at" and len(parts) == 3:
                  # Add locations from 'at' facts
                  self.locations.add(parts[2])
             elif predicate == "link" and len(parts) == 3:
                  # Add locations from 'link' facts and build graph
                  l1, l2 = parts[1], parts[2]
                  self.locations.add(l1)
                  self.locations.add(l2)
                  graph.setdefault(l1, []).append(l2)
                  graph.setdefault(l2, []).append(l1)


        # Assuming there is exactly one man based on domain structure
        if len(man_candidates) == 1:
            self.man_name = list(man_candidates)[0]
        # Else: man_name remains None, heuristic will return inf if man location isn't found

        # Parse 'at' facts again to get initial nut locations now that we know nut candidates
        for fact in initial_facts:
             if match(fact, "at", "*", "*"):
                  obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                  if obj in nut_candidates:
                       self.nut_locations[obj] = loc

        # Ensure all locations from graph are in self.locations
        for l in graph:
            self.locations.add(l)
            for neighbor in graph[l]:
                self.locations.add(neighbor)

        # Compute all-pairs shortest paths
        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = bfs(graph, loc)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find man's current location
        man_location = None
        if self.man_name:
            for fact in state:
                if match(fact, "at", self.man_name, "*"):
                    _, _, man_location = get_parts(fact)
                    break

        if man_location is None:
             # Man's location is unknown or man_name was not identified
             return float('inf')

        # Identify loose nuts that are goals
        loose_goal_nuts = {
            nut for nut in self.goal_nuts
            if f"(tightened {nut})" not in state
        }
        num_loose_goals = len(loose_goal_nuts)

        if num_loose_goals == 0:
            return 0  # Goal reached

        # Identify usable spanners carried by the man
        usable_spanners_carried = {
            s for fact in state
            if match(fact, "carrying", self.man_name, s) and f"(usable {s})" in state
        }
        num_carrying_usable = len(usable_spanners_carried)

        # Identify usable spanners on the ground
        usable_spanners_ground = {
             s for fact in state
             if match(fact, "at", s, "*") and f"(usable {s})" in state
        }
        num_ground_usable = len(usable_spanners_ground)

        # Check if enough usable spanners exist in total
        if num_carrying_usable + num_ground_usable < num_loose_goals:
            return float('inf') # Unsolvable

        # Heuristic calculation
        h = 0

        # 1. Cost for tighten actions
        h += num_loose_goals

        # 2. Cost for pickup actions needed
        pickups_needed = max(0, num_loose_goals - num_carrying_usable)
        h += pickups_needed

        # 3. Cost for walks

        # Locations of loose goal nuts
        nut_locations_needed = {self.nut_locations[nut] for nut in loose_goal_nuts if nut in self.nut_locations}
        # If a goal nut's location wasn't found in init, it's an issue with parsing or problem definition.
        # Assuming all goal nuts have a fixed location defined in init/static.

        # Locations of usable spanners on the ground
        spanner_locations_usable_ground = {}
        for fact in state:
             if match(fact, "at", "*", "*"):
                  obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                  if obj in usable_spanners_ground:
                       spanner_locations_usable_ground[obj] = loc

        # Find the locations of the 'pickups_needed' closest usable spanners on the ground
        spanner_locations_needed = set()
        if pickups_needed > 0:
            # Get list of (distance, location) for usable ground spanners
            spanner_dist_locs = []
            for s, loc in spanner_locations_usable_ground.items():
                 dist = self.distances.get(man_location, {}).get(loc, float('inf'))
                 if dist != float('inf'):
                      spanner_dist_locs.append((dist, loc))
                 # If dist is inf, this spanner is unreachable, but we might have others.
                 # We only return inf later if the *needed* ones are unreachable.

            # Sort by distance and take the closest 'pickups_needed' locations
            spanner_dist_locs.sort()
            # Ensure we don't try to take more locations than available reachable spanners
            spanner_locations_needed = {loc for dist, loc in spanner_dist_locs[:min(pickups_needed, len(spanner_dist_locs))]}

            # If we need pickups but couldn't find enough reachable spanner locations, it's unsolvable
            if len(spanner_locations_needed) < pickups_needed:
                 return float('inf')


        # Estimate walk cost as sum of distances from man's current location
        # to each unique required location (nut or needed spanner).
        required_locations = nut_locations_needed.union(spanner_locations_needed)

        for loc in required_locations:
             dist = self.distances.get(man_location, {}).get(loc, float('inf'))
             if dist == float('inf'): return float('inf') # Unreachable required location
             h += dist

        return h
