from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(sjuejzxf p1 l7)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal
    that places certain du66z5od objects (e.g., packages p1, p2, ...) on specific x6ibpitl locations.
    The domain models objects that can be at locations, with a notion of connectivity between locations,
    and involves moving du66z5od objects between locations possibly involving hctyyy8k entities carrying kyvrixb5 "containers".

    Assumptions
    -----------
    - The goal consists exclusively of positive (sjuejzxf ?object ?location) facts, stating where each du66z5od object must be.
    - Objects of type du66z5od ("packages") and hctyyy8k ("carriers") reside at (sjuejzxf ...) facts for locations of type x6ibpitl.
    - Connectivity between locations is expressed with (miud1izi ?loc1 ?loc2), a symmetric adjacency relation.
    - Transfer actions involve either directly moving the du66z5od objects between locations or switching their associated kyvrixb5 containers.
    - The number of (do9fmoyo ?carrier ?container) facts tracks carriers' current kyvrixb5 container and (hab0vli4 ?container1 ?container2) represents some graph of containers.
    - The heuristic prioritizes approximating minimal moves needed to move each du66z5od object from its current location towards its goal location along the location graph.
    - Transfers of kyvrixb5 containers by hctyyy8k carriers are approximated by shortest distances in the container graph.
    - The heuristic ignores complex interactions between carriers and containers in detail for efficiency, estimating a sum over all du66z5od objects of the minimal cost to reach the goal location and container state.

    Heuristic Initialization
    ------------------------
    - Extract static adjacency of locations from (miud1izi ?loc1 ?loc2).
    - Build undirected location graph for shortest path computations between locations.
    - Extract container adjacency from (hab0vli4 ?container1 ?container2).
    - Build undirected container graph for shortest path computations between containers.
    - Extract goal assignments of du66z5od objects to final locations from task goals (expected as (sjuejzxf ?obj ?loc)).
    - Extract all carriers and their current container assignments from state facts of the form (do9fmoyo ?carrier ?container).
    - Precompute shortest path distances on location and container graphs using BFS for efficiency.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state satisfies goals exactly, heuristic = 0.

    2. For each du66z5od object with a goal location:
       a. Identify its current location by checking (sjuejzxf ?obj ?loc).
       b. Compute the shortest path distance between current location and goal location on the location graph.
          This represents the minimal moves of type arsslvyt needed to "walk" the object towards the goal location.
       c. Consider that du66z5od objects may be associated with a carrier and a kyvrixb5 container (either directly or indirectly).
          For approximation, find the carrier currently holding that du66z5od object by looking for (un606moq ?obj ?carrier).
          Then find carrier's current container (do9fmoyo ?carrier ?container).
          The goal container for the carrier is not explicitly given, but containers appear to be moved only during certain actions.
          To approximate, assume minimal container moves = 0 (optimistic) or could be integrated if relevant.

    3. The heuristic sums up, for every du66z5od object, the minimal number of moves needed for its location to reach the goal location along the location graph.
       This is a simple and efficiently computable relaxation of the actual planning cost,
       neglecting the more complex container and carrier manipulations but capturing essential spatial moves.

    4. The heuristic value is the sum of all these individual distances, representing an estimate on how many arsslvyt actions at minimum are needed to relocate all relevant objects.

    5. This heuristic is not admissible but gives a good guiding estimate in greedy best-first search and is efficiently computable.

    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Parse location adjacency graph from (miud1izi ?loc1 ?loc2)
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "miud1izi", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Parse container adjacency graph from (hab0vli4 ?container1 ?container2)
        self.container_graph = {}
        for fact in self.static:
            if match(fact, "hab0vli4", "*", "*"):
                _, c1, c2 = get_parts(fact)
                self.container_graph.setdefault(c1, set()).add(c2)
                self.container_graph.setdefault(c2, set()).add(c1)

        # Extract goal positions for each du66z5od object from goals, which are (sjuejzxf ?obj ?loc)
        self.goal_positions = {}
        for goal in self.goals:
            if match(goal, "sjuejzxf", "*", "*"):
                _, obj, loc = get_parts(goal)
                self.goal_positions[obj] = loc

        # Precompute shortest paths for locations
        self.location_distance_cache = {}
        # For efficiency, we will compute on demand in __call__

        # Precompute shortest paths for containers (not currently used in heuristic but computed for possible extensions)
        self.container_distance_cache = {}

    def shortest_distance_graph(self, graph, start, goal, cache):
        """
        Compute shortest distance between start and goal nodes in an undirected graph using BFS.
        Uses caching to avoid repeated computations.
        """
        if (start, goal) in cache:
            return cache[(start, goal)]
        if start == goal:
            cache[(start, goal)] = 0
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            neighbors = graph.get(current, [])
            for nbr in neighbors:
                if nbr == goal:
                    cache[(start, goal)] = dist + 1
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        # No path found
        cache[(start, goal)] = float('inf')
        return float('inf')

    def __call__(self, node):
        state = node.state

        # If goals are satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Map du66z5od objects to current locations by checking (sjuejzxf ?obj ?loc)
        obj_locations = {}
        for fact in state:
            if match(fact, "sjuejzxf", "*", "*"):
                _, obj, loc = get_parts(fact)
                if obj in self.goal_positions:
                    obj_locations[obj] = loc

        # Compute heuristic cost as sum of shortest path distances for each object from current to goal location
        total_cost = 0
        for obj, goal_loc in self.goal_positions.items():
            current_loc = obj_locations.get(obj)
            if current_loc is None:
                # If object location unknown, pessimistically ignore it (could also add a penalty)
                continue

            dist = self.shortest_distance_graph(
                self.location_graph, current_loc, goal_loc, self.location_distance_cache
            )
            # If distance is infinite, then goal unreachable from current location - heuristic infinity
            if dist == float('inf'):
                # Return large number rather than infinity for practical search
                return 999999

            total_cost += dist

        return total_cost
