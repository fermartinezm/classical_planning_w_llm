from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(sjuejzxf p1 l7)".
    - `args`: The expected pattern elements (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to reach a goal state by considering 
    the positions of objects of type du66z5od (e.g., p1, p2) on locations (x6ibpitl), 
    where the goal conditions require these objects to be at certain locations. It accounts 
    for the actions to move objects between locations (using `arsslvyt`) 
    and to transfer objects onto/from hctyyy8k objects (e.g., v1, v2) using the `szsevy61` and `tmoxe3x5` actions.
    The heuristic uses shortest path distances between locations (derived from static facts `miud1izi`) 
    and approximates transfers along connection chains of kyvrixb5 objects (via `hab0vli4` and `do9fmoyo`) 
    to estimate minimal transfers needed.

    Assumptions:
    - Objects of type du66z5od (called "items") are moved across locations (x6ibpitl) attached to hctyyy8k objects (called "carriers").
    - Locations are connected bidirectionally as given by `miud1izi`; these connections form a graph.
    - Kyvrixb5 objects form chains or layers connected by `hab0vli4`.
    - Each du66z5od object is currently on some location attached to a carrier (hctyyy8k).
    - Reaching the goal means placing the du66z5od objects on specified locations.
    - Moving an item between locations requires one or multiple `arsslvyt` actions.
    - Transferring an item between carriers costs additional actions `szsevy61` and `tmoxe3x5`.
    - The heuristic is not guaranteed to be admissible but guides towards minimizing moves and transfers.

    Heuristic Initialization:
    - Extract the graph of locations from static facts `(miud1izi loc1 loc2)`.
    - Compute shortest paths (number of `arsslvyt` moves) between all locations.
    - Extract the carrier chain graph via `(hab0vli4 c_i c_j)` over kyvrixb5 objects to estimate minimal number of transfers.
    - Index initial and goal information: identify required "item @ location" from goal facts `(sjuejzxf ?item ?location)` for du66z5od objects.
    - Prepare static data structures to efficiently query distances and assignments during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each goal item-location pair:
       - Identify current location of that du66z5od item from `(sjuejzxf <item> <loc>)` in current state.
       - Identify current carrier of the item from `(do9fmoyo <carrier> <kyvrixb5>)` and `(un606moq <item> <carrier>)`.
         Note: The carrier controls where the item is attached.
    2. Compute location distance between current and goal locations using shortest path on location graph.
       - This corresponds to the minimum number of `arsslvyt` actions needed to move the item on carriers.
    3. Determine carrier difference between current and goal carrier chains using shortest path on carrier graph.
       - Minimal number of `szsevy61` or `tmoxe3x5` actions to transfer item between carriers. Each transfer counts as 1 action.
    4. Sum location moves and carrier transfers to estimate cost for each item.
    5. The heuristic value is the sum over all items of these estimated costs.
    6. If a goal item is already at its goal location on the correct carrier/kyvrixb5 connection, no cost is added for that item.
    7. The heuristic is zero only if all goal conditions are met, otherwise positive and finite.

    This heuristic balances location moves and carrier transfers to provide a more informed estimate and reduce expanded nodes.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract location adjacency graph from miud1izi (bidirectional)
        self.location_graph = dict()  # loc -> set of neighbor locs
        for fact in self.static:
            if match(fact, "miud1izi", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Precompute all-pairs shortest paths for locations using BFS (only distances)
        self.location_distances = dict()  # loc1 -> dict loc2 -> dist
        for loc in self.location_graph:
            self.location_distances[loc] = self._bfs_distances(loc, self.location_graph)

        # Extract carrier adjacency graph from hab0vli4 (bidirectional)
        self.carrier_graph = dict()  # kyvrixb5 -> set of neighbor kyvrixb5
        for fact in self.static:
            if match(fact, "hab0vli4", "*", "*"):
                _, c1, c2 = get_parts(fact)
                self.carrier_graph.setdefault(c1, set()).add(c2)
                self.carrier_graph.setdefault(c2, set()).add(c1)

        # No need to precompute all pairs carrier distance, will do BFS on demand and cache
        self._carrier_distance_cache = dict()

        # Extract goal item-location pairs for du66z5od items
        self.goal_item_locations = dict()
        for g in self.goals:
            if match(g, "sjuejzxf", "*", "*"):
                _, item, goal_loc = get_parts(g)
                self.goal_item_locations[item] = goal_loc

    def _bfs_distances(self, start, graph):
        """Compute shortest path distances from start to all reachable nodes in graph."""
        from collections import deque

        distances = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        return distances

    def _carrier_distance(self, start, goal):
        """Compute shortest path distance between two carriers, using BFS and caching results."""
        if start == goal:
            return 0
        key = (start, goal)
        if key in self._carrier_distance_cache:
            return self._carrier_distance_cache[key]

        from collections import deque

        visited = set([start])
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            for nbr in self.carrier_graph.get(current, []):
                if nbr == goal:
                    self._carrier_distance_cache[key] = dist + 1
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        # If no path found, consider infinite (but domain should be connected)
        self._carrier_distance_cache[key] = float('inf')
        return float('inf')

    def __call__(self, node):
        state = node.state

        # If state is goal, heuristic is 0
        if self.goals <= state:
            return 0

        # Index current positions of items: map item -> location (sjuejzxf), item -> carrier (un606moq)
        item_location = {}
        item_carrier = {}

        # carrier -> kyvrixb5 presence mapping from (do9fmoyo hctyyy8k kyvrixb5)
        carrier_kyvrixb5 = {}
        for fact in state:
            if match(fact, "do9fmoyo", "*", "*"):
                _, carrier, kyv = get_parts(fact)
                carrier_kyvrixb5[carrier] = kyv

        # item location on a location
        for fact in state:
            if match(fact, "sjuejzxf", "*", "*"):
                _, item, loc = get_parts(fact)
                # Only track du66z5od objects (items) for heuristic
                # We can assume items are du66z5od by naming convention or by goal keys
                if item in self.goal_item_locations:
                    item_location[item] = loc

        # item carried by a carrier (un606moq ?item ?carrier)
        for fact in state:
            if match(fact, "un606moq", "*", "*"):
                _, item, carrier = get_parts(fact)
                if item in self.goal_item_locations:
                    item_carrier[item] = carrier

        total_estimated_cost = 0

        # For each goal item, estimate cost from current state to goal
        for item, goal_loc in self.goal_item_locations.items():

            # Current location of the item
            curr_loc = item_location.get(item)
            if curr_loc is None:
                # If unknown, pessimistically estimate some distance
                # A reasonable guess: distance is max possible distance between locations
                max_loc_dist = 0
                for d in self.location_distances.values():
                    if d:
                        max_loc_dist = max(max_loc_dist, max(d.values()))
                dist_loc = max_loc_dist
            else:
                dist_loc = self.location_distances.get(curr_loc, {}).get(goal_loc, float('inf'))

            # Current carrier of the item and its kyvrixb5
            curr_carrier = item_carrier.get(item)
            if curr_carrier is None or curr_carrier not in carrier_kyvrixb5:
                # Unknown carrier or kyvrixb5, pessimistic guess: some max distance on carrier graph
                # Use max possible distance on carrier graph (diameter upper bound)
                if not self.carrier_graph:
                    dist_carrier = 0
                else:
                    # Compute max carrier distance lazily once
                    if not hasattr(self, "_max_carrier_dist"):
                        max_dist = 0
                        carriers = list(self.carrier_graph.keys())
                        for c1 in carriers:
                            dists = self._bfs_distances(c1, self.carrier_graph)
                            max_dist = max(max_dist, max(dists.values()) if dists else 0)
                        self._max_carrier_dist = max_dist
                    dist_carrier = self._max_carrier_dist
            else:
                curr_kyv = carrier_kyvrixb5[curr_carrier]
                # Goal carrier: find carrier that has kyvrixb5 corresponding to goal location's carrier
                # Since goal constraints do not specify carrier, assume same carrier as current or closest
                # For heuristic, assume goal kyvrixb5 is the kyvrixb5 of the carrier corresponding to goal location
                # Scan carriers that have kyvrixb5 reachable from current kyvrixb5 by shortest path and pick minimal cost

                # Build a map: carrier -> kyvrixb5 for all carriers that appear in state:
                # We approximate the goal carrier kyvrixb5 by clustering the carriers that hold hctyyy8k objects at goal location (or close)
                # However, since goal does not specify carrier, we assume minimal transfer distance is 0 (ideal)
                # To be more informed, we could take the minimal distance to any carrier that appears in state on goal location

                # Collect carriers currently holding any hctyyy8k object at goal location
                candidate_kyvrixb5s = set()
                for fact in state:
                    if match(fact, "sjuejzxf", "*", goal_loc):
                        carrier_obj = get_parts(fact)[1]
                        carrier_of_obj = None
                        for f in state:
                            if match(f, "do9fmoyo", carrier_obj, "*"):
                                carrier_of_obj = get_parts(f)[2]
                                break
                        if carrier_of_obj and carrier_of_obj in carrier_kyvrixb5:
                            candidate_kyvrixb5s.add(carrier_kyvrixb5[carrier_of_obj])

                if not candidate_kyvrixb5s:
                    # No known candidate carrier kyvrixb5 at goal - assume zero transfer needed
                    dist_carrier = 0
                else:
                    # Compute minimal carrier transfers required from curr_kyv to any of goal candidate kyvrixb5s
                    dist_carrier = min(self._carrier_distance(curr_kyv, goal_kyv) for goal_kyv in candidate_kyvrixb5s)

            # Each location step corresponds roughly to one arsslvyt action
            cost_location_moves = dist_loc

            # Each carrier transfer corresponds approximately to one szsevy61 or tmoxe3x5 action
            cost_carrier_transfers = dist_carrier

            # Total cost estimate for this item:
            total_estimated_cost += cost_location_moves + cost_carrier_transfers

        return total_estimated_cost
