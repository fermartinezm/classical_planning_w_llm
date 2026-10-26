from fnmatch import fnmatch
from collections import deque
import heapq
import math

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If running standalone, uncomment the following mock class
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

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

def bfs(start_loc, links):
    """Computes shortest path distances from start_loc to all reachable locations."""
    distances = {start_loc: 0}
    queue = deque([start_loc])
    
    # Convert links to adjacency list for faster lookup
    adj = {}
    for l1, l2 in links:
        adj.setdefault(l1, set()).add(l2)
        adj.setdefault(l2, set()).add(l1) # Assuming links are bidirectional

    while queue:
        current_loc = queue.popleft()
        if current_loc in adj:
            for neighbor in adj[current_loc]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)
    return distances

def mst_cost(nodes, distances):
    """
    Computes the cost of the Minimum Spanning Tree for a set of nodes
    given pairwise shortest path distances. Uses Prim's algorithm.

    Args:
        nodes: A list or set of node identifiers (locations).
        distances: A dictionary where distances[u][v] is the shortest path
                   distance between node u and node v. Assumes distances[u][v]
                   might be float('inf') if unreachable.

    Returns:
        The total weight of the MST, or float('inf') if the graph is disconnected.
    """
    if not nodes:
        return 0

    nodes_set = set(nodes)
    start_node = nodes[0] # Start Prim's from an arbitrary node

    visited = {start_node}
    mst_weight = 0
    # Priority queue stores (weight, node)
    edges = []

    # Add initial edges from the start node to all other nodes
    if start_node in distances:
        for neighbor in nodes_set - visited:
            if neighbor in distances[start_node]:
                 heapq.heappush(edges, (distances[start_node][neighbor], neighbor))

    while edges and len(visited) < len(nodes_set):
        weight, current_node = heapq.heappop(edges)

        if current_node in visited:
            continue

        # If the edge weight is infinity, the graph is disconnected
        if weight == float('inf'):
             return float('inf')

        visited.add(current_node)
        mst_weight += weight

        # Add edges from the newly visited node to all unvisited nodes
        if current_node in distances:
            for neighbor in nodes_set - visited:
                 if neighbor in distances[current_node]:
                     heapq.heappush(edges, (distances[current_node][neighbor], neighbor))

    # If not all nodes were visited, the graph is disconnected
    if len(visited) < len(nodes_set):
         return float('inf')

    return mst_weight


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions (tighten, pickup, walk)
    required to tighten all goal nuts. It considers the need to acquire
    usable spanners and travel to nut locations.

    # Assumptions
    - There is exactly one man.
    - Nuts are static (do not change location).
    - Spanners are used once per tighten action.
    - The man can carry multiple spanners simultaneously.
    - Links between locations are bidirectional.
    - Solvable problems have enough usable spanners and a connected location graph.

    # Heuristic Initialization
    - Extracts the set of goal nuts from the task's goal conditions.
    - Builds the location graph (links) from static facts for pathfinding (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of loose nuts that are part of the goal (`LooseGoalNuts`).
    2. If `LooseGoalNuts` is empty, the goal is reached, return 0.
    3. Find the man and his current location (`ManLoc`).
    4. Identify all usable spanners currently in the state (`UsableSpannersInState`).
    5. Check if the total number of usable spanners is less than the number of loose goal nuts. If so, the problem is likely unsolvable (or requires spanner repair, which isn't in the domain), return infinity.
    6. Identify usable spanners the man is currently carrying (`CarriedUsable`).
    7. Calculate the number of additional usable spanners the man needs to pick up (`p = max(0, |LooseGoalNuts| - |CarriedUsable|)`).
    8. The cost for `tighten_nut` actions is `|LooseGoalNuts|`.
    9. The cost for `pickup_spanner` actions is `p`.
    10. Identify the locations of all loose goal nuts (`NutLocations`).
    11. Identify the locations of all usable spanners currently on the ground (`AvailableUsableLocs`).
    12. Determine the set of locations the man *must* visit:
        - All locations in `NutLocations`.
        - If `p > 0`, the `p` locations from `AvailableUsableLocs` that are closest to `ManLoc`.
    13. Compute the shortest path distances between the man's current location and all required visit locations, and between all pairs of required visit locations. This can be done by running BFS starting from the man's location and from each required visit location.
    14. Estimate the walk cost as the cost of the Minimum Spanning Tree (MST) connecting the man's current location and all required visit locations, using the shortest path distances as edge weights. This provides a lower bound on the travel needed to visit all required locations.
    15. The total heuristic value is the sum of the tighten cost, pickup cost, and the estimated walk cost (MST cost).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        super().__init__(task)

        # Store goal nuts from the task goals
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                self.goal_nuts.add(args[0])

        # Build the location graph from links
        self.links = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.links.add((l1, l2))
                self.links.add((l2, l1)) # Assuming links are bidirectional

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts
        loose_nuts_in_state = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}
        loose_goal_nuts = self.goal_nuts.intersection(loose_nuts_in_state)

        # 2. If no loose goal nuts, goal is reached
        if not loose_goal_nuts:
            return 0

        # Find the man (assuming there's only one)
        man = None
        man_loc = None
        carried_spanners = set()
        for fact in state:
            if match(fact, "at", "*", "*") and get_parts(fact)[1] not in self.goal_nuts and not get_parts(fact)[1].startswith("spanner"):
                 # Assuming the object that is 'at' a location and is not a nut or spanner is the man
                 man = get_parts(fact)[1]
                 man_loc = get_parts(fact)[2]
                 break # Found the man and his location

        if man is None:
             # Man might be carrying spanners but not 'at' a location if state representation is tricky
             # Find man from carrying predicate
             for fact in state:
                 if match(fact, "carrying", "*", "*"):
                     man = get_parts(fact)[1]
                     # Need to find his location from a previous state or assume a default?
                     # Let's assume man is always 'at' a location or carrying spanners
                     # If carrying, his location must be implicitly where he picked them up last?
                     # This is tricky. Let's assume man is always explicitly 'at' a location.
                     # If not found, something is wrong with the state representation or domain.
                     # For robustness, let's try to find location from where he picked up spanners
                     # This requires state history or more complex state analysis, which is not typical for h(s).
                     # Let's stick to finding (at man loc)
                     pass # ManLoc remains None if not found via (at man loc)

        if man_loc is None:
             # This state might be unreachable or invalid for a solvable problem
             # Or the man is carrying spanners but his location fact was deleted without adding a new one?
             # Based on 'walk' action, (at man loc) is always present if man is on ground.
             # If man is carrying, he must have been at a location to pick up.
             # Let's assume man_loc is always findable via (at man loc).
             # If not, return infinity as state is likely invalid/unsolvable.
             return float('inf')


        # Find spanners carried by the man
        carried_spanners = {get_parts(fact)[2] for fact in state if match(fact, "carrying", man, "*")}

        # 4. Identify usable spanners in state
        usable_spanners_in_state = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        # 5. Check if enough usable spanners exist in total
        if len(usable_spanners_in_state) < len(loose_goal_nuts):
            return float('inf') # Not enough spanners to tighten all nuts

        # 6. Identify carried usable spanners
        carried_usable = carried_spanners.intersection(usable_spanners_in_state)

        # 7. Calculate needed pickups
        k = len(loose_goal_nuts)
        c = len(carried_usable)
        p = max(0, k - c)

        # 8. Tighten cost
        tighten_cost = k

        # 9. Pickup cost
        pickup_cost = p

        # 10. Identify nut locations
        nut_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1] in loose_goal_nuts}
        nut_locations_set = set(nut_locations.values())

        # 11. Identify available usable spanner locations (on the ground)
        available_usable_spanners = usable_spanners_in_state - carried_spanners
        available_usable_spanner_locs = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1] in available_usable_spanners}
        available_usable_locs_set = set(available_usable_spanner_locs.values())

        # 12. Determine required visit locations
        required_visit_locations = set(nut_locations_set)

        spanner_pickup_locs = set()
        if p > 0:
            # Need to find the p closest available usable spanner locations
            dists_from_man = bfs(man_loc, self.links)
            
            # Filter locations that are reachable and have available spanners
            reachable_available_locs = {loc for loc in available_usable_locs_set if loc in dists_from_man}

            if len(reachable_available_locs) < p:
                 # Not enough reachable spanner locations to get needed spanners
                 return float('inf')

            # Sort reachable available locations by distance from man_loc and take the top p
            sorted_available_locs = sorted(reachable_available_locs, key=lambda loc: dists_from_man[loc])
            spanner_pickup_locs = set(sorted_available_locs[:p])

            required_visit_locations.update(spanner_pickup_locs)

        # 13. & 14. Calculate walk cost using MST
        nodes_for_mst = {man_loc} | required_visit_locations

        # Compute pairwise shortest path distances between nodes_for_mst
        pairwise_dists = {}
        for u in nodes_for_mst:
            dists_from_u = bfs(u, self.links)
            pairwise_dists[u] = {v: dists_from_u.get(v, float('inf')) for v in nodes_for_mst}

        walk_cost = mst_cost(list(nodes_for_mst), pairwise_dists)

        # If MST cost is infinity, it means the required locations are disconnected
        # from the man's current location or from each other.
        if walk_cost == float('inf'):
             return float('inf')

        # 15. Total heuristic value
        total_cost = tighten_cost + pickup_cost + walk_cost

        return total_cost

