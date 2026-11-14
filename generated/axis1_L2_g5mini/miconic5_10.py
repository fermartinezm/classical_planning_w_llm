from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10**6


def get_parts(fact):
    """Return the tokens of a PDDL fact string without the surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (unkhrp2l) domain.

    In this domain there is a single mobile vehicle-like predicate (v0yy74n7)
    that indicates the vehicle's current location (a ql6b2h31 object).
    Items (objects of type jxhpkqr5) are initially at some locations (pxch7pi8).
    The vehicle can move along directed edges between locations (pvhp8m0p).
    The vehicle can pick up items at its current location (makpz2tc), which
    makes eacj8ftl(item) true (item is "held"), and can deliver held items to
    the item's goal/destination using jg64qouf when the vehicle is at the
    item's designated destination (pahbmn6t(item, dest)), which creates
    guby8wxu(item) (the goal atom).

    The heuristic estimates the number of actions (moves, pickups, deliveries)
    required to reach a goal state in a reasonably informed but efficiently
    computable way.

    Assumptions
    -----------
    - The static predicate "pahbmn6t" maps each item to its delivery location
      (destination) and is available in the static facts.
    - The directed edges between locations are provided by the static predicate
      "pvhp8m0p". Movement along an edge costs 1 action (jt9wnwgs / v90zoaro).
    - Picking up an item (makpz2tc) costs 1 action and requires the vehicle to
      be at the item's current location.
    - Delivering a held item (jg64qouf) costs 1 action and requires the vehicle
      to be at the item's destination and the item to be held.
    - The vehicle can carry multiple items simultaneously (no capacity limit).
    - If a fact that we expect (e.g., pickup location) is missing in the state,
      the heuristic handles it gracefully by assuming a large travel cost
      (but still finite).

    Heuristic Initialization
    ------------------------
    During initialization the heuristic extracts:
    - The set of goal items (those that must satisfy guby8wxu(item)).
    - A mapping from item -> destination location using static facts of the
      form (pahbmn6t item location).
    - The directed adjacency graph of locations using static facts of the form
      (pvhp8m0p loc-from loc-to).
    - All-pairs shortest-path distances (directed) between known locations
      (computed with BFS from every known location). Distances that do not
      exist are treated as "very large" (INF).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic proceeds as follows:

    1. If all goal atoms are already true in the state, return 0.

    2. Determine the vehicle's current location by inspecting facts of the form
       (v0yy74n7 LOCATION). If missing, the vehicle position is considered
       unknown (the heuristic still returns finite values but penalizes missing
       information).

    3. For each goal item (an item appearing in the goal as guby8wxu ITEM):
       - If guby8wxu(ITEM) already holds in the state, the item is ignored.
       - Otherwise determine whether the item is currently held
         (eacj8ftl ITEM in state). If so, its pickup location is treated as
         the current vehicle location (no extra pick action required).
       - If not held, find the item's current location from facts of the form
         (pxch7pi8 ITEM LOCATION) in the current state.
       - Retrieve the item's destination location from the static mapping
         (pahbmn6t ITEM DEST). If that mapping is missing, a conservative
         choice is made (treated as unknown destination and penalized).

       For each undelivered item we compute a base cost:
         base_cost(item) = pick_cost + deliver_cost + dist(pickup_loc, dest)
       where:
         - pick_cost = 0 if item is held, else 1 (makpz2tc)
         - deliver_cost = 1 (jg64qouf)
         - dist(pickup_loc, dest) is the shortest-path distance in the
           directed location graph. If pickup_loc or dest is missing or
           unreachable the distance uses a large finite value.

    4. Visiting multiple pickup locations: the base costs already include the
       travel from each pickup to its destination. However the vehicle may need
       to move from its current location to the first pickup and between pickup
       locations before the pickups occur. To approximate this extra required
       travel, we:
         - Collect all distinct pickup locations for items that are not held.
         - Estimate the additional travel to visit all pickup locations using
           a Minimum Spanning Tree (MST) heuristic on these pickup nodes under
           the metric induced by the precomputed distances. This approximates a
           short route to cover all pickup locations (it is a lower-cost
           approximation to a tour).
         - Add the minimal distance from the vehicle's current location to the
           closest pickup as the cost to start the pickup tour.

       If there are no unpicked items, no MST or start-cost is added (only
       held items contribute via their base costs).

    5. The final heuristic value is:
         h(state) = sum(base_cost(item) for undelivered items) + extra_travel_visiting_pickups

       Notes on admissibility and practicality:
       - This heuristic is not guaranteed to be admissible: the MST and the
         per-item decomposition are approximations intended to make the value
         informed and efficiently computable. However it tends to be
         optimistic about the travel required to visit multiple locations
         (since actual routes may be longer), which helps greedy best-first
         search by providing informative priorities.
       - If any distance is missing/unreachable the heuristic uses a large
         finite penalty (INF) to keep values finite while strongly discouraging
         exploring such states.

    """

    def __init__(self, task):
        # Goal atoms (strings like "(guby8wxu item)")
        self.goals = task.goals

        # Extract item->destination mapping from static facts (pahbmn6t item loc)
        self.dest = {}
        # Build directed adjacency graph of locations from static facts pvhp8m0p locFrom locTo
        self.adj = {}
        self.locations = set()

        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "pahbmn6t" and len(parts) >= 3:
                item, loc = parts[1], parts[2]
                self.dest[item] = loc
                self.locations.add(loc)
            elif pred == "pvhp8m0p" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.locations.add(a)
                self.locations.add(b)
                self.adj.setdefault(a, set()).add(b)

        # Also include any destinations (already added) and adjacency endpoints as locations
        # Precompute directed shortest-path distances between known locations using BFS
        self.dist = {}  # dist[src][tgt] = distance (int)
        for src in self.locations:
            # BFS from src
            d = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.adj.get(u, ()):
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            self.dist[src] = d

        # Extract list of items that appear in goals (guby8wxu item)
        self.goal_items = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "guby8wxu" and len(parts) >= 2:
                self.goal_items.append(parts[1])

    def __call__(self, node):
        state = node.state

        # 1) If all goals already satisfied -> heuristic 0
        if self.goals <= state:
            return 0

        # 2) Find vehicle location (v0yy74n7 LOC)
        vehicle_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "v0yy74n7" and len(parts) >= 2:
                vehicle_loc = parts[1]
                break

        # 3) For each goal item not yet delivered compute base cost
        undelivered_items = []
        sum_base = 0
        pickup_locations = set()  # only for items not currently held
        for item in self.goal_items:
            goal_fact = f"(guby8wxu {item})"
            if goal_fact in state:
                continue  # already delivered

            held_fact = f"(eacj8ftl {item})"
            if held_fact in state:
                # item is held: pickup cost 0, need to travel from vehicle to destination and deliver
                pick_cost = 0
                deliver_cost = 1
                # pickup location is vehicle location for distance computation
                pickup_loc = vehicle_loc
            else:
                # not held: we must visit the item's current location, pick it, then go to dest
                pick_cost = 1
                deliver_cost = 1
                pickup_loc = None
                # find a pxch7pi8 fact in the state that gives the item's location
                # pxch7pi8 ITEM LOC
                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == "pxch7pi8" and len(parts) >= 3 and parts[1] == item:
                        pickup_loc = parts[2]
                        break
                if pickup_loc:
                    pickup_locations.add(pickup_loc)

            # destination from static mapping
            dest_loc = self.dest.get(item, None)

            # distance from pickup_loc to destination
            if pickup_loc is None or dest_loc is None:
                # missing information -> penalize heavily but keep finite
                dist_pick_dest = 50
            else:
                # if pickup_loc or dest_loc unknown to precomputed locations, treat as unreachable
                if pickup_loc in self.dist and dest_loc in self.dist[pickup_loc]:
                    dist_pick_dest = self.dist[pickup_loc][dest_loc]
                else:
                    dist_pick_dest = 50  # finite large penalty

            base_cost = pick_cost + deliver_cost + dist_pick_dest
            sum_base += base_cost
            undelivered_items.append(item)

        # 4) Extra travel to visit pickup locations (for items that are not held)
        extra = 0
        if pickup_locations:
            # minimal distance from vehicle to any pickup
            if vehicle_loc is None:
                min_start = 10  # unknown vehicle -> small penalty
            else:
                min_start = min(
                    (self.dist.get(vehicle_loc, {}).get(p, INF) if vehicle_loc in self.dist else INF)
                    for p in pickup_locations
                )
                if min_start >= INF:
                    min_start = 50  # unreachable -> penalize

            # approximate cost to connect all pickup locations via MST (Prim)
            # nodes are pickup_locations; edge weights are shortest-path distances
            nodes = list(pickup_locations)
            if len(nodes) == 1:
                mst_len = 0
            else:
                # Prim's algorithm
                in_mst = {nodes[0]}
                remaining = set(nodes[1:])
                mst_len = 0
                while remaining:
                    best = None
                    best_d = INF
                    for u in in_mst:
                        for v in remaining:
                            # distance u->v might not be defined; since we need undirected metric
                            # approximate distance by min(dist(u,v), dist(v,u)) if available
                            d_uv = INF
                            if u in self.dist and v in self.dist[u]:
                                d_uv = self.dist[u][v]
                            d_vu = INF
                            if v in self.dist and u in self.dist[v]:
                                d_vu = self.dist[v][u]
                            d = min(d_uv, d_vu)
                            if d < best_d:
                                best_d = d
                                best = v
                    if best is None or best_d >= INF:
                        # graph is disconnected in known metric: add penalty and break
                        mst_len += 50
                        # move one arbitrary node into MST to continue
                        node_to_add = remaining.pop()
                        in_mst.add(node_to_add)
                    else:
                        mst_len += best_d
                        in_mst.add(best)
                        remaining.remove(best)

            extra = min_start + mst_len

        # Final heuristic combine base costs and extra travel
        h_val = int(sum_base + extra)

        # Guarantee non-negative finite integer
        if h_val < 0:
            h_val = 0
        if h_val >= INF:
            h_val = INF

        return h_val