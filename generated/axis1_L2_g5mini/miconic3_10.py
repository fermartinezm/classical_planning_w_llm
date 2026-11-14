from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a fact string like '(pred a b)' -> ['pred','a','b']."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the secret_domain (unkhrp2l). The domain models a single
    movable marker (predicate v0yy74n7) that can traverse a directed graph
    (edges pvhp8m0p). Objects of type jxhpkqr5 need to be transported from
    their pickup locations (pxch7pi8 facts in the state) to their delivery
    locations (pahbmn6t facts, available as static info). Transport
    consists of:
      - moving the marker along pvhp8m0p edges (each move is one action),
      - performing a pickup at the object's pickup location (makpz2tc, 1 action),
      - performing a delivery at the object's target location (jg64qouf, 1 action).

    The heuristic estimates the remaining number of actions (moves + pickup +
    delivery) required to achieve all goal guby8wxu(...) facts. It simulates
    a greedy route for the marker: always move to the nearest relevant node
    (a pickup location for an unpicked object or a delivery location for a
    picked object), picking/delivering all objects available at the node
    before moving on.

    Assumptions
    -----------
    - The directed graph for marker movement is encoded by static pvhp8m0p facts.
    - Delivery targets (pahbmn6t object -> location) are static facts (or at
      least available in the task static information).
    - Pickup facts (pxch7pi8) are dynamic and present in the current state when
      an object is still at its pickup location.
    - The marker can carry multiple objects simultaneously (the domain allows
      multiple eacj8ftl facts). Therefore the heuristic allows picking multiple
      objects at the same pickup node and subsequently delivering them when
      the marker reaches their delivery nodes.
    - Each pickup and each delivery counts as one action.
    - Each directed edge traversal between locations counts as one action.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts:
    - The set of goal objects (those that must satisfy guby8wxu(...)).
    - The static directed graph of locations from pvhp8m0p facts.
    - The mapping object -> delivery_location from pahbmn6t static facts.
    The graph is stored as adjacency lists for BFS-based distance computation.
    Distances are computed on demand (BFS) and cached.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the provided state already satisfies all goal facts, return 0.
    2. Parse the current state to determine:
       - Current marker location (the unique fact v0yy74n7 X).
       - For each goal object:
         * Whether it is already delivered (guby8wxu present) -> skip.
         * Whether it is already picked up (eacj8ftl present) -> mark as
           "picked", so only delivery action remains.
         * Whether it is still at a pickup location (pxch7pi8 present) -> record pickup location.
         * Its static delivery location (from pahbmn6t).
    3. Prepare a greedy simulation:
       - current_loc := marker location.
       - Loop until all goal objects are delivered:
         a) If there are any picked objects whose delivery location == current_loc,
            deliver all of them (1 action each).
         b) Else if there are any unpicked objects whose pickup location == current_loc,
            pick all of them (1 action each) and mark them as picked.
         c) Else choose the nearest relevant node (either a delivery location for
            some picked object or a pickup location for some unpicked object)
            using BFS shortest-path distances on the directed graph. Move the marker
            there (cost = distance in edges).
       - Each pick adds +1 action, each delivery +1, and each move adds number-of-edges actions.
    4. If a required distance between nodes does not exist in the graph (no path),
       a large finite penalty is used to keep the heuristic finite; such states
       are likely unsolvable, but we still return a finite number.
    5. Return the accumulated estimated action count.
    """

    def __init__(self, task):
        """Extract static graph and delivery targets from task static facts."""
        self.goals = task.goals  # set of goal fact strings like '(guby8wxu obj)'
        self.static = task.static

        # Build directed adjacency list for locations from pvhp8m0p facts.
        self.graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.graph.setdefault(a, set()).add(b)
                # ensure nodes exist
                self.graph.setdefault(b, set())

        # Map object -> delivery location from pahbmn6t static facts.
        self.delivery = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pahbmn6t" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                self.delivery[obj] = loc

        # Cache for BFS distances between locations: dist_cache[src][dst] = int
        self.dist_cache = {}

        # Max penalty for unreachable distances (finite but large)
        # We set it relative to graph size to remain finite.
        self.large_penalty = max(100, len(self.graph) * 10)

    def bfs_distance(self, src, dst):
        """Return shortest directed distance in edges from src to dst or large_penalty if unreachable."""
        if src == dst:
            return 0
        if src not in self.graph or dst not in self.graph:
            return self.large_penalty
        if src in self.dist_cache and dst in self.dist_cache[src]:
            return self.dist_cache[src][dst]

        # BFS
        visited = {src}
        q = deque([(src, 0)])
        while q:
            node, d = q.popleft()
            for nb in self.graph.get(node, ()):
                if nb == dst:
                    # store into cache
                    self.dist_cache.setdefault(src, {})[dst] = d + 1
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))

        # unreachable
        self.dist_cache.setdefault(src, {})[dst] = self.large_penalty
        return self.large_penalty

    def __call__(self, node):
        """Compute and return the heuristic estimate (integer)."""
        state = node.state

        # If all goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Find marker location.
        marker_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "v0yy74n7" and len(parts) >= 2:
                marker_loc = parts[1]
                break
        if marker_loc is None:
            # No marker found; return a finite large heuristic.
            return self.large_penalty * len(self.goals)

        # Build per-object status for all goal objects.
        goal_objects = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "guby8wxu" and len(parts) >= 2:
                goal_objects.append(parts[1])

        # For each object, determine pickup loc (if any) and delivery loc.
        objs = {}
        for obj in goal_objects:
            # If already delivered, skip entirely.
            if f"(guby8wxu {obj})" in state:
                continue

            picked = f"(eacj8ftl {obj})" in state
            # Find pickup location from pxch7pi8 facts in the state (dynamic).
            pickup_loc = None
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "pxch7pi8" and parts[1] == obj and len(parts) >= 3:
                    pickup_loc = parts[2]
                    break

            # Delivery location from static mapping. If not present, heuristic will still be finite.
            delivery_loc = self.delivery.get(obj, None)

            objs[obj] = {
                "picked": picked,
                "pickup_loc": pickup_loc,
                "delivery_loc": delivery_loc,
                "delivered": False,
            }

        # If no remaining objects, return 0.
        if not objs:
            return 0

        # Greedy simulation: move marker to nearest next relevant node, pick/deliver all possible.
        cur = marker_loc
        cost = 0

        # Helper to check if any remaining to deliver
        def any_remaining():
            return any(not info["delivered"] for info in objs.values())

        # Precompute set of all relevant locations to ensure BFS can see them
        for info in objs.values():
            for loc in (info["pickup_loc"], info["delivery_loc"]):
                if loc is not None and loc not in self.graph:
                    # add isolated node to graph so BFS can reason about equality and presence
                    self.graph.setdefault(loc, set())

        # Simulation loop
        while any_remaining():
            made_progress = False

            # 1) Deliver all picked objects whose delivery location == cur
            to_deliver = [
                obj
                for obj, info in objs.items()
                if (not info["delivered"])
                and info["picked"]
                and info["delivery_loc"] == cur
            ]
            if to_deliver:
                for obj in to_deliver:
                    cost += 1  # jg64qouf deliver action
                    objs[obj]["delivered"] = True
                    # also picked -> false (no longer carrying), but we mark delivered
                made_progress = True
                continue  # re-evaluate at same location

            # 2) Pick up all unpicked objects whose pickup_loc == cur
            to_pick = [
                obj
                for obj, info in objs.items()
                if (not info["delivered"])
                and (not info["picked"])
                and (info["pickup_loc"] == cur)
            ]
            if to_pick:
                for obj in to_pick:
                    cost += 1  # makpz2tc pickup action
                    objs[obj]["picked"] = True
                made_progress = True
                continue  # re-evaluate at same location

            # 3) If nothing to do here, choose nearest relevant node
            # Candidate nodes: delivery locations of picked objects, pickup locations of unpicked objects
            candidates = set()
            for info in objs.values():
                if info["delivered"]:
                    continue
                if info["picked"]:
                    if info["delivery_loc"] is not None:
                        candidates.add(info["delivery_loc"])
                else:
                    if info["pickup_loc"] is not None:
                        candidates.add(info["pickup_loc"])
                    else:
                        # If pickup location unknown (unexpected), we must at least go to delivery location
                        if info["delivery_loc"] is not None:
                            candidates.add(info["delivery_loc"])

            if not candidates:
                # No place to go (missing info), add a finite penalty for each undelivered object.
                remaining = sum(1 for info in objs.values() if not info["delivered"])
                cost += remaining * (1 + self.large_penalty)
                break

            # Choose candidate with minimal BFS distance from cur
            best_loc = None
            best_dist = None
            for loc in candidates:
                d = self.bfs_distance(cur, loc)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_loc = loc

            # Move there
            if best_loc is None:
                # defensive: add penalty and break
                remaining = sum(1 for info in objs.values() if not info["delivered"])
                cost += remaining * (1 + self.large_penalty)
                break

            # Add move cost and update cur
            cost += best_dist
            cur = best_loc
            made_progress = True

            if not made_progress:
                # Safety net to avoid infinite loop.
                remaining = sum(1 for info in objs.values() if not info["delivered"])
                cost += remaining * (1 + self.large_penalty)
                break

        return int(cost)