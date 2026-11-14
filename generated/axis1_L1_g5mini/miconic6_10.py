from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string, e.g. '(pred a b)' -> ['pred','a','b']"""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (ud3uf11z).
    The domain models a single mobile agent that can move between nodes (t4rdxh32)
    using adjacency facts `acovodrf`, pick up objects of type nxbd1hc3 at locations
    where `kut2ksnk` holds (action `diwlmgg9`), and deliver them at locations
    specified by `klw11kjf` (action `y7nik9q2`). The heuristic estimates the
    number of actions remaining to make all goal predicates `a7sb56om <obj>` true.

    Assumptions
    -----------
    - Movement between two nodes (one application of either movement action)
      costs 1 action.
    - Picking up an object (action diwlmgg9) costs 1 action.
    - Delivering an object (action y7nik9q2) costs 1 action.
    - The agent can carry multiple objects simultaneously (the domain does not
      restrict carrying). The heuristic is optimistic/relaxed in that it treats
      each object's remaining cost independently and sums them.
    - Movement is possible from node X to node Y if the static relation
      `acovodrf X Y` OR `acovodrf Y X` exists (both movement operators together
      permit travel in either direction if either directed edge is present).
    - If an object already has predicate `a7sb56om` in the state, it is
      considered delivered and contributes 0 cost.

    Heuristic Initialization
    ------------------------
    In the constructor we extract static information from the task:
    - The static adjacency (acovodrf) facts are converted into an undirected
      adjacency graph (edges in either orientation allow movement).
    - The static delivery locations (klw11kjf) are stored as a mapping
      from object -> set(of delivery nodes). These are typically static facts.
    - We also remember the set of known location nodes (t4rdxh32) observed
      in the static facts for BFS initialization.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the state's goals are already satisfied return 0.

    2. Identify the agent's current node by locating the single `(omggbn9z <node>)`
       fact in the state. If none is present, return a large finite cost (agent
       not present or malformed state).

    3. For every object p that still needs to be delivered (i.e., the goal
       `a7sb56om p` is not yet true in the current state) compute a lower-bound
       (relaxed) cost to deliver p:
       - If `f2xo7hb6 p` is true in the state (the object is already picked up),
         the minimal remaining cost for p is:
             dist(agent_node, some_delivery_node_for_p) + 1
         (distance to a valid delivery node plus one action to perform the
         `y7nik9q2` delivery).
       - Otherwise (object not yet picked up), find the pickup location(s)
         `w_pick` from state facts `(kut2ksnk p w_pick)` and the delivery
         location(s) `w_del` from static `(klw11kjf p w_del)` and compute:
             dist(agent_node, w_pick) + 1 (pick action) + dist(w_pick, w_del) + 1 (deliver)
         Take the minimum over all pairs of pickup and delivery nodes.
       - If any required distance is unreachable (no path in the adjacency
         graph), treat the cost for that object as a very large number.

    4. Sum the per-object costs to obtain the heuristic value. This heuristic is
       admissible only in special cases; it is designed as an informed, efficiently
       computable estimate for guiding greedy best-first search, trading strict
       admissibility for stronger guidance. It guarantees a finite value for
       solvable states (provided the static graph connects necessary nodes),
       and returns a large finite number if some object is currently unreachable.

    Notes on Implementation
    -----------------------
    - All PDDL facts are parsed using get_parts which strips the surrounding
      parentheses and splits by whitespace.
    - BFS is used for distance computation since all moves have unit cost.
    """

    def __init__(self, task):
        # store the goal conditions for quick goal-check
        self.goals = task.goals

        # Build adjacency (undirected) from static acovodrf facts:
        # Movement from X->Y is possible if acovodrf X Y OR acovodrf Y X.
        self.adj = {}  # node -> set(neighbor nodes)
        self.locations = set()  # set of all known location nodes (t4rdxh32)
        static_facts = task.static

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "acovodrf" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.locations.add(a)
                self.locations.add(b)
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)  # undirected interpretation

        # Extract static delivery mapping from klw11kjf facts
        self.delivery_nodes = {}  # obj -> set(nodes)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "klw11kjf" and len(parts) >= 3:
                obj, node = parts[1], parts[2]
                self.delivery_nodes.setdefault(obj, set()).add(node)
                self.locations.add(node)

        # Ensure every known location is present in adjacency dict
        for loc in list(self.locations):
            self.adj.setdefault(loc, set())

        # large finite cost used for unreachable situations
        self._INF_COST = 10 ** 6

    def _bfs_distances(self, source):
        """
        Return a dict mapping node -> distance (int) in the undirected adjacency
        graph from source. If source is unknown (not in self.adj), return empty dict.
        """
        if source not in self.adj:
            return {}
        dist = {source: 0}
        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = node.state

        # 0 for goal states (check using the stored goals)
        if self.goals <= state:
            return 0

        # Parse state facts to find robot location, pickups, picked-up flags, delivered flags.
        robot_node = None
        picked_up = set()  # objects with f2xo7hb6
        delivered = set()  # objects with a7sb56om
        pickups = {}  # obj -> set(pickup_nodes) from state `(kut2ksnk obj node)`
        # also collect all nodes that appear in the state to extend adjacency if needed
        state_locations = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "omggbn9z" and len(parts) >= 2:
                robot_node = parts[1]
                state_locations.add(robot_node)
            elif pred == "f2xo7hb6" and len(parts) >= 2:
                picked_up.add(parts[1])
            elif pred == "a7sb56om" and len(parts) >= 2:
                delivered.add(parts[1])
            elif pred == "kut2ksnk" and len(parts) >= 3:
                obj, node = parts[1], parts[2]
                pickups.setdefault(obj, set()).add(node)
                state_locations.add(node)
            elif pred == "klw11kjf" and len(parts) >= 3:
                # sometimes klw11kjf could be present in state as well; incorporate as delivery
                obj, node = parts[1], parts[2]
                self.delivery_nodes.setdefault(obj, set()).add(node)
                state_locations.add(node)

        # Make sure adjacency includes any nodes that appeared only in the state
        for loc in state_locations:
            self.adj.setdefault(loc, set())

        # If no robot location found, return a large finite cost (malformed state)
        if robot_node is None:
            return self._INF_COST

        # Precompute distances from robot and from any pickup nodes we will encounter.
        dist_from_robot = self._bfs_distances(robot_node)

        # Build set of all objects that might require delivery:
        # Any object that appears in delivered/picked_up/pickups/delivery_nodes
        candidate_objs = set(self.delivery_nodes.keys()) | set(pickups.keys()) | picked_up | delivered

        total_cost = 0

        # For caching BFS distances from arbitrary pickup nodes to delivery nodes
        bfs_cache = {}

        for obj in candidate_objs:
            # Skip if already delivered
            if obj in delivered:
                continue

            # If the domain goal requires a7sb56om for a different set of objects,
            # but this object is not part of the goal, then we don't need to deliver it.
            # Determine whether this object is relevant by checking if "a7sb56om obj" is a goal:
            goal_fact = f"(a7sb56om {obj})"
            if self.goals and goal_fact not in self.goals and any(g.startswith("(a7sb56om ") for g in self.goals):
                # If goals contain at least one a7sb56om predicate but not for this object,
                # then skip this object (it's not part of the goal set).
                # Otherwise keep it (conservative).
                continue

            # Determine set of delivery nodes for this object
            dels = self.delivery_nodes.get(obj, set())
            if not dels:
                # If no delivery location is known for this object, treat as unreachable.
                total_cost += self._INF_COST
                continue

            # If object already picked up
            if obj in picked_up:
                # cost = dist(robot -> nearest delivery) + 1 (drop)
                best = self._INF_COST
                for d in dels:
                    d_dist = dist_from_robot.get(d)
                    if d_dist is None:
                        # If we haven't precomputed distance to d from robot, compute it now
                        d_dist = dist_from_robot.get(d, None)
                    if d_dist is None:
                        # unreachable
                        continue
                    best = min(best, d_dist + 1)
                if best == self._INF_COST:
                    total_cost += self._INF_COST
                else:
                    total_cost += best
                continue

            # Object not picked up yet: need to go to some pickup, pick, go to delivery, drop.
            # Find pickup nodes for this object in the current state.
            pick_nodes = pickups.get(obj, set())
            if not pick_nodes:
                # No pickup known in state -> unreachable (or already removed), charge INF
                total_cost += self._INF_COST
                continue

            best_obj_cost = self._INF_COST
            # For each pickup node, compute robot->pickup + 1 + pickup->delivery + 1
            for p in pick_nodes:
                # distance robot -> p
                d1 = dist_from_robot.get(p)
                if d1 is None:
                    # If robot can't reach the pickup, skip this pickup
                    continue

                # BFS from pickup p to get distances to delivery nodes (cached)
                if p not in bfs_cache:
                    bfs_cache[p] = self._bfs_distances(p)
                dist_from_pick = bfs_cache[p]

                # For each possible delivery node
                for d in dels:
                    d2 = dist_from_pick.get(d)
                    if d2 is None:
                        continue
                    cost = d1 + 1 + d2 + 1  # move to pick + pick + move to del + drop
                    if cost < best_obj_cost:
                        best_obj_cost = cost

            if best_obj_cost == self._INF_COST:
                total_cost += self._INF_COST
            else:
                total_cost += best_obj_cost

        # If total_cost is extremely large, cap it to the large finite value to keep heuristic finite.
        if total_cost >= self._INF_COST:
            return self._INF_COST
        return total_cost