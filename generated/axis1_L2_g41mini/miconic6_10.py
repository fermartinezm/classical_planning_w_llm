from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary:
    This heuristic estimates the number of actions required to transition all objects of type `jxhpkqr5`
    from having the predicate `eacj8ftl` true to the goal condition where the predicate `guby8wxu` is true
    for those same objects. The domain involves predicates that relate two objects of type `ql6b2h31` in a graph-like structure,
    and a special unary predicate `v0yy74n7` holds exactly on one `ql6b2h31` object representing a "current position".
    The heuristic estimates how many move actions and state transitions for each relevant `jxhpkqr5` object are necessary,
    relying on shortest-path computations over graph edges defined by `pvhp8m0p` static facts, and uses static relations
    `pahbmn6t` and `pxch7pi8` to estimate the minimal steps to enable the necessary switch from `eacj8ftl` to `guby8wxu`.

    Assumptions:
    - Each `jxhpkqr5` object is initially either in state `eacj8ftl` or already in goal state `guby8wxu`.
    - There is exactly one `ql6b2h31` object currently having `(v0yy74n7 ?loc)`, representing the current position.
    - The predicate `(pvhp8m0p ?from ?to)` defines directed edges between `ql6b2h31` objects, forming a graph.
    - The operator `jt9wnwgs` and `v90zoaro` move the current position along these edges.
    - To switch a `jxhpkqr5` object from `eacj8ftl` to `guby8wxu`, it must be at certain positions related via `pahbmn6t` and `pxch7pi8`.
    - The heuristic is not required to be admissible but should be efficiently computable and informative.

    Heuristic Initialization:
    - Extract static facts defining the graph edges (`pvhp8m0p`) and construct a bidirectional adjacency list.
    - Extract static relations `pahbmn6t` and `pxch7pi8` linking `jxhpkqr5` objects to specific `ql6b2h31` objects.
    - Store all `jxhpkqr5` objects that appear in the goal with `guby8wxu`.
    - Precompute a mapping from each `ql6b2h31` node to its reachable nodes via shortest distance using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current location of the "robot" or current position from `(v0yy74n7 ?loc)`.
    2. Collect all `jxhpkqr5` objects that still require to reach goal `(guby8wxu)` but currently hold `(eacj8ftl)` (not yet transformed).
    3. For each such `jxhpkqr5` object, there are two relevant sets of nodes:
       - The `pahbmn6t` node(s) where that object can be transformed (requires robot to be at that node).
       - The `pxch7pi8` node(s) where the transformation can be undone or prepared.
    4. The transformation consists of:
       - Moving the position to a `pahbmn6t` node of the object (to use `jg64qouf` to switch from `eacj8ftl` to `guby8wxu`).
       - The heuristic approximates the effort as the shortest path from the current position to that node.
       - As some preconditions involve possibly first going to some `pxch7pi8` node to "rearm" or "reset" states (`makpz2tc`), the heuristic considers minimal combined effort.
    5. The heuristic sums the minimal movements and transformations needed for all not-yet-goal objects.
    6. To inform the number of required moves:
       - Uses shortest distances in the graph between relevant nodes.
       - Adds minimal required transformations assuming each requires one action beyond moving.
    7. Returns 0 if the state already satisfies the goal predicate `guby8wxu` for all relevant objects.

    This approach balances the cost of moving the position across the graph and the state changes on the objects,
    using static information of connectivity and object-location relations to guide search effectively.
    """

    def __init__(self, task):
        """Initialize based on the static facts and goal information."""
        self.goals = task.goals
        self.static = task.static

        # Identify all jxhpkqr5 objects from goals that require (guby8wxu ?obj).
        self.goal_jxhpkqr5 = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "guby8wxu" and len(parts) == 2:
                self.goal_jxhpkqr5.add(parts[1])

        # Extract the graph edges: pvhp8m0p ?from ?to
        # Build adjacency list for undirected moves (jt9wnwgs and v90zoaro both allow bidirectional moves)
        self.graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p" and len(parts) == 3:
                src, dest = parts[1], parts[2]
                self.graph.setdefault(src, set()).add(dest)
                self.graph.setdefault(dest, set()).add(src)

        # Extract pahbmn6t relations: (pahbmn6t ?obj ?loc)
        # Used for transformation (eacj8ftl -> guby8wxu)
        self.pahbmn6t = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pahbmn6t" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.pahbmn6t.setdefault(obj, set()).add(loc)

        # Extract pxch7pi8 relations: (pxch7pi8 ?obj ?loc)
        # Used for "arming" or "resetting" (switching pxch7pi8 -> eacj8ftl)
        self.pxch7pi8 = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pxch7pi8" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.pxch7pi8.setdefault(obj, set()).add(loc)

        # Precompute shortest paths in the graph between all nodes using BFS on-demand cache.
        # We implement a shortest_path_length function with caching.
        self._dist_cache = {}

    def shortest_path_length(self, start, goal):
        """Compute shortest path length between two nodes in the graph using BFS."""
        if start == goal:
            return 0
        if (start, goal) in self._dist_cache:
            return self._dist_cache[(start, goal)]
        if (goal, start) in self._dist_cache:
            return self._dist_cache[(goal, start)]

        from collections import deque

        visited = set([start])
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.graph.get(current, []):
                if neighbor == goal:
                    self._dist_cache[(start, goal)] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # If unreachable, return a large number to indicate difficulty.
        self._dist_cache[(start, goal)] = float("inf")
        return float("inf")

    def __call__(self, node):
        """Compute heuristic estimate of actions needed to reach the goal from the current state."""
        state = node.state

        # If already in goal, heuristic is zero.
        if self.goals <= state:
            return 0

        # Find current position = the single node with (v0yy74n7 ?loc)
        current_positions = [get_parts(fact)[1] for fact in state if fact.startswith("(v0yy74n7 ")]
        if not current_positions:
            # Defensive: no current position found, very bad state, return large heuristic.
            return 1000
        pos = current_positions[0]

        # Identify objects that still need transition from eacj8ftl to guby8wxu:
        # Objects in goal with guby8wxu but currently missing that predicate => must transform
        # Objects with eacj8ftl predicate currently true need to be switched to guby8wxu
        objs_to_transform = set()
        for obj in self.goal_jxhpkqr5:
            goal_fact = f"(guby8wxu {obj})"
            eacj_fact = f"(eacj8ftl {obj})"
            if goal_fact not in state and eacj_fact in state:
                objs_to_transform.add(obj)
            # Also if the object neither has eacj8ftl nor guby8wxu goal satisfied, heuristic won't be zero; count as transform needed.
            elif goal_fact not in state and eacj_fact not in state:
                # Possibly intermediate or invalid state, consider transform needed
                objs_to_transform.add(obj)

        # If none need transform, heuristic is zero.
        if not objs_to_transform:
            return 0

        total_cost = 0

        # For each object to transform:
        # 1) Find minimal cost to go to a pahbmn6t location (where transformation to goal occurs)
        # 2) Consider if the object might need preliminary "arming" at pxch7pi8 locations
        #    We assume at least one pxch7pi8 location must be visited before transformation if the object still has pxch7pi8 predicate true.
        # 3) Add a cost of 1 action for the transformation itself (jg64qouf action)
        # 4) Add potential cost for "arming" action (makpz2tc), count it as 1
        # Because we do not track full states exactly, we estimate by minimal distances.

        for obj in objs_to_transform:
            # The robot must move to a pahbmn6t site of this object to apply final transformation.
            pahb_nodes = self.pahbmn6t.get(obj, set())
            pxch_nodes = self.pxch7pi8.get(obj, set())

            if not pahb_nodes:
                # No place to transform? Return an infinite heuristic since goal is unreachable for this object.
                return float("inf")

            # We check if the object currently holds pxch7pi8 predicate (indicating it might need "arming")
            obj_has_pxch = f"(pxch7pi8 {obj} " in [fact[:10] for fact in state]  # approximate check - better to scan full

            # More precise: check if any fact starts with (pxch7pi8 obj ...)
            obj_pxch_facts = [fact for fact in state if fact.startswith(f"(pxch7pi8 {obj} ")]
            obj_has_pxch = bool(obj_pxch_facts)

            # Compute minimal distance cost from current position to any pxch node (if needed)
            dist_to_pxch = float("inf")
            if obj_has_pxch and pxch_nodes:
                for n in pxch_nodes:
                    d = self.shortest_path_length(pos, n)
                    if d < dist_to_pxch:
                        dist_to_pxch = d

            # Compute minimal distance cost from current position (or from pxch node if arming needed) to any pahbmn6t node
            dist_to_pahb = float("inf")
            # if arming needed: position starts from pxch node; otherwise from pos
            candidates_start = [pos]
            if obj_has_pxch and pxch_nodes:
                # We consider cost if passing through pxch first, so cost from pxch to pahbmn6t plus dist_to_pxch
                for pxch_node in pxch_nodes:
                    for pahb_node in pahb_nodes:
                        dist = self.shortest_path_length(pxch_node, pahb_node)
                        if dist != float("inf") and dist + dist_to_pxch < dist_to_pahb:
                            dist_to_pahb = dist + dist_to_pxch
                # If direct from pos to pahb is cheaper than through pxch, consider that also (choose min)
                for pahb_node in pahb_nodes:
                    direct_dist = self.shortest_path_length(pos, pahb_node)
                    if direct_dist < dist_to_pahb:
                        dist_to_pahb = direct_dist
            else:
                for pahb_node in pahb_nodes:
                    d = self.shortest_path_length(pos, pahb_node)
                    if d < dist_to_pahb:
                        dist_to_pahb = d

            # If unreachable scenario
            if dist_to_pahb == float("inf"):
                return float("inf")

            # Calculate estimated cost for this object:
            # Movement cost: dist_to_pahb
            # plus 1 action for the transform (jg64qouf)
            # plus 1 action for arming (makpz2tc) if needed
            # plus 0 or 1 moves for the arming if distant
            arming_cost = 0
            arming_move_cost = 0
            if obj_has_pxch and pxch_nodes:
                # At least one arming action needed.
                arming_cost = 1
                # Also count movement from pos to pxch node if it requires moves (dist_to_pxch)
                # Already included in dist_to_pahb via sum above if passing through pxch,
                # but if choosing direct dist_to_pahb, we add missing movement separately.
                if dist_to_pahb < dist_to_pxch + min(self.shortest_path_length(n, p) for n in pxch_nodes for p in pahb_nodes):
                    # If direct dist_to_pahb chosen might underestimate movement via pxch, 
                    # add extra movement cost for going to pxch then to pahb
                    arming_move_cost = 0
                else:
                    # Movement to pxch counted already
                    arming_move_cost = 0
            total_cost += dist_to_pahb + 1 + arming_cost + arming_move_cost

            # Update the current position estimate to the last pahb node visited for next object:
            # Approximate by setting pos to closest pahb node
            # Pick pahb node with minimal distance from pos
            min_pahb_node = None
            min_dist = float("inf")
            for n in pahb_nodes:
                d = self.shortest_path_length(pos, n)
                if d < min_dist:
                    min_pahb_node = n
                    min_dist = d
            if min_pahb_node is not None:
                pos = min_pahb_node

        # Return total cost rounded up to integer for heuristic consistency
        return int(total_cost)
