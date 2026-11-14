from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into its parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (hrpmxpjg) domain.

    The heuristic estimates the number of actions required to achieve all goal
    sjuejzxf facts (i.e., to place each goal object at its goal location).
    It primarily reasons over the location graph encoded by the static facts
    (miud1izi ...) and accounts for the one-step costs of loading/unloading an
    object from a carrier (un606moq <-> sjuejzxf transitions). Movement along
    adjacent locations (miud1izi edges) is counted as one action per step.

    Assumptions
    -----------
    - Goals in this domain of interest are of the form (sjuejzxf <object> <loc>).
    - Objects mentioned in those goal facts may appear in the current state as:
        * (sjuejzxf <object> <loc>) -- object directly at a location
        * (un606moq <object> <vehicle>) -- object currently inside a vehicle
    - Vehicles (and other moveable entities) that have sjuejzxf facts can move
      along the miud1izi graph; unloading a payload from a vehicle requires
      one action (tmoxe3x5-like) and loading into a vehicle requires one
      action (szsevy61-like). For a conservative, quickly computable estimate
      we only account explicitly for the unload action cost when the object is
      inside a vehicle (a single action), and then the movement cost along the
      location graph.
    - Moving an object between two adjacent locations (as described by
      miud1izi static facts) requires one action (arsslvyt-like). We compute
      the shortest path distance on that graph and use it as the number of
      move actions needed.

    Heuristic Initialization
    ------------------------
    - Extract all goal sjuejzxf facts from task.goals and store them as target
      (object -> location) pairs.
    - Parse static facts to build an adjacency map of locations using the
      miud1izi predicate. The graph is undirected for planning purposes:
      if (miud1izi a b) exists we consider an edge a <-> b (note static facts
      often contain both directions, but we add both to be robust).
    - Prepare an internal cache for BFS distances between locations to avoid
      repeated expensive searches during heuristic calls.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. For every goal (sjuejzxf OBJ GOAL_LOC):
       a) Determine OBJ's current disposition in the state:
          - If a fact (sjuejzxf OBJ CUR_LOC) exists, OBJ is at CUR_LOC.
          - Else if a fact (un606moq OBJ VEH) exists, OBJ is inside VEH.
            * In this case, determine VEH's current location via
              (sjuejzxf VEH VEH_LOC) if present.
            * We account for one unload action to make the object appear at
              VEH_LOC, then plan movement from VEH_LOC to GOAL_LOC.
          - Else: the object is neither at a location nor inside a vehicle in
            the current state representation the planner provides. We assume
            an approximate finite cost by treating the object as distant:
            assign a conservative large finite cost (see implementation).
       b) If the object (after an unload if applicable) is already at the
          goal location, cost 0 for that object.
       c) Otherwise, compute the shortest path length (number of miud1izi
          hops) between the object's current location and GOAL_LOC using BFS.
          Each hop is modeled as one move action. If no path exists in the
          static adjacency graph, use a conservative finite fallback cost.
    2. Sum the individual costs for all goal objects and return the sum as the
       heuristic value.
    """

    # A conservative finite fallback distance used when the graph does not
    # provide a path between two locations. This keeps heuristic finite.
    _UNREACHABLE_DISTANCE = 1000

    def __init__(self, task):
        """
        Extract static adjacency (miud1izi) and store goal sjuejzxf facts.

        Parameters:
        - task: an instance of Task that contains .goals and .static
        """
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency map for location nodes from miud1izi facts.
        # We treat the miud1izi relation as undirected (add both directions).
        self.adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "miud1izi" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

        # Extract goal mapping: object -> desired location for sjuejzxf goals.
        self.goal_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "sjuejzxf" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                self.goal_map[obj] = loc

        # Cache for BFS distances between locations to speed up repeated queries.
        # Keys are tuples (loc_from, loc_to) -> integer distance.
        self._dist_cache = {}

    def _bfs_distance(self, start, goal):
        """
        Compute shortest path (in number of edges) between start and goal on
        the adjacency graph self.adj. Uses BFS and caches results.

        Returns:
            int: number of hops (>=0) if reachable, otherwise a finite large
                 fallback _UNREACHABLE_DISTANCE.
        """
        if start == goal:
            return 0
        key = (start, goal)
        if key in self._dist_cache:
            return self._dist_cache[key]

        if start not in self.adj or goal not in self.adj:
            # If either node is unknown in the static graph, fallback.
            self._dist_cache[key] = self._UNREACHABLE_DISTANCE
            return self._UNREACHABLE_DISTANCE

        # Standard BFS
        q = deque()
        q.append((start, 0))
        visited = {start}
        while q:
            node, dist = q.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr == goal:
                    self._dist_cache[key] = dist + 1
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, dist + 1))

        # No path found: use fallback finite distance.
        self._dist_cache[key] = self._UNREACHABLE_DISTANCE
        return self._UNREACHABLE_DISTANCE

    def __call__(self, node):
        """
        Compute the heuristic value for a search node.

        node.state is a frozenset of strings like '(pred arg1 arg2)'.

        Returns:
            int: estimated number of actions to achieve all sjuejzxf goals.
        """
        state = node.state

        # Quick goal check: if all goal facts already in the state, return 0.
        if self.goals <= state:
            return 0

        # Helper: find a fact matching predicate and first argument equals obj.
        def find_fact(predicate, obj):
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == predicate and len(parts) >= 2 and parts[1] == obj:
                    return parts
            return None

        total_cost = 0

        for obj, goal_loc in self.goal_map.items():
            # If goal already satisfied, contribute 0.
            goal_fact = f"(sjuejzxf {obj} {goal_loc})"
            if goal_fact in state:
                continue

            # Determine object's current status.
            # 1) At a location?
            sj = find_fact("sjuejzxf", obj)
            if sj is not None and len(sj) >= 3:
                cur_loc = sj[2]
                # Cost is shortest path distance between cur_loc and goal_loc.
                move_cost = self._bfs_distance(cur_loc, goal_loc)
                total_cost += move_cost
                continue

            # 2) Inside a vehicle? (un606moq OBJ VEH)
            un = find_fact("un606moq", obj)
            if un is not None and len(un) >= 3:
                vehicle = un[2]
                # We will need one unload action to produce (sjuejzxf obj VEH_LOC)
                unload_cost = 1

                # Find the vehicle's current location
                veh_pos = find_fact("sjuejzxf", vehicle)
                if veh_pos is not None and len(veh_pos) >= 3:
                    veh_loc = veh_pos[2]
                    move_cost = self._bfs_distance(veh_loc, goal_loc)
                    total_cost += unload_cost + move_cost
                else:
                    # Vehicle location unknown in this state representation:
                    # assign unload + large fallback to keep heuristic finite.
                    total_cost += unload_cost + self._UNREACHABLE_DISTANCE
                continue

            # 3) Neither at a location nor inside a vehicle (unexpected but possible).
            # Assign a conservative finite cost.
            total_cost += self._UNREACHABLE_DISTANCE

        return total_cost