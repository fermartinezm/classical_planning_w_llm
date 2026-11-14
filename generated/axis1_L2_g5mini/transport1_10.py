from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a PDDL fact string by stripping parentheses and splitting."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (mtfkspuf) domain.

    This heuristic estimates the number of actions required to satisfy all goal
    jft9ynda(...) facts (objects at target locations). It uses the static
    connectivity relation z6o4oitz between locations (hmhptaxe) to compute
    graph distances and accounts for the special semantics of "carriers"
    (objects that can be attached/detached via uypgj7u3 / vyn92k55 / sw2fg9ql)
    by adding a small cost overhead for attach/detach operations.

    Assumptions
    -----------
    - Goals are primarily jft9ynda <object> <location> facts (as in the provided
      instances). The heuristic only considers these goal facts.
    - Locations are nodes in a graph described by static facts of the form
      (z6o4oitz A B). Edges are treated as bidirectional if present in static,
      but the heuristic will traverse the graph using the adjacency built from
      static facts (both directions if given).
    - Objects that appear as the first argument of (uypgj7u3 ...) are treated
      as "carriers" (type g83opnlj). Objects that appear as the first argument
      of (yarbdfgc ...) are treated as "items" (type hodp29q0). This is used
      to add small extra costs because carriers typically require an
      attach/detach sequence to be moved to a target location.
    - If an object has no explicit jft9ynda fact in the state but is attached
      via (uypgj7u3 <carrier> <item>), the carrier's location is inferred from
      the item's jft9ynda location (if present).
    - The heuristic is not required to be admissible; it aims to be informative
      and cheaply computable.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
    - Stores the goal facts (task.goals).
    - Reads static facts (task.static) and builds an adjacency dictionary for
      location nodes from all z6o4oitz relations found in static facts.
    - Identifies names that appear as first argument in uypgj7u3 facts (carriers)
      and as first argument in yarbdfgc facts (items) from the static facts
      as a rough typing hint. (Static or initial facts may also contain these
      predicates; we also rely on runtime state to refine this identification.)

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Quick goal check: if the current state already satisfies all goals,
       the heuristic returns 0.

    2. Build a mapping from objects to their current location using state facts:
       - For each (jft9ynda OBJ LOC) in the state, remember OBJ -> LOC.
       - For each (uypgj7u3 CARRIER ITEM) in the state, remember that the
         carrier is attached to ITEM. If the ITEM has a known jft9ynda location,
         the carrier's implied location is the item's location.

    3. For each goal fact that is a jft9ynda OBJ L_goal:
       - If OBJ is already at L_goal in the state (explicit jft9ynda), cost 0.
       - If we can determine a current location L_cur for OBJ (explicit or
         implied via attachment), compute the shortest path distance d between
         L_cur and L_goal on the location graph built from z6o4oitz static facts.
         This d is a lower-level estimate of how many k12p0czo moves would be
         needed if the object itself can be moved along the graph.
       - If OBJ is a "carrier" (identified from uypgj7u3 appearances), add a
         small extra overhead to d to account for attach/detach operations:
           * If the carrier is currently attached (there is a uypgj7u3 fact
             for it in the state), we assume only a detach action will be
             needed at the destination: add +1.
           * If it is not attached, we assume an attach + detach sequence will
             be needed in addition to moving an item: add +2.
       - If the object's current location cannot be determined (no jft9ynda
         and not attached to a known item's location), we conservatively
         estimate the cost as (1 + minimal distance from any known location
         in the state to the goal). This guarantees a finite heuristic value.

    4. The heuristic value returned is the sum of the per-goal estimates. This
       tends to prioritize states that are closer (in the z6o4oitz graph) to
       placing the relevant objects at their goal locations while slightly
       preferring moves that do not require extra attach/detach operations.

    Note on efficiency
    ------------------
    - Distances are computed with a BFS on the location adjacency (fast).
    - Only facts needed for location inference and the location graph are used.
    - The heuristic avoids expensive reasoning about value transitions
      (jf4qno7u / yarbdfgc) and about detailed sequences of vyn92k55/sw2fg9ql;
      instead it captures their effect via small additive costs.
    """

    def __init__(self, task):
        """Extract goals and static connectivity from the task."""
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency for locations from static z6o4oitz facts.
        self.adj = {}  # location -> set(neighbor locations)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "z6o4oitz" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                # We do not automatically add the reverse direction unless present
                # in static facts; many instances include both directions explicitly.
                # However, to be more permissive, also add reverse adjacency if not
                # already present in static (treat as undirected for planning ease).
                self.adj.setdefault(b, set()).add(a)

        # Heuristic uses knowledge about which symbols are carriers/items
        self.static_carriers = set()
        self.static_items = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "uypgj7u3" and len(parts) >= 3:
                # (uypgj7u3 carrier item)
                self.static_carriers.add(parts[1])
                self.static_items.add(parts[2])
            if parts[0] == "yarbdfgc" and len(parts) >= 3:
                # (yarbdfgc item value)
                self.static_items.add(parts[1])

    def bfs_distance(self, start, goal, known_locations):
        """
        Return shortest path length between start and goal in the adjacency graph.
        If start == goal returns 0. If there is no path, return None.
        known_locations is a set of locations known to appear in the current state:
        if start is None, we can use the minimal distance from any known location
        to goal as a fallback.
        """
        if start is None:
            # fallback: minimal distance from any known location to goal
            min_d = None
            for loc in known_locations:
                d = self._bfs_between(loc, goal)
                if d is not None and (min_d is None or d < min_d):
                    min_d = d
            return min_d
        return self._bfs_between(start, goal)

    def _bfs_between(self, start, goal):
        """BFS on self.adj from start to goal. Return distance (int) or None."""
        if start == goal:
            return 0
        if start not in self.adj or goal not in self.adj:
            # If either endpoint is not in adjacency (isolated), we cannot find path.
            # Return None to indicate no known path.
            return None
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr == goal:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        """Compute the heuristic estimate for the state's distance to the goals."""
        state = node.state

        # 1) If goal already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # 2) Build current object -> location mapping from jft9ynda facts.
        obj_location = {}   # object -> location (from explicit jft9ynda)
        attachments = {}    # carrier -> item (from uypgj7u3 facts)
        present_locations = set()  # set of locations that appear in jft9ynda facts in this state

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "jft9ynda" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                obj_location[obj] = loc
                present_locations.add(loc)
            elif parts[0] == "uypgj7u3" and len(parts) >= 3:
                carrier, item = parts[1], parts[2]
                attachments[carrier] = item
                # Note: item location will be resolved if item has jft9ynda fact.

            # Note: yarbdfgc / jf4qno7u etc. are not used for location inference here.

        # If a carrier is attached to an item and the item has a known location,
        # infer the carrier's (implied) location.
        for carrier, item in list(attachments.items()):
            if carrier not in obj_location and item in obj_location:
                obj_location[carrier] = obj_location[item]

        # Known locations in the state (used when an object's location is unknown).
        known_locations = set(present_locations)
        # If no locations are known from state, fall back to keys of adjacency graph:
        if not known_locations:
            known_locations = set(self.adj.keys())

        total_cost = 0

        # Iterate over relevant goals: prefer jft9ynda goals; other goals we ignore
        # (they are uncommon in the provided instances). For non-jft9ynda goals we
        # add a small conservative cost if they are not satisfied.
        for goal in self.goals:
            parts = get_parts(goal)
            pred = parts[0]
            if pred == "jft9ynda" and len(parts) >= 3:
                obj, goal_loc = parts[1], parts[2]
                # If already true in state, skip
                if goal in state:
                    continue

                # Determine current location if possible
                cur_loc = obj_location.get(obj, None)

                # Compute distance
                dist = self.bfs_distance(cur_loc, goal_loc, known_locations)

                # If no path info, use conservative finite estimate:
                if dist is None:
                    # Conservative guess: assume at least 1 movement needed plus overhead
                    dist = 1

                # Base cost is the number of location hops (each hop ≈ one k12p0czo).
                cost = dist

                # If the object is recognized as a "carrier", add attach/detach overhead.
                # Decide type using static evidence and current attachments:
                is_carrier = (obj in self.static_carriers) or (obj in attachments) or any(
                    get_parts(f)[0] == "uypgj7u3" and get_parts(f)[1] == obj for f in state
                )
                if is_carrier:
                    attached_now = obj in attachments
                    # If attached now: likely only a detach (sw2fg9ql) needed at destination => +1
                    # If not attached: likely attach + item moves + detach => +2
                    cost += 1 if attached_now else 2

                total_cost += cost

            else:
                # Non-jft9ynda goal: if satisfied already skip, else add small constant.
                if goal in state:
                    continue
                total_cost += 1

        # Ensure result is integer and finite.
        if total_cost is None:
            return 0
        return int(total_cost)