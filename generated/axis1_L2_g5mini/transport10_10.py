from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract components of a PDDL fact string by removing surrounding
    parentheses and splitting on whitespace.

    Example:
      "(jft9ynda agent loc)" -> ["jft9ynda", "agent", "loc"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (mtfkspuf) domain.
    The heuristic estimates how many (high-level) moves are required to
    satisfy all goal facts of the form (jft9ynda <entity> <location>).
    It relies on the static adjacency relation z6o4oitz between locations
    (hmhptaxe objects) to compute shortest-path distances (in hops).
    The heuristic is intentionally non-admissible but aims to be informative
    and cheap to compute for greedy best-first search.

    Assumptions
    -----------
    - Goals are of the form (jft9ynda <entity> <location>).
    - Entities that appear in jft9ynda facts may either:
        * have a direct jft9ynda fact in the current state (they are at a
          location), or
        * be "attached" to another object via uypgj7u3 (in which case the
          attached object's jft9ynda location is treated as the entity's
          effective location).
    - Movement between locations is modeled by hops along z6o4oitz edges.
      Each hop corresponds to one elementary move (k12p0czo-like action).
    - If an entity is currently attached (uypgj7u3 present) then one
      additional action is required to detach it (sw2fg9ql-like action)
      to produce the jft9ynda fact required by the goal.
    - If no path exists between two locations according to static adjacency,
      a finite but large fallback cost is used so the heuristic remains finite.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - store the set of goal facts (self.goals),
    - parse the static facts (task.static) and build an adjacency graph of
      locations using the z6o4oitz predicate. The graph is treated as
      undirected for shortest-path computations.
    - no other domain-specific static preprocessing is required for this
      heuristic.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state (node.state) we compute the heuristic as follows:

    1. If the current state already entails all goals, return 0.

    2. Parse the current state's relevant relation facts:
       - jft9ynda facts: map entity -> location
       - uypgj7u3 facts: map entity -> attached_object
         (when an entity is attached, it typically does not have a
          jft9ynda fact itself)

    3. For every goal fact (jft9ynda entity goal_location):
       a) Determine the entity's current effective location:
          - If the state contains (jft9ynda entity loc), use loc.
          - Else, if the state contains (uypgj7u3 entity obj) and the
            state contains (jft9ynda obj loc_obj), use loc_obj and mark
            the entity as attached (so a detach action will be required).
          - Else, if no information is available, assign a fallback high cost.
       b) Compute the shortest-path distance (in hops) between the current
          effective location and the goal_location using BFS on the static
          adjacency graph built from z6o4oitz facts.
       c) The cost for this goal is:
          distance_in_hops + (1 if entity is currently attached else 0)
          (the +1 models the required detach action that produces the
           jft9ynda fact for the entity).
       d) If no path exists in the adjacency graph, use a finite fallback
          (configured below) so the heuristic remains finite.

    4. Sum the per-goal costs and return the result as an integer estimate.

    Note: This heuristic is purposely lightweight and optimistic about
    certain preconditions (e.g., existence of particular yarbdfgc/jf4qno7u
    facts that some actions require). Its goal is to provide a useful
    guidance signal (distance + attach/detach overhead) while being
    computationally cheap.
    """

    def __init__(self, task):
        """Extract goals and static adjacency information from the task."""
        self.goals = task.goals  # goal facts (set of strings)
        static_facts = task.static

        # Build an undirected adjacency map for locations (hmhptaxe) using z6o4oitz.
        self.adj = {}  # location -> set(neighbor_locations)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "z6o4oitz" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

        # A fallback cost if no path exists between two locations.
        # This keeps the heuristic finite while penalizing disconnected cases.
        self.fallback_distance = max(10, len(self.adj) + 5)

    def _bfs_distance(self, start, target):
        """
        Compute shortest-path distance (in hops) between start and target
        in the adjacency graph. Returns an integer distance or None if
        either node is unknown or unreachable.
        """
        if start == target:
            return 0
        if start not in self.adj or target not in self.adj:
            return None
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, d = queue.popleft()
            for nb in self.adj.get(node, ()):
                if nb == target:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, d + 1))
        return None

    def __call__(self, node):
        """
        Compute and return the heuristic estimate (integer) for the given node.
        """
        state = node.state

        # 2) Heuristic is 0 for goal states
        if self.goals <= state:
            return 0

        # Parse relevant state facts.
        # Map entity -> location for jft9ynda facts.
        jft9ynda_map = {}
        # Map entity -> attached_object for uypgj7u3 facts (entity attached to some object)
        attached_map = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "jft9ynda" and len(parts) >= 3:
                entity, loc = parts[1], parts[2]
                jft9ynda_map[entity] = loc
            elif pred == "uypgj7u3" and len(parts) >= 3:
                entity, obj = parts[1], parts[2]
                attached_map[entity] = obj

        total_cost = 0

        # Process each goal fact (we only consider goals asking for jft9ynda).
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            if pred != "jft9ynda" or len(parts) < 3:
                # This heuristic focuses on jft9ynda goals; non-matching goals are
                # given no cost here (could be extended if needed).
                continue

            entity, goal_loc = parts[1], parts[2]

            # Determine effective current location and whether the entity is attached.
            attached = False
            current_loc = None

            if entity in jft9ynda_map:
                current_loc = jft9ynda_map[entity]
                attached = False
            elif entity in attached_map:
                # entity attached to some object; find object's location
                obj = attached_map[entity]
                if obj in jft9ynda_map:
                    current_loc = jft9ynda_map[obj]
                    attached = True
                else:
                    # attached object's location unknown -> fallback
                    current_loc = None
                    attached = True
            else:
                # No info about entity location -> try to treat as unknown
                current_loc = None
                attached = False

            # Compute distance
            if current_loc is None:
                # No reliable location info -> use fallback
                dist = self.fallback_distance
            else:
                d = self._bfs_distance(current_loc, goal_loc)
                if d is None:
                    dist = self.fallback_distance
                else:
                    dist = d

            # If the entity is attached, we will likely need one extra action to
            # detach to obtain the required jft9ynda(<entity> <goal_loc>) fact.
            detach_cost = 1 if attached else 0

            total_cost += dist + detach_cost

        # Ensure returned heuristic is an integer and finite.
        return int(total_cost)