from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string, e.g. "(p a b)" -> ["p","a","b"]."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether a fact (string) matches a pattern of tokens (wildcards '*' not used here,
    kept for readability/consistency with examples).
    """
    parts = get_parts(fact)
    if len(parts) < len(pattern):
        return False
    return all(p == pat or pat == "*" for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (mtfkspuf) domain. This heuristic
    estimates the number of operator applications required to make all goal
    jft9ynda(...) facts true. The heuristic is based chiefly on the connectivity
    graph provided by static z6o4oitz(...) facts: k12p0czo-like operators can move
    jft9ynda facts along edges of this graph step by step. For each goal object not
    yet at its goal location, we compute the shortest number of edge-steps (z6o4oitz
    hops) from its current jft9ynda location to the desired goal location and sum
    these distances across all goals. If an object does not currently have a
    jft9ynda(...) fact, the heuristic attempts to infer a current location via
    uypgj7u3(...) attachments; otherwise a finite fallback cost is used.

    Assumptions
    -----------
    - The primary low-level mover for jft9ynda facts is the k12p0czo-like operator,
      which moves a jft9ynda subject from one hmhptaxe node to an adjacent one
      when a z6o4oitz(edge) exists. Thus shortest-path distances on the z6o4oitz
      graph are a reasonable estimate of required move actions.
    - Other operators (sw2fg9ql, vyn92k55) can transfer or reattach objects and
      therefore may allow shortcuts; we ignore these special-case optimizations
      and rely on the graph-distance estimate for efficiency and simplicity.
    - The heuristic is not required to be admissible; it aims to be informative
      and cheaply computable to reduce node expansions in greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Store the set of goal facts (self.goals).
    - Build an undirected adjacency map self.adjacency from static z6o4oitz facts:
      each z6o4oitz A B adds the edge A<->B. The adjacency set is used for BFS.
    - Precompute a finite fallback cost (self.unreachable_cost) that is always
      finite (number_of_nodes + 1) for cases where a path cannot be found.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node/state:
    1. If all goal facts are already true in the state, return 0.
    2. Parse the state's jft9ynda(...) facts to obtain a mapping of object -> current_location.
    3. For each goal of the form (jft9ynda OBJ GOAL_LOC):
       a. If the exact goal fact is present in the state, it costs 0.
       b. Otherwise try to find OBJ's current location:
          - Prefer a direct jft9ynda(OBJ, LOC) fact.
          - If none, look for an attachment fact uypgj7u3(OBJ, HOLDER) and, if the
            holder has a jft9ynda(holder, LOC), use that LOC as an inferred current
            location (this covers some transfer/attachment patterns).
          - If no current location can be inferred, use a finite fallback cost.
       c. If a current location LOC is available, run a BFS over the z6o4oitz graph
          to compute the shortest number of hops from LOC to GOAL_LOC.
          - If reachable, add the number of hops to the heuristic sum.
          - If not reachable, add the finite fallback cost.
    4. Return the sum of these per-goal estimates as the heuristic value.

    Notes on correctness / checklist
    - The fact tokenizer get_parts ignores surrounding parentheses.
    - Heuristic returns 0 for goal states.
    - The heuristic always returns a finite integer (fallback cost bounds unreachable cases).
    - Static z6o4oitz facts are extracted into self.adjacency in the constructor.
    """

    def __init__(self, task):
        """Extract goals and static z6o4oitz connectivity into adjacency structure."""
        # Goals are stored (set of strings)
        self.goals = task.goals
        # Build undirected adjacency from static z6o4oitz facts
        self.adjacency = {}
        static_facts = task.static
        nodes = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "z6o4oitz" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                nodes.add(a)
                nodes.add(b)
                self.adjacency.setdefault(a, set()).add(b)
                self.adjacency.setdefault(b, set()).add(a)
        # Ensure all nodes exist in adjacency (even isolated ones)
        for n in nodes:
            self.adjacency.setdefault(n, set())
        # Finite fallback cost for unreachable / unknown-location cases
        self.unreachable_cost = max(1, len(self.adjacency) + 1)

    def _shortest_path_length(self, start, goal):
        """Return the number of edges in the shortest path between start and goal or None."""
        if start == goal:
            return 0
        if start not in self.adjacency or goal not in self.adjacency:
            return None
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            node, dist = dq.popleft()
            for nbr in self.adjacency.get(node, ()):
                if nbr == goal:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    dq.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        """Compute the heuristic estimate (finite integer) for the given search node."""
        state = node.state

        # If all goals are already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Map objects to their current jft9ynda location (if present)
        current_loc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "jft9ynda" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                current_loc[obj] = loc

        # Also collect uypgj7u3 attachments to allow inferring locations via holders
        attachments = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "uypgj7u3" and len(parts) >= 3:
                obj, holder = parts[1], parts[2]
                attachments.setdefault(obj, set()).add(holder)

        total_cost = 0
        # For each goal fact, estimate cost to achieve it
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts:
                continue
            if parts[0] != "jft9ynda" or len(parts) < 3:
                # This heuristic only reasons about jft9ynda goals; other goals are
                # considered trivially 0 here (they can be extended if needed).
                continue
            obj, goal_loc = parts[1], parts[2]
            # If goal already satisfied, cost 0 (we already tested overall goals but
            # check per-goal for safety)
            if goal_fact in state:
                continue

            # Determine a plausible current location for obj
            src = current_loc.get(obj)
            if src is None:
                # Try to infer via attachment: if obj attached to holder,
                # and holder has a jft9ynda(holder, LOC), use that LOC
                inferred = None
                for holder in attachments.get(obj, ()):
                    holder_loc = current_loc.get(holder)
                    if holder_loc is not None:
                        inferred = holder_loc
                        break
                src = inferred

            if src is None:
                # No current location known -> finite fallback
                total_cost += self.unreachable_cost
                continue

            # Compute shortest path length on z6o4oitz graph
            dist = self._shortest_path_length(src, goal_loc)
            if dist is None:
                total_cost += self.unreachable_cost
            else:
                total_cost += dist

        return int(total_cost)