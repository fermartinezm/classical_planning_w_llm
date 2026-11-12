from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return list of tokens inside a fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches a pattern of tokens (supports '*' wildcards).
    Example: match("(at a b)", "at", "*", "b") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "spanner" (secret_domain) domain.
    The heuristic estimates how many actions are (approximately) required to
    achieve all goals of the form (tightened-2 <locatable> <location>).

    Assumptions
    -----------
    - Goals are only of the form (tightened-2 <locatable> <location>) (as in
      the provided instances).
    - Nuts that can be used to produce a goal are those that currently have
      a (tightened <nut> <locatable>) fact in the state or may be moved along
      the static "loose" adjacency chain.
    - Changing a nut's loose-2 location (the nut's target location for
      tightened-2 production) can be done in one action (tighten_nut),
      because tighten_nut's precondition checks only the current loose-2
      and whether an at-2 location is available.
    - Transferring a tightened relation from one locatable to an adjacent
      locatable (along the static "loose" edges) is approximated by a unit
      cost per edge (there are various operators that move tightened along
      adjacent locatables; we ignore link-flipping dynamics and assume the
      static adjacency indicates feasibility).
    - The final action that produces (tightened-2 <locatable> <location>)
      (e.g., walk / pickup_spanner variants) has cost 1.
    - This heuristic is NOT required to be admissible; it aims to be
      informative and efficiently computable.

    Heuristic Initialization
    ------------------------
    The constructor extracts static facts and goals from the task and builds:
    - self.goals: list of (locatable, location) pairs that must become true.
    - self.loose_edges: static adjacency set/list derived from facts of the
      form "(loose <child> <parent>)". For the movement of tightened facts,
      an edge from parent -> child is considered (because operators move a
      tightened relation from parent to child when loose(child,parent) holds).
    - self.at2_locations: set of locations that appear in static "(at-2 ...)"
      facts. These are the locations to which a nut's loose-2 can be set in
      one action (tighten_nut).
    The constructor therefore prepares the information needed for fast
    per-state heuristic computation.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state, the heuristic proceeds as follows for each goal
    (tightened-2 <goal_locatable> <goal_location>):

    1. If the goal fact is already present in the state, its cost is 0.

    2. Otherwise, consider candidate nuts (all nuts that occur in the state
       either as (tightened <nut> <locatable>) or as (loose-2 <nut> <location>)).
       For each candidate nut compute an approximate cost to realize the
       goal using that nut:
       a) cost_to_set_nut_location:
          - If the nut's current loose-2 location equals the goal_location,
            cost = 0.
          - Else, we assume one action (tighten_nut) can change loose-2 to
            match the goal_location (cost = 1), provided at-2 locations
            exist (we treat this as always possible if at-2 facts exist).
       b) cost_to_move_tightened_along_locatables:
          - If the nut is currently tightened at some locatable S, compute
            the shortest distance (number of edges) in the static adjacency
            graph from S to any locatable P that satisfies loose(goal_locatable, P)
            (i.e., a P such that the final action can use that P to produce
            tightened-2 for the goal_locatable). Each edge represents one
            operator that moves the tightened relation to an adjacent locatable.
          - If the nut is not currently tightened at any locatable in the
            state, we conservatively assume a small fixed penalty (we treat
            such nuts as less favorable candidates).
       c) final_action_cost = 1 (the action that produces tightened-2 once the
          nut is at the needed locatable and its loose-2 equals goal_location).
       d) candidate_cost = cost_to_set_nut_location + cost_to_move_tightened_along_locatables + final_action_cost

    3. The estimated cost for that goal is the minimum candidate_cost over all nuts.
       If no candidate nut is usable (unreachable), we fall back to a safe
       upper bound (1 + number of locatables) so the heuristic remains finite.

    4. The heuristic value for the state is the sum of the estimated costs for
       all unsatisfied tightened-2 goals. This is a relaxed, optimistic-cost
       style estimate (it does not fully account for resource contention or
       link-flag dynamics), but it is informative and cheap to compute.

    Notes on Correctness and Practicality
    -------------------------------------
    - The heuristic returns 0 exactly when all goal facts are present (the
      Task.goal_reached check would also identify that).
    - The heuristic always returns a finite integer (we bound unreachable
      candidates with a finite fallback).
    - The implementation computes shortest-path distances on the static
      adjacency graph with BFS (fast for the size of instances observed).
    """

    def __init__(self, task):
        # Goals: store as list of (locatable, location) pairs for tightened-2 goals.
        self.goals = []
        for g in task.goals:
            parts = get_parts(g)
            if parts and parts[0] == "tightened-2":
                _, locatable, location = parts
                self.goals.append((locatable, location))

        # Static facts processing.
        static_facts = task.static

        # Build static loose edges: for each "(loose A B)" create edge B -> A
        # because operators can move "tightened" from B to A when loose(A,B) holds.
        self.loose_edges = {}  # adjacency: node -> set(neighbors)
        for fact in static_facts:
            if match(fact, "loose", "*", "*"):
                parts = get_parts(fact)
                child, parent = parts[1], parts[2]
                self.loose_edges.setdefault(parent, set()).add(child)

        # Set of at-2 locations (useful for judging feasibility of tighten_nut)
        self.at2_locations = {get_parts(f)[1] for f in static_facts if match(f, "at-2", "*")}

        # A conservative upper bound used if no candidate nut is reachable.
        # Use number of locatables in loose graph + 5 to ensure finite heuristic.
        # Count locatables from static loose and edges.
        locs = set()
        for a, nbrs in self.loose_edges.items():
            locs.add(a)
            locs.update(nbrs)
        self.max_fallback = max(1, len(locs) + 5)

    def _bfs_distance(self, start, targets):
        """
        BFS on the static adjacency graph self.loose_edges.
        Edges represent one operator that can move a tightened relation from
        a node to its neighbor (parent -> child as stored).
        Returns minimal number of edges from start to any node in targets.
        If start is already in targets returns 0.
        If start is None or targets unreachable returns None.
        """
        if start is None:
            return None
        if start in targets:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self.loose_edges.get(node, ()):
                if nbr in visited:
                    continue
                if nbr in targets:
                    return dist + 1
                visited.add(nbr)
                queue.append((nbr, dist + 1))
        return None

    def __call__(self, node):
        # If all goals already satisfied return 0.
        state = node.state
        if all(f"(tightened-2 {loc} {loc2})" in state for (loc, loc2) in self.goals):
            return 0

        # Extract current nut information from the state.
        # tightened_map: nut -> locatable (where the nut is currently tightened)
        tightened_map = {}
        # loose2_map: nut -> location (nut's loose-2 location)
        loose2_map = {}

        for fact in state:
            if match(fact, "tightened", "*", "*"):
                parts = get_parts(fact)
                nut, locatable = parts[1], parts[2]
                tightened_map[nut] = locatable
            elif match(fact, "loose-2", "*", "*"):
                parts = get_parts(fact)
                nut, location = parts[1], parts[2]
                loose2_map[nut] = location

        # Candidate nuts are those that appear in either mapping or in static facts
        # (we conservatively consider union of nuts observed in state).
        candidate_nuts = set(tightened_map.keys()) | set(loose2_map.keys())

        total_estimate = 0

        for (goal_locatable, goal_location) in self.goals:
            goal_fact = f"(tightened-2 {goal_locatable} {goal_location})"
            if goal_fact in state:
                continue  # already satisfied

            # Determine the set of locatables P such that static loose(goal_locatable, P) holds.
            possible_parent_locatables = {
                get_parts(f)[2]
                for f in self.loose_edges.get(get_parts(f)[0], set())
                for f in []  # placeholder to keep comprehension form consistent; replaced below
            }
            # The above comprehension is vestigial; instead construct possible_parent_locatables by scanning static edges.
            possible_parent_locatables = {
                parent  # Actually we need children P such that loose(goal_locatable, P) in static
                for parent, children in self.loose_edges.items()
                for child in children
                if child == goal_locatable  # but definition: static loose child parent => loose(child,parent)
            }
            # The previous attempt is brittle; simpler: scan all static facts for loose(goal_locatable, P)
            # But we don't have static facts here; we have self.loose_edges which stores parent->children (parent -> set(children))
            # We want all P such that loose(goal_locatable, P) is true -> that is P appears as second arg in "(loose goal_locatable P)"
            # The mapping we stored was parent -> set(children) where children are those that referenced parent.
            # In other words, if we find parent such that goal_locatable in children, then P = parent? Let's reconstruct correctly:
            possible_parent_locatables = {
                parent for parent, children in self.loose_edges.items() if goal_locatable in children
            }

            # If there are no static parents (rare), we fall back to assuming any locatable could be used (conservative).
            if not possible_parent_locatables:
                # Use all locatables known in the static graph.
                possible_parent_locatables = set(self.loose_edges.keys()) | {c for s in self.loose_edges.values() for c in s}

            # For this goal, compute minimal cost over all candidate nuts.
            best_cost = None
            for nut in candidate_nuts:
                # 1) cost to set nut's loose-2 to goal_location.
                if loose2_map.get(nut) == goal_location:
                    cost_set_loose2 = 0
                else:
                    # If there is at least one at-2 location, we assume one action suffices.
                    cost_set_loose2 = 1 if self.at2_locations else self.max_fallback

                # 2) cost to move the tightened relation from its current locatable to
                #    one of the possible_parent_locatables.
                start_loc = tightened_map.get(nut)  # may be None
                distance = self._bfs_distance(start_loc, possible_parent_locatables)
                if distance is None:
                    # If nut not tightened or unreachable, assign a conservative penalty
                    # larger than a simple move but still finite.
                    move_penalty = 2 + (0 if start_loc is None else 0)
                else:
                    move_penalty = distance

                # 3) final action (walk/pickup_spanner) to produce tightened-2
                final_action = 1

                candidate_cost = cost_set_loose2 + move_penalty + final_action

                if best_cost is None or candidate_cost < best_cost:
                    best_cost = candidate_cost

            # If we found no candidate nut at all (empty candidate_nuts), fallback to max_fallback.
            if best_cost is None:
                best_cost = self.max_fallback

            total_estimate += best_cost

        # Ensure heuristic returns integer finite value.
        return int(total_estimate)