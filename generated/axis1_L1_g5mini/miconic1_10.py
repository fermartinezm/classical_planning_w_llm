from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (ud3uf11z).
    The domain models a single movable marker `omggbn9z` that can be moved
    along `acovodrf` edges between location objects (t4rdxh32). Two predicates
    relate persons (nxbd1hc3) to locations:
      - kut2ksnk(person, loc): a location where we can "collect" (diwlmgg9) a person,
      - klw11kjf(person, loc): a location where we can "finalize" (y7nik9q2) a person.
    The general plan to establish (a7sb56om person) is:
      1. Ensure `f2xo7hb6 person` (achieved by diwlmgg9 at a location satisfying kut2ksnk and where the marker is).
      2. Move the marker to a location satisfying klw11kjf for that person (if needed).
      3. Apply y7nik9q2 to obtain a7sb56om(person).
    Movement of the marker is achieved by the two move operators (qs8pdrqj / i0veo5kv) along edges described by acovodrf facts.

    Assumptions
    -----------
    - The planner represents facts as strings like "(predicate arg1 arg2)".
    - The `acovodrf` relations represent directed facts; the domain provides two movement
      operators that together allow movement along an edge if acovodrf is present in either direction.
      For heuristic purposes an edge is treated as usable in both directions if either
      acovodrf(X,Y) or acovodrf(Y,X) is present.
    - The marker `omggbn9z` may be present at exactly one location in most instances,
      but the heuristic handles multiple positions conservatively.
    - Some kut2ksnk / klw11kjf facts may appear either in the static facts or the current state;
      both are considered.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal persons from the task goals (all goals are expected to be
      of the form "(a7sb56om person)").
    - Extract the static acovodrf relations and build a static adjacency map (undirected
      edges for heuristic search).
    - Store the static facts set for quick lookup.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic estimates the number of actions required to achieve
    all remaining (a7sb56om person) goals. It is computed as the sum of per-person estimates.
    For each person that does not yet satisfy the goal:
      1. Combine dynamic state facts with static facts to find:
         - Locations L_kut where (kut2ksnk person L_kut) holds.
         - Locations L_klw where (klw11kjf person L_klw) holds.
         - Whether (f2xo7hb6 person) already holds.
         - Current marker positions Om = {locations | (omggbn9z loc)}.
      2. Build an adjacency graph by combining static adjacency (from static facts)
         with any acovodrf facts present in the current state (to handle cases where
         movement edges are not purely static).
      3. Compute shortest move-distances (number of marker-move actions) via BFS:
         - If f2xo7hb6 is already true:
             cost_person = min_distance(Om, L_klw) + 1
             (1 for y7nik9q2)
         - Otherwise:
             For each candidate collection location k in L_kut:
                 d1 = min_distance(Om, {k})   # moves to reach k to perform diwlmgg9
                 d2 = min_distance({k}, L_klw) # moves from k to a klw location to perform y7nik9q2
             cost_person = min_over_k (d1 + d2) + 2
             (+1 for diwlmgg9 and +1 for y7nik9q2)
      4. If any required distance is unreachable (no path), that person is considered
         currently unsolvable from this state. The heuristic returns a large finite
         value to reflect that (we use a configurable large constant).
    Notes:
    - The heuristic is not admissible (it may double-count movement of the marker when
      multiple persons could be served in a single sweep), but it is informative: it
      captures the need to move the marker to collection locations and then to finalization
      locations, plus the two domain actions required per person (diwlmgg9 and y7nik9q2).
    - The heuristic returns 0 iff the state already satisfies all goals.
    """

    # A large finite penalty for unreachable subgoals (keeps heuristic finite).
    UNREACHABLE_COST = 9999

    def __init__(self, task):
        # Goals stored as given (set of goal fact strings).
        self.goals = task.goals

        # Static facts (frozenset of fact strings).
        self.static = task.static if task.static is not None else frozenset()

        # Identify goal persons: all goals of form (a7sb56om PERSON)
        self.goal_persons = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "a7sb56om" and len(parts) >= 2:
                self.goal_persons.append(parts[1])

        # Build static adjacency map from static acovodrf facts.
        # For heuristic purposes an edge exists between X and Y if acovodrf X Y or acovodrf Y X appears.
        self.static_adjacency = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "acovodrf" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                # add directed edge a -> b
                self.static_adjacency.setdefault(a, set()).add(b)
                # also add reverse to allow movement along either direction if present
                self.static_adjacency.setdefault(b, set())  # ensure key exists

        # To also capture reverse edges declared separately (we will combine in __call__ with state edges).

    def __call__(self, node):
        # If all goals already satisfied, return 0.
        if self.goals <= node.state:
            return 0

        # Combine state and static for lookup of person-location relations.
        combined = set(node.state) | set(self.static)

        # Extract current marker positions (omggbn9z facts).
        om_positions = {get_parts(f)[1] for f in combined if get_parts(f)[0] == "omggbn9z"}

        # If there is no marker anywhere, likely unsolvable; return large cost.
        if not om_positions:
            return self.UNREACHABLE_COST

        # Helper: collect locations satisfying a predicate for a given person
        def locs_for(predicate, person):
            return {get_parts(f)[2] for f in combined if get_parts(f)[0] == predicate and len(get_parts(f)) >= 3 and get_parts(f)[1] == person}

        # Precompute adjacency for this state by starting with static adjacency and adding any acovodrf from combined.
        adjacency = {k: set(v) for k, v in self.static_adjacency.items()}
        # ensure keys and neighbor sets exist for nodes that may appear only in state facts
        # Add edges for acovodrf facts found in combined (state or static)
        for fact in combined:
            parts = get_parts(fact)
            if parts and parts[0] == "acovodrf" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set())  # ensure reverse key exists
        # As described in the docstring, movement along an edge is allowed if acovodrf(a,b) OR acovodrf(b,a).
        # Therefore we augment adjacency to be undirected for any pair that appears in either direction.
        # Build an undirected view:
        undirected = {}
        for u, nbrs in adjacency.items():
            undirected.setdefault(u, set())
            for v in nbrs:
                undirected.setdefault(v, set())
                undirected[u].add(v)
                undirected[v].add(u)

        # BFS shortest path from any of start_set to any element of target_set on undirected graph.
        def shortest_distance(start_set, target_set):
            if not target_set:
                return None  # no targets available
            # Quick check if any start intersects target
            if start_set & target_set:
                return 0
            visited = set()
            q = deque()
            for s in start_set:
                visited.add(s)
                q.append((s, 0))
            while q:
                node_name, dist = q.popleft()
                # neighbors
                for nbr in undirected.get(node_name, ()):
                    if nbr in visited:
                        continue
                    if nbr in target_set:
                        return dist + 1
                    visited.add(nbr)
                    q.append((nbr, dist + 1))
            return None  # unreachable

        total_cost = 0

        # For each person that still needs a7sb56om, compute an estimate.
        for person in self.goal_persons:
            goal_fact = f"(a7sb56om {person})"
            if goal_fact in node.state:
                continue  # already satisfied

            # If f2xo7hb6 already present, only need to get marker to a klw location then y7nik9q2.
            has_f2 = f"(f2xo7hb6 {person})" in combined

            kut_locs = locs_for("kut2ksnk", person)
            klw_locs = locs_for("klw11kjf", person)

            # If there are no klw locations at all, impossible to finalize this person.
            if not klw_locs:
                return self.UNREACHABLE_COST

            if has_f2:
                d = shortest_distance(om_positions, klw_locs)
                if d is None:
                    return self.UNREACHABLE_COST
                # +1 for y7nik9q2
                total_cost += d + 1
            else:
                # Need to first reach some kut location, do diwlmgg9, then reach some klw and y7nik9q2.
                if not kut_locs:
                    # No collection locations -> cannot obtain f2xo7hb6
                    return self.UNREACHABLE_COST

                best = None
                # For each candidate kut location, compute d1 from current marker positions to kut,
                # and d2 from that kut to any klw location.
                for kut in kut_locs:
                    d1 = shortest_distance(om_positions, {kut})
                    if d1 is None:
                        continue
                    # after diwlmgg9 at 'kut', marker is still at kut (diwlmgg9 does not move marker),
                    # so distance to klw is from 'kut' node.
                    d2 = shortest_distance({kut}, klw_locs)
                    if d2 is None:
                        continue
                    cand = d1 + d2
                    if best is None or cand < best:
                        best = cand
                if best is None:
                    return self.UNREACHABLE_COST
                # +1 for diwlmgg9 and +1 for y7nik9q2
                total_cost += best + 2

            # Safety cap to avoid overflow in pathological cases
            if total_cost > self.UNREACHABLE_COST:
                return self.UNREACHABLE_COST

        return int(total_cost)