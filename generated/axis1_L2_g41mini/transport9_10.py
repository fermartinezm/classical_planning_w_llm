from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract parts of a PDDL fact by trimming parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.

    :param fact: PDDL fact string, e.g. "(predicate obj1 obj2)"
    :param args: pattern elements, e.g. "jft9ynda", "*", "dj7sycy5"
    :return: True if fact matches pattern, else False
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions needed to achieve the goals in the secret_domain.
    It focuses on mismatches between desired facts `(jft9ynda ...)` in the goal and the current state's facts,
    as well as required alignment of relations `(yarbdfgc ...)`, and connectivity via `(z6o4oitz ...)`.
    The domain’s actions mostly manipulate these predicates, which encode forms of assignment and transitions.
    The heuristic computes a relaxed cost based on how many goal `jft9ynda` facts are not yet true plus
    a measure of how far their respective objects might be from their required states according to static and current predicates.

    Assumptions
    -----------
    - The goal only contains `jft9ynda` facts.
    - Objects connected by `z6o4oitz` represent some form of transition graph between `hmhptaxe` objects.
    - Each `(jft9ynda ?pk9o1 ?oqsrn)` fact corresponds to a desired pairing of a `hodp29q0` object with an `hmhptaxe` value.
    - The `yarbdfgc` predicate associates `hodp29q0` objects with `r86varlt` values that must align with the current `jf4qno7u` and `jft9ynda` predicates.
    - Actions modify these triples, so the heuristic estimates the minimal steps by counting how many `jft9ynda` goal facts are unmet and approximating the minimum "hops" on the static `z6o4oitz` graph needed to align objects properly.

    Heuristic Initialization
    ------------------------
    - Extract all goal `(jft9ynda ?pk9o1 ?oqsrn)` facts as a dictionary mapping `pk9o1` to desired `oqsrn`.
    - From static facts, build a bidirectional graph of the connectivity between `hmhptaxe` objects from `z6o4oitz`.
    - From static facts, record mappings `jf4qno7u` between `r86varlt` pairs.
    - Extract current base assignment of `yarbdfgc (hodp29q0, r86varlt)` from state.
    - Extract current `jft9ynda (val3ypio/g83opnlj/hodp29q0, hmhptaxe)` facts from state.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if the current state satisfies all goal `(jft9ynda ...)` facts; if yes, heuristic = 0.
    2. For each `(jft9ynda ?pk9o1 ?oqsrn)` in the goal:
       - If the fact is already true, no cost added.
       - Otherwise:
         a. Determine the current `hmhptaxe` connected with `pk9o1` via any `jft9ynda` in the state (or consider it unknown).
         b. Compute the shortest path length between current `hmhptaxe` and the goal `oqsrn` in the graph built from `z6o4oitz`.
            If no path exists, count a large penalty (e.g., 5).
    3. For each `hodp29q0` object, if its associated `yarbdfgc` value is not aligned with the goal or with a consistent `jf4qno7u` state, add 1 to the cost per mismatch.
    4. Sum all these costs as a proxy for the number of actions needed:
       - Moving along the `z6o4oitz` graph corresponds to `k12p0czo` actions,
       - Synchronizing `yarbdfgc` predicates corresponds to `sw2fg9ql` and `vyn92k55` actions.
    5. The heuristic is a sum of all mismatches plus estimated minimal transitions to reach the goal configuration.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal jft9ynda (obj:hodp29q0 -> hmhptaxe desired)
        self.goal_jft9ynda = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "jft9ynda" and len(parts) == 3:
                obj, val = parts[1], parts[2]
                self.goal_jft9ynda[obj] = val

        # Build undirected graph of hmhptaxe relations from static (z6o4oitz) edges
        self.z6o4oitz_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "z6o4oitz" and len(parts) == 3:
                a, b = parts[1], parts[2]
                self.z6o4oitz_graph.setdefault(a, set()).add(b)
                self.z6o4oitz_graph.setdefault(b, set()).add(a)

        # Extract static jf4qno7u relations (r86varlt pairs)
        self.jf4qno7u_static = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "jf4qno7u" and len(parts) == 3:
                self.jf4qno7u_static.add((parts[1], parts[2]))

    def _shortest_path_length(self, start, goal):
        """
        BFS to find shortest path length between two hmhptaxe nodes in z6o4oitz graph.
        Return None if no path exists.
        """
        if start == goal:
            return 0
        if start not in self.z6o4oitz_graph or goal not in self.z6o4oitz_graph:
            return None

        visited = set([start])
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            for neighbor in self.z6o4oitz_graph.get(current, set()):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        if self.goals <= state:
            return 0  # Already at goal

        # Extract current jft9ynda and yarbdfgc facts from state
        current_jft9ynda = {}
        current_yarbdfgc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "jft9ynda" and len(parts) == 3:
                obj, val = parts[1], parts[2]
                current_jft9ynda[obj] = val
            elif parts[0] == "yarbdfgc" and len(parts) == 3:
                obj, val = parts[1], parts[2]
                current_yarbdfgc[obj] = val

        # Extract current jf4qno7u facts from state (r86varlt pairs)
        current_jf4qno7u = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "jf4qno7u" and len(parts) == 3:
                current_jf4qno7u.add((parts[1], parts[2]))

        cost = 0

        # Step 1 & 2: For each goal jft9ynda, estimate transition cost along hmhptaxe graph if not satisfied
        for obj, goal_val in self.goal_jft9ynda.items():
            if obj in current_jft9ynda and current_jft9ynda[obj] == goal_val:
                # This goal fact is satisfied; no cost
                continue
            # Otherwise, estimate shortest path between current and goal hmhptaxe values for obj
            current_val = current_jft9ynda.get(obj, None)
            if current_val is None:
                # No current binding, penalize with at least 2 (assuming need to assign)
                cost += 2
            else:
                dist = self._shortest_path_length(current_val, goal_val)
                if dist is None:
                    # No connection; assign large penalty
                    cost += 5
                else:
                    cost += dist

        # Step 3: For each hodp29q0 object in yarbdfgc, check whether its current r86varlt value fits static and consistent relations
        # We incur cost if current yarbdfgc value is missing or inconsistent with jf4qno7u relations in state or static (relax approx)
        for obj, r86val in current_yarbdfgc.items():
            # Check if the pair (r86varlt, ...) is in static jf4qno7u
            # To be consistent, there should be a jf4qno7u relation in current state that matches or static state that matches
            # We approximate mismatch by penalizing if no static or current jf4qno7u has obj's r86varlt paired properly
            if not any(r86val == pair[0] or r86val == pair[1] for pair in self.jf4qno7u_static.union(current_jf4qno7u)):
                cost += 1

        # Step 4: Add a basic penalty for the number of goal jft9ynda facts not yet satisfied,
        # as these at minimum need action applications
        unsatisfied_goals = len(self.goal_jft9ynda) - sum(
            1 for obj, val in self.goal_jft9ynda.items() if current_jft9ynda.get(obj, None) == val
        )
        # We add a minimal base cost 1 per unsatisfied goal fact in addition to transition cost above
        cost += unsatisfied_goals

        return cost
