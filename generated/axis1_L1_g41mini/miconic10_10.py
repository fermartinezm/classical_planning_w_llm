from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a fact matches a given pattern with wildcards '*' allowed."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (ud3uf11z) domain.

    Summary
    -------
    This heuristic estimates the remaining number of actions to achieve the goal:
    making every goal fact `(a7sb56om ?p)` true for specific nxbd1hc3 objects (?p).

    Assumptions
    -----------
    - The ultimate goal predicates are only of the form `(a7sb56om ?praz5)`.
    - Each object `?p` of type nxbd1hc3 must have `(a7sb56om ?p)` true in the goal.
    - The domain models a two-step process for each such object:
        1) Achieving `(f2xo7hb6 ?p)` by applying `diwlmgg9` action (which requires `(kut2ksnk ?p ?w)` and `(omggbn9z ?w)`),
        2) Then achieving `(a7sb56om ?p)` by applying `y7nik9q2` action (which requires `(klw11kjf ?p ?w)`, `(omggbn9z ?w)`, and `(f2xo7hb6 ?p)`).
    - The predicate `(omggbn9z ?w)` represents the current location of the "active" resource needed for these actions.
    - The predicate `(acovodrf ?w1 ?w2)` represents connectivity (adjacency) between t4rdxh32 type objects (locations).
    - The movements between locations to bring `(omggbn9z ?w)` from one location to another are modeled by `qs8pdrqj` and `i0veo5kv` actions.
    - Only one location has `(omggbn9z ?w)` true at a time (the agent's or resource's position).

    Heuristic Initialization
    ------------------------
    - Extract all goal `?p` objects (nxbd1hc3) that must satisfy `(a7sb56om ?p)`.
    - Extract the static connectivity relation `(acovodrf ...)` between locations (t4rdxh32).
    - Build a graph of locations for shortest path computations.
    - Extract static "kut2ksnk" and "klw11kjf" relations for each `?p` that link them to two locations.
      For each object `?p`, store its required "kut2ksnk location(s)" and "klw11kjf location(s)".
      Usually, exactly one such location each is expected in the static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1) Identify the current location `cur_loc` where `(omggbn9z ?loc)` holds uniquely in the state.

    2) For each goal object `?p`:
       - If `(a7sb56om ?p)` already holds in the state, no cost needed for this object.
       - Else if `(f2xo7hb6 ?p)` holds but not `(a7sb56om ?p)`, we only need the final step:
         a) Move agent to the `klw11kjf` location of `?p` (if not already there).
         b) Perform `y7nik9q2` action (cost 1) to get `(a7sb56om ?p)`.

       - Else (neither `(f2xo7hb6 ?p)` nor `(a7sb56om ?p)` holds), need full process:
         a) Move agent to the `kut2ksnk` location of `?p` (if not already there).
         b) Perform `diwlmgg9` action (cost 1) to get `(f2xo7hb6 ?p)`.
         c) Move to `klw11kjf` location of `?p` (if different from kut2ksnk location, movement cost).
         d) Perform `y7nik9q2` action (cost 1) to get `(a7sb56om ?p)`.

    3) The cost of movement between locations is estimated by the shortest path length in the connectivity graph (number of `(qs8pdrqj/i0veo5kv)` moves).

    4) The heuristic sums the minimal estimated action costs over all outstanding goal objects.

    Note: This heuristic is not necessarily admissible but aims to guide search efficiently by approximating
    the actual number of needed moves and actions.

    """

    def __init__(self, task):
        # Goal objects that must have (a7sb56om ?p)
        self.goal_objects = set()
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "a7sb56om" and len(parts) == 2:
                self.goal_objects.add(parts[1])

        # Build adjacency graph of locations from static acovodrf facts (undirected graph)
        self.adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "acovodrf" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.adj.setdefault(loc1, set()).add(loc2)
                self.adj.setdefault(loc2, set()).add(loc1)

        # Extract locations for 'kut2ksnk' and 'klw11kjf' predicates linking each object to locations
        self.kut2ksnk_map = {}  # Maps nxbd1hc3 object -> set of t4rdxh32 locations
        self.klw11kjf_map = {}  # Maps nxbd1hc3 object -> set of t4rdxh32 locations

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "kut2ksnk" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.kut2ksnk_map.setdefault(obj, set()).add(loc)
            elif parts[0] == "klw11kjf" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.klw11kjf_map.setdefault(obj, set()).add(loc)

        # For heuristic efficiency, convert sets to a unique location per object (choose any if multiple)
        # If multiple locations exist, select the shortest path assumption will handle it.
        self.kut2ksnk_loc = {
            obj: next(iter(locs)) for obj, locs in self.kut2ksnk_map.items() if locs
        }
        self.klw11kjf_loc = {
            obj: next(iter(locs)) for obj, locs in self.klw11kjf_map.items() if locs
        }

    def shortest_path_len(self, start, goal):
        """Compute shortest path length between two locations using BFS, or 0 if equal or unreachable."""

        if start == goal:
            return 0
        if start not in self.adj or goal not in self.adj:
            return 0  # Unable to move, but treat as zero to keep heuristic finite.

        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.adj.get(current, ()):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # If no path found, return large cost to encourage exploration elsewhere
        return 10**6  # Large value to indicate difficult/unreachable

    def __call__(self, node):
        state = node.state

        # If already a goal state, heuristic is zero.
        if all(f"(a7sb56om {obj})" in state for obj in self.goal_objects):
            return 0

        # Identify current location where (omggbn9z ?loc) holds
        cur_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "omggbn9z" and len(parts) == 2:
                cur_loc = parts[1]
                break

        # If for some reason there's no (omggbn9z ...) fact, heuristic cannot proceed well; assume 0
        if cur_loc is None:
            return 0

        total_cost = 0
        for obj in self.goal_objects:
            goal_achieved = f"(a7sb56om {obj})" in state
            if goal_achieved:
                # Already achieved for this object, no cost
                continue

            f2xo7hb6_achieved = f"(f2xo7hb6 {obj})" in state
            kut_loc = self.kut2ksnk_loc.get(obj)
            klw_loc = self.klw11kjf_loc.get(obj)

            # Defensive fallback if static info missing
            if kut_loc is None or klw_loc is None:
                # If missing info, guess cost as 2 actions minimal
                total_cost += 2
                continue

            if f2xo7hb6_achieved:
                # Only need to do the second step (y7nik9q2)

                # Move from current location to klw11kjf location if needed
                move_cost = self.shortest_path_len(cur_loc, klw_loc)

                # The action y7nik9q2 costs 1
                total_cost += move_cost + 1

                # After this object is done, update current location to klw_loc as heuristic assumes sequential process
                cur_loc = klw_loc

            else:
                # Need both steps (diwlmgg9 and y7nik9q2)

                # Move to kut2ksnk location first
                move_to_kut = self.shortest_path_len(cur_loc, kut_loc)
                # Action diwlmgg9 costs 1
                cost_for_first = move_to_kut + 1

                # Then move from kut2ksnk to klw11kjf location (if distinct)
                move_kut_to_klw = self.shortest_path_len(kut_loc, klw_loc)
                # Action y7nik9q2 costs 1
                cost_for_second = move_kut_to_klw + 1

                total_cost += cost_for_first + cost_for_second

                # Update current location to klw_loc for next object
                cur_loc = klw_loc

        return total_cost
