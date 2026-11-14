from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(u4srx1nb bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to achieve all goal facts of the form (g59iwyxf ?nut),
    which indicates that a given rupjajag object (?gdh2s) is in the final desired state. The main challenge is
    to (1) move ilfxqhcc objects (?v7he5, e.g. 'bob') along a chain of i6azntdy locations, (2) use these ilfxqhcc objects
    with e2zu8y93 objects (?aqz3v, e.g. 'spanner') to manipulate rupjajag objects (?gdh2s, e.g. 'nut'),
    unlocking their goal condition status (g59iwyxf). The heuristic estimates steps based on
    distances along locations, counting necessary movements of ilfxqhcc objects and activation of e2zu8y93 objects.

    Assumptions:
    - The goal is to make all (g59iwyxf ?nut) facts true for each nut in the goal.
    - Each (g59iwyxf ?gdh2s) fact results from one application of the "kjsqfni8" action.
    - To apply "kjsqfni8" on a nut, the nut must be at some location (?r2bzp), the ilfxqhcc object must be at that location
      and hold the yud7t3vb relation to the e2zu8y93 object (spanner) which also holds the g0brvc06 predicate.
    - The network of i6azntdy locations forms a graph connected by (kxokb5hx) edges, assumed to be symmetric.
    - Movement is done by the action "n9sbfmrs" that moves the ilfxqhcc object (e.g. bob) along kxokb5hx edges between i6azntdy nodes.
    - (yud7t3vb ?v7he5 ?aqz3v) indicates which spanner an ilfxqhcc object holds/controls. This must be true to use kjsqfni8.
    - The number of kjsqfni8 applications corresponds to the number of nuts at their locations that are still not (g59iwyxf).

    Heuristic Initialization:
    - Extract the edges (kxokb5hx) of the undirected graph between locations (i6azntdy) from static facts.
    - Extract the set of all spanners (e2zu8y93) currently holding g0brvc06.
    - Extract the locations of all i6azntdy objects and store u4srx1nb facts for ilfxqhcc and rupjajag objects.
    - Build a graph with adjacency lists for fast shortest path computations among locations.
    - Extract the goal nuts for which (g59iwyxf ?nut) should be true in the goal.
    - Store initial relation (yud7t3vb) from ilfxqhcc objects to spanners controlling them.
    - Note: the domain does not explicitly model a moving robot, but ilfxqhcc entities that can be moved.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state is a goal state (all goal (g59iwyxf ?nut) present), heuristic = 0.
    2. For each nut in the goal (rupjajag objects), if (g59iwyxf nut) is already true in the state, no cost for it.
    3. Otherwise, estimate the cost to achieve (g59iwyxf nut):
       a) Find the nut's current location (?r2bzp) via (u4srx1nb nut location).
       b) Identify which spanner currently holds g0brvc06 and is needed to activate this nut.
       c) Identify an ilfxqhcc object (?v7he5) that controls a needed spanner (via (yud7t3vb ?v7he5 ?aqz3v)).
       d) Determine the current location of that ilfxqhcc object (via (u4srx1nb ?v7he5 loc_v)).
       e) Compute the shortest path distance in the location graph from ilfxqhcc's location to the nut's location.
          Each edge corresponds to one "n9sbfmrs" move action to move ilfxqhcc from one location to another.
       f) The cost to achieve the nut goal includes:
          - Moving ilfxqhcc to nut's location (distance)
          - Applying "kjsqfni8" (one action)
    4. Sum the estimated costs for all nuts not yet achieved.
    5. Return this sum as the heuristic value.

    This heuristic uses shortest path estimates for ilfxqhcc movement plus one application of the nut unlocking action,
    aiming to guide search efficiently toward moving necessary agents and unlocking nuts at their locations.
    """

    def __init__(self, task):
        # Store the goal facts: set of (g59iwyxf nut) facts.
        self.goal_nuts = set()
        for goal in task.goals:
            if goal.startswith("(g59iwyxf "):
                # Extract the nut object name.
                parts = get_parts(goal)
                if len(parts) == 2:
                    self.goal_nuts.add(parts[1])

        # Extract graph edges from static information (kxokb5hx is undirected)
        self.graph = {}
        for fact in task.static:
            if match(fact, "kxokb5hx", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1)

    def shortest_distance(self, start, goal):
        """Compute shortest path length between two locations in the graph using BFS. Return large number if unreachable."""
        if start == goal:
            return 0
        if start not in self.graph or goal not in self.graph:
            # No connectivity information; return a large cost.
            return 1000

        from collections import deque

        visited = set()
        queue = deque([(start, 0)])
        visited.add(start)
        while queue:
            current, dist = queue.popleft()
            for nbr in self.graph.get(current, []):
                if nbr == goal:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        # If goal not reachable:
        return 1000

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is 0
        # Here, the goal is set of (g59iwyxf nut) facts all true
        state_goals = {fact for fact in state if fact.startswith("(g59iwyxf ")}
        if self.goal_nuts <= {f[10:-1] for f in state_goals}:
            return 0

        # Extract relevant facts:
        # ilfxqhcc location: (u4srx1nb ?v7he5 ?loc), with ?v7he5 of type ilfxqhcc
        # spanner ownership: (yud7t3vb ?v7he5 ?aqz3v) means ilfxqhcc ?v7he5 controls spanner ?aqz3v
        # spanner state: (g0brvc06 ?aqz3v) means spanner active/usable
        # nut location: (u4srx1nb ?gdh2s ?loc), with ?gdh2s of type rupjajag
        # goal nuts: need (g59iwyxf ?gdh2s)

        # First collect:
        # 1) Map ilfxqhcc -> location
        ilfxqhcc_locations = {}
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                _, entity, loc = get_parts(fact)
                # We infer ilfxqhcc objects by which appear as keys in (yud7t3vb), so gather below later.
                ilfxqhcc_locations[entity] = loc

        # 2) Map nut -> location (nut = rupjajag)
        nut_locations = {}
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                _, obj, loc = get_parts(fact)
                # We assume nuts are those with (bu1om0m1 ?gdh2s) facts or in goal nuts set.
                if obj in self.goal_nuts:
                    nut_locations[obj] = loc

        # 3) Map ilfxqhcc -> spanner controlled (yud7t3vb)
        ilf_spanner = {}
        for fact in state:
            if match(fact, "yud7t3vb", "*", "*"):
                _, ilf, spanner = get_parts(fact)
                ilf_spanner[ilf] = spanner

        # 4) Set of active spanners (g0brvc06)
        active_spanners = set()
        for fact in state:
            if match(fact, "g0brvc06", "*"):
                _, spanner = get_parts(fact)
                active_spanners.add(spanner)

        # 5) Nuts already achieved (g59iwyxf)
        nuts_achieved = set()
        for fact in state:
            if match(fact, "g59iwyxf", "*"):
                _, nut = get_parts(fact)
                nuts_achieved.add(nut)

        # For heuristic: sum over all nuts in goal that are NOT yet achieved,
        # cost = distance from ilfxqhcc (that can control a currently active spanner) to nut's location
        # + 1 for 'kjsqfni8' action to activate nut's goal state.

        total_cost = 0

        for nut in self.goal_nuts:
            if nut in nuts_achieved:
                continue  # no cost, already achieved

            nut_loc = nut_locations.get(nut)
            if nut_loc is None:
                # Nut location unknown, assign large cost to encourage exploration
                total_cost += 1000
                continue

            # For each ilfxqhcc object controlling an active spanner
            # find the minimal cost to reach nut's location and enable unlocking it
            min_cost_for_nut = None
            for ilf, spanner in ilf_spanner.items():
                if spanner in active_spanners:
                    ilf_loc = ilfxqhcc_locations.get(ilf)
                    if ilf_loc is None:
                        continue
                    dist = self.shortest_distance(ilf_loc, nut_loc)
                    # Cost is: distance (number of moves) + 1 (application of kjsqfni8)
                    cost = dist + 1
                    if min_cost_for_nut is None or cost < min_cost_for_nut:
                        min_cost_for_nut = cost

            # If no ilfxqhcc with active spanner found, assign large cost to encourage activation first
            if min_cost_for_nut is None:
                min_cost_for_nut = 1000

            total_cost += min_cost_for_nut

        # Return heuristic estimate (finite number, 0 if solved)
        return total_cost
