from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract parts of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if the fact matches a pattern with possible wildcards.

    Args:
        fact: A PDDL fact string, e.g. "(predicate obj1 obj2)".
        args: Pattern parts with possible wildcards '*'.

    Returns:
        True if the pattern matches the fact, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain.

    Summary:
    This heuristic estimates the minimum number of actions required to achieve
    the goal conditions based on the current state's unary and binary predicates,
    leveraging the domain's structure and static facts. The goal is to closely estimate
    the progression towards achieving (ne6sl557 ?ris62) predicates from the given initial conditions.

    Assumptions:
    - The primary goal is to achieve (ne6sl557 ?ris62) for certain objects of type wqid8mog.
    - The static facts lpmls56r define a directed connectivity ("next") relation over mz6r1d30 objects.
    - The predicate qkdx0xe2 is a key "location/position" relation between some objects (?n54xx) and mz6r1d30 elements.
    - The predicate fqyrzl30 marks an association between pt4b08k4 and du770mtw objects.
    - Only one object of type du770mtw can have the (ndg963qz) fact true at once, representing some resource or location state.
    - Objects of type wqid8mog start with the (of6wn8m4) fact and require the goal fact (ne6sl557).
    - Actions manipulate qkdx0xe2 relations to move or transfer locations, and the special action f7y6nzq3 changes the status of wqid8mog objects completing ne6sl557.

    Heuristic Initialization:
    - Parse and store the static "lpmls56r" facts as a graph to quickly estimate shortest paths.
    - Extract and store which objects of type du770mtw currently hold the (ndg963qz) predicate.
    - Identify the wqid8mog objects that have (of6wn8m4) initially and require (ne6sl557) in the goal.
    - This heuristic will estimate steps mainly by:
      1) Counting how many wqid8mog objects still lack (ne6sl557).
      2) Calculating how far qkdx0xe2 facts need to move these objects along the connectivity graph.
      3) Estimating needed actions to activate f7y6nzq3 for each target wqid8mog.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify wqid8mog objects in the goal by the presence of (ne6sl557 ?ris62) facts.
       Count how many wqid8mog objects still lack the ne6sl557 fact in the current state.
       If zero, heuristic value is zero (goal state).

    2. For each wqid8mog object missing ne6sl557:
       - Find the current mz6r1d30 location it is associated with via qkdx0xe2.
       - Determine the du770mtw object associated with it (via fqyrzl30).
       - Identify which du770mtw currently holds ndg963qz (there should be exactly one per state).
       - Estimate the cost of moving the qkdx0xe2 relation of corresponding du770mtw object along the lpmls56r graph edges to reach the "ndg963qz"-marked du770mtw.
         This corresponds to moving the "location" of the wqid8mog object closer or aligning it properly.

    3. Add one step per wqid8mog object to represent applying f7y6nzq3 action that achieves ne6sl557 (assuming prerequisite location and states).

    4. Sum these costs to get the heuristic estimate.

    5. This heuristic is not guaranteed admissible but tries to guide search efficiently by:
       - Using shortest paths over lpmls56r static graph.
       - Counting only required moves and final "activation" step f7y6nzq3 per goal object.

    6. Infeasible states or states where no path exists yield a finite estimate since all objects missing ne6sl557 contribute a positive cost.

    Note:
    - This heuristic relies on static connectivity to estimate minimal repositioning steps.
    - The heuristic attempts to minimize expanded nodes by assessing how far each goal wqid8mog object is from enabling the final action f7y6nzq3.
    """

    def __init__(self, task):
        # Store goal facts and static facts for quick reference
        self.goals = task.goals
        self.static = task.static

        # Extract lpmls56r relations (connectivity graph between mz6r1d30 objects)
        self.connectivity = {}
        for fact in self.static:
            if match(fact, "lpmls56r", "*", "*"):
                parts = get_parts(fact)
                src, tgt = parts[1], parts[2]
                if src not in self.connectivity:
                    self.connectivity[src] = set()
                self.connectivity[src].add(tgt)

        # Precompute shortest paths in connectivity graph using BFS for efficient heuristic calls
        self._shortest_paths_cache = {}

        # Extract the wqid8mog objects that must achieve ne6sl557 in the goal
        self.goal_wqid8mog = set()
        for g in self.goals:
            if match(g, "ne6sl557", "*"):
                obj = get_parts(g)[1]
                self.goal_wqid8mog.add(obj)

    def _shortest_path_length(self, start, goal):
        """Return shortest path length in the directed connectivity graph from start to goal or None if unreachable."""
        if (start, goal) in self._shortest_paths_cache:
            return self._shortest_paths_cache[(start, goal)]

        from collections import deque
        if start == goal:
            self._shortest_paths_cache[(start, goal)] = 0
            return 0

        visited = set()
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == goal:
                self._shortest_paths_cache[(start, goal)] = dist
                return dist
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self.connectivity.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        self._shortest_paths_cache[(start, goal)] = None
        return None

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Find which wqid8mog objects have ne6sl557 (goal achieved)
        achieved_wqid8mog = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "ne6sl557", "*")
        }

        # Wqid8mog objects still needing ne6sl557
        remaining = self.goal_wqid8mog - achieved_wqid8mog

        if not remaining:
            # Should be zero cost, but already handled above
            return 0

        # Find which du770mtw object has ndg963qz predicate (the active "location" or resource)
        ndg963qz_du_objs = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "ndg963qz", "*")
        }
        # There should be exactly one such object if problem is well-formed
        # If none or multiple, heuristic still computes with what we have
        ndg_obj = None
        if len(ndg963qz_du_objs) == 1:
            ndg_obj = next(iter(ndg963qz_du_objs))
        # Otherwise set None and treat accordingly (meaning no active location)

        # Helper to find qkdx0xe2 relations: mapping objects (?n54xx or ?ris62) to mz6r1d30 locations
        qkdx_map = {}
        for fact in state:
            if match(fact, "qkdx0xe2", "*", "*"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                qkdx_map[obj] = loc

        # Build reverse map from pt4b08k4 objects to du770mtw via fqyrzl30
        # We need du770mtw object (cz2xr) associated with pt4b08k4 (n54xx)
        fqyrzl30_map = {}
        for fact in state:
            if match(fact, "fqyrzl30", "*", "*"):
                parts = get_parts(fact)
                pt, du = parts[1], parts[2]
                fqyrzl30_map[pt] = du

        # Estimate cost per each remaining goal wqid8mog object
        total_cost = 0

        for wobj in remaining:
            # Find current location (mz6r1d30) of this wqid8mog object from qkdx0xe2
            loc = qkdx_map.get(wobj, None)
            if loc is None:
                # If no location, pessimistically count some high cost to position it
                loc_cost = 3
            else:
                loc_cost = 0

            # Find associated pt4b08k4 object ?n54xx that corresponds to this wqid8mog (wobj)
            # The domain links pt4b08k4 and du770mtw via fqyrzl30
            # Also note nm6lzp2x moves qkdx0xe2 between locations for pt4b08k4

            # We try to find pt4b08k4 `pt_obj` from fqyrzl30 that points to some du770mtw,
            # and whose location (qkdx0xe2) matches loc of wobj
            # Because wobj is wqid8mog and fqyrzl30 connects pt4b08k4->du770mtw,
            # while qkdx0xe2 connects du770mtw -> mz6r1d30 and pt4b08k4 -> mz6r1d30 for nm6lzp2x
            # Actually from domain: jql8txb3 moves fqyrzl30: (fqyrzl30 ?n54xx ?cz2xr)
            # nm6lzp2x moves qkdx0xe2 for pt4b08k4 obj
            # We'll simplify by linking wobj -> loc, then try to estimate how far this loc is from the loc of the active du770mtw with ndg963qz.

            # Find du770mtw associated with wobj if any (from fqyrzl30 for some pt4b08k4?)
            # Because wobj is wqid8mog, relates qkdx0xe2 to mz6r1d30, but we don't have direct link wobj -> pt4b08k4 in state
            # So we don't try to recover that, we just estimate distance from loc of wobj to loc of the ndg963qz du770mtw object.

            # Get location of the ndg963qz du770mtw object
            ndg_loc = None
            if ndg_obj is not None:
                ndg_loc = qkdx_map.get(ndg_obj, None)

            if loc is not None and ndg_loc is not None:
                dist = self._shortest_path_length(loc, ndg_loc)
                # If no path, assign a penalty
                if dist is None:
                    dist = 3
                loc_cost += dist
            else:
                # If missing info, assign penalty
                loc_cost += 3

            # Add 1 for the activation action f7y6nzq3 required per wqid8mog object missing
            total_cost += loc_cost + 1

        return total_cost
