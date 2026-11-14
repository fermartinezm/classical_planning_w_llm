from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ypt9mrgu en18p stq0j)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (hi4l5zmb) domain.

    Summary:
    ----------
    The heuristic estimates how many actions are needed to achieve the goal facts
    of the form (cb6fw2xd ?location ?item), i.e., to have certain qekv4465-typed
    objects 'attached' to nh1sy8v9-typed locations with predicate cb6fw2xd.
    The main challenge is to transport or move entities ypt9mrgu(?en18p ?loc)
    from one nh1sy8v9 location to another, while respecting availability of
    intermediate "free" locations (fguw2142) and adjacency relations (ga4l9do0,
    s4jrpk2b, vom4pfig, vjcqgdgn) encoding the possible movements or handovers.

    Assumptions:
    ------------
    - The critical goal facts to achieve are the cb6fw2xd predicate instances.
    - Transporting an en18p object from one location to another requires a sequence
      of moves following adjacency predicates and availability of free locations.
    - The predicates ypt9mrgu encode the current locations of the objects that must be eventually
      attached as per cb6fw2xd goals.
    - Free locations fguw2142 can be used for intermediate holds during moves.
    - Adjacency and connectivity of locations are static and given by predicates s4jrpk2b,
      vom4pfig, vjcqgdgn, ga4l9do0, to guide valid move actions.
    - The actions mostly move ypt9mrgu objects between nh1sy8v9 locations, possibly changing
      free flags and using adjacency.
    - The heuristic is non-admissible and aims to estimate minimal action costs intuitively
      while being fast to compute.

    Heuristic Initialization:
    --------------------------
    - Extract all goal target facts with predicate cb6fw2xd indicating that a certain nh1sy8v9
      location must hold a qekv4465 object.
    - Extract static adjacency relations (s4jrpk2b, vom4pfig, vjcqgdgn, ga4l9do0) into dictionaries
      to represent graph edges between nh1sy8v9 objects.
    - Extract which qekv4465 objects must be at each location from the goal.
    - Static predicate fezojtj3 marking certain qekv4465 objects is noted but not directly used;
      primarily used in preconditions of actions affecting qp2t6u4s facts.
    - The heuristic focuses on the shortest "cost" to move each en18p object's location
      from its current to its goal constraints, estimating cost by adjacency steps.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify all en18p objects involved in the goal cb6fw2xd facts: which qekv4465 objects
       must be attached to which nh1sy8v9 locations.
    2. For each en18p object, find the current location(s) by facts ypt9mrgu(?en18p ?loc).
       Because predicates ypt9mrgu show where each en18p is currently.
    3. Construct adjacency graphs for nh1sy8v9 locations from static predicates to model connectivity.
    4. For each goal cb6fw2xd(?loc ?item), find the en18p object that currently holds or can hold
       ?item by scanning qp2t6u4s facts in current state (each en18p linked with qekv4465 objects).
    5. For each needed object movement, estimate a minimal path length via the adjacency
       graphs from the current location to the goal location of en18p or the location holding
       the qekv4465 object in the goal.
    6. Add heuristic costs to cover:
       - Moving en18p object along adjacency edges (each estimated as 1 action cost).
       - Actions to change the location of qekv4465 objects (load/unload steps) counted minimally as 1 per goal fact.
    7. Count free location availability (fguw2142) as a gating factor for moves; if no free location,
       heuristic cost increases because moves require intermediate free states.
    8. Sum all partial costs across all goal facts, yielding heuristic estimate.
    9. If the goal is already reached (all cb6fw2xd facts are true in state), return 0.

    Note: The heuristic favors efficiency over admissibility: actual plans might involve additional
    steps due to complex predicates usage and intermediate holds, but the heuristic attempts
    a weighted shortest-path estimation based on adjacency.

    """

    def __init__(self, task):
        """Initialize heuristic using static predicates and goal facts."""
        self.goals = task.goals  # goal facts (frozenset)

        # Extract all goal cb6fw2xd facts and map location to desired qekv4465 objects.
        self.goal_cb6fw2xd = {}
        for fact in self.goals:
            if fact.startswith("(cb6fw2xd"):
                parts = get_parts(fact)
                loc, obj = parts[1], parts[2]
                self.goal_cb6fw2xd[loc] = obj

        static = task.static

        # Build adjacency dictionaries for nh1sy8v9 locations based on static predicates.
        # Predicates encode adjacency and connectivity for moving ypt9mrgu (en18p) objects.
        self.adj_s4jrpk2b = {}  # s4jrpk2b(?loc1 ?loc2)
        self.adj_vom4pfig = {}  # vom4pfig(?loc1 ?loc2)
        self.adj_vjcqgdgn = {}  # vjcqgdgn(?loc1 ?loc2)
        self.adj_ga4l9do0 = {}  # ga4l9do0(?loc1 ?loc2)

        # Helper to build dict from static facts for bidirectional adjacency
        def build_adj(facts, predicate):
            adj = {}
            for fact in facts:
                if fact.startswith("(" + predicate):
                    parts = get_parts(fact)
                    a, b = parts[1], parts[2]
                    adj.setdefault(a, set()).add(b)
                    # Also add reverse edge to represent undirected adjacency
                    adj.setdefault(b, set()).add(a)
            return adj

        self.adj_s4jrpk2b = build_adj(static, "s4jrpk2b")
        self.adj_vom4pfig = build_adj(static, "vom4pfig")
        self.adj_vjcqgdgn = build_adj(static, "vjcqgdgn")
        self.adj_ga4l9do0 = build_adj(static, "ga4l9do0")

        # Combine all adjacency relations in one unified adjacency graph.
        self.adjacency = {}
        for adj_dict in [
            self.adj_s4jrpk2b,
            self.adj_vom4pfig,
            self.adj_vjcqgdgn,
            self.adj_ga4l9do0,
        ]:
            for key, neighbors in adj_dict.items():
                self.adjacency.setdefault(key, set()).update(neighbors)

    def __call__(self, node):
        """Compute heuristic estimate of actions needed to achieve all cb6fw2xd goals."""

        state = node.state

        # 1) If in goal state, return 0 heuristic.
        if self.goals <= state:
            return 0

        # 2) Parse current facts from state to get:
        #    - current ypt9mrgu links (which en18p objects are at which nh1sy8v9 location)
        #    - current qp2t6u4s links (which qekv4465 objects en18p objects hold)
        #    - which locations are free (fguw2142)
        #    - current cb6fw2xd facts (which locations hold which qekv4465 objects)
        # We rely mainly on ypt9mrgu and qp2t6u4s to track en18p objects' locations and held objects.

        # Maps en18p object -> nh1sy8v9 current location (from ypt9mrgu)
        en18p_locs = {}
        for fact in state:
            if fact.startswith("(ypt9mrgu"):
                parts = get_parts(fact)
                en18p, loc = parts[1], parts[2]
                en18p_locs[en18p] = loc

        # Maps en18p objects -> set of qekv4465 objects currently held (from qp2t6u4s)
        en18p_holds = {}
        for fact in state:
            if fact.startswith("(qp2t6u4s"):
                parts = get_parts(fact)
                en18p, obj = parts[1], parts[2]
                en18p_holds.setdefault(en18p, set()).add(obj)

        # Current cb6fw2xd facts: location -> qekv4465 objects there now
        current_cb6fw2xd = {}
        for fact in state:
            if fact.startswith("(cb6fw2xd"):
                parts = get_parts(fact)
                loc, obj = parts[1], parts[2]
                current_cb6fw2xd[loc] = obj

        # Set of free locations
        free_locations = set()
        for fact in state:
            if fact.startswith("(fguw2142"):
                parts = get_parts(fact)
                free_locations.add(parts[1])

        # 3) Heuristic calculation:
        #   For each goal cb6fw2xd(loc, obj), calculate:
        #   - If already true in state, cost = 0
        #   - Else identify en18p that holds obj (from en18p_holds)
        #   - Compute minimal adjacency distance from en18p current location to goal location
        #   - Add cost proportional to distance + 1 to represent placing the obj at loc
        #
        #   We approximate the needed moves of en18p objects toward the goal location.

        def bfs_distance(start, goal, adjacency):
            """Breadth-first search for shortest undirected path distance between start and goal."""
            if start == goal:
                return 0
            visited = set([start])
            frontier = [(start, 0)]
            while frontier:
                current, dist = frontier.pop(0)
                for neighbor in adjacency.get(current, []):
                    if neighbor == goal:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append((neighbor, dist + 1))
            return None  # No path

        total_cost = 0

        for goal_loc, goal_obj in self.goal_cb6fw2xd.items():
            # If goal already achieved, skip
            if current_cb6fw2xd.get(goal_loc) == goal_obj:
                continue

            # Find which en18p holds the goal_obj currently (if any)
            en18p_owner = None
            for en18p, objs in en18p_holds.items():
                if goal_obj in objs:
                    en18p_owner = en18p
                    break

            # If no en18p holds the object, heuristic penalty: must be acquired or moved by some means
            if en18p_owner is None:
                # Heuristic penalty for unknown or missing holder (~2 actions)
                total_cost += 4
                continue

            # Get current location of en18p_owner
            start_loc = en18p_locs.get(en18p_owner)
            if start_loc is None:
                # If no location info, penalize a bit
                total_cost += 3
                continue

            dist = bfs_distance(start_loc, goal_loc, self.adjacency)
            if dist is None:
                # No path found, large penalty (but finite)
                total_cost += 10
                continue

            # Base cost is distance to move en18p_owner to goal_loc
            # Plus 1 action cost to place the object at goal_loc (achieve cb6fw2xd)
            # Plus 1 extra if goal_loc has no free location nearby to enable swapping (fguw2142)
            free_adjacent = any(
                free_loc in self.adjacency.get(goal_loc, set()) for free_loc in free_locations
            )
            swap_cost = 0 if free_adjacent or (goal_loc in free_locations) else 1

            # Sum partial cost for this goal relation
            total_cost += dist + 1 + swap_cost

        # 4) Also add cost proportional to number of unmatched goal cb6fw2xd facts as rough penalty
        unmatched_goals = sum(
            1 for loc, obj in self.goal_cb6fw2xd.items() if current_cb6fw2xd.get(loc) != obj
        )
        total_cost += unmatched_goals  # At least one action per unmatched goal

        # 5) Minimal heuristic is 1 for any unsolved goal, never 0 unless solved
        return max(total_cost, 1)
