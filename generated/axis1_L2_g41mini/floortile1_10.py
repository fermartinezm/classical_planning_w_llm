from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def parse_fact(fact):
    """Parse a PDDL fact string into a list of components without parentheses."""
    return fact[1:-1].split()


def match_fact(fact, *pattern):
    """
    Check if a fact matches a pattern with wildcards.

    Example:
       match_fact("(ypt9mrgu p1 n1)", "ypt9mrgu", "*", "n1") returns True
    """
    parts = parse_fact(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions needed to achieve all goal facts
    of the form (cb6fw2xd ?loc ?obj) that represent some form of object-to-value assignments.
    It measures which goal (cb6fw2xd ...) facts are still missing in the state and estimates
    the effort needed to produce them based on available static relations and dynamic facts.

    Assumptions:
    - The goal consists exclusively of (cb6fw2xd <nh1sy8v9> <qekv4465>) facts (as given by examples).
    - To achieve (cb6fw2xd L O), the object O must be at location L with predicate cb6fw2xd.
    - The domain actions revolve around moving and swapping ypt9mrgu facts (linking k9b7lknr to nh1sy8v9),
      controlling "fguw2142" flags on nh1sy8v9 objects, and changing the cb6fw2xd facts on nh1sy8v9 objects.
    - Static relations ga4l9do0, s4jrpk2b, vom4pfig, vjcqgdgn describe connectivity or accessibility between objects.
    - The heuristic ignores complicated intermediate steps and plans approximations by counting:
      how many goal facts are unmet, and how many moves might be needed on the ypt9mrgu and fguw2142 relations to move objects
      and enable the final assignment (cb6fw2xd).

    Heuristic Initialization:
    Extract and store:
    - The goal cb6fw2xd facts to track which (location, value) pairs are required.
    - Sets of static relations:
      ga4l9do0 (nh1sy8v9 to nh1sy8v9),
      s4jrpk2b (nh1sy8v9 to nh1sy8v9),
      vom4pfig (nh1sy8v9 to nh1sy8v9),
      vjcqgdgn (nh1sy8v9 to nh1sy8v9),
      fezojtj3 (qekv4465) values,
    - These static relations provide "paths" or "links" used in actions to move ypt9mrgu links or fguw2142 flags.

    Step-By-Step Thinking for Computing Heuristic:
    1. Extract:
       - Current (cb6fw2xd L O) facts achieved in the state.
       - Current (ypt9mrgu E L) facts stating which entity E "holds" location L.
       - Current (fguw2142 L) showing which location currently has the "flag" for moving.
       - Current (qp2t6u4s E V) describing object-value bindings.
    2. For each goal (cb6fw2xd L O) not achieved, check if the prerequisite facts exist:
       - A location L′ that can be used to move O or a proxy.
       - A way to move the ypt9mrgu from current location of E to target location L.
    3. Estimate heuristic cost by:
       - Counting the number of missing goal cb6fw2xd facts.
       - Adding estimated moves needed for ypt9mrgu reassignments or fguw2142 moves to allow required transfers.
       - Use static graph connectivity approximations among nh1sy8v9 objects to approximate minimal moves.
    4. Combine counts to produce a heuristic estimate of remaining actions.
    5. Return 0 if the goal is already reached.

    The heuristic is not admissible but attempts to capture the major effort needed to reach the goal from the current state.
    """

    def __init__(self, task):
        # Store goal (cb6fw2xd L O) facts, target location-object assignments.
        self.goal_cb6fw2xd = set()
        for fact in task.goals:
            if match_fact(fact, "cb6fw2xd", "*", "*"):
                self.goal_cb6fw2xd.add(fact)

        # Extract and index static relations for faster lookup.
        self.static_ga4l9do0 = set()
        self.static_s4jrpk2b = set()
        self.static_vom4pfig = set()
        self.static_vjcqgdgn = set()
        self.static_fezojtj3 = set()

        for fact in task.static:
            if match_fact(fact, "ga4l9do0", "*", "*"):
                self.static_ga4l9do0.add(fact)
            elif match_fact(fact, "s4jrpk2b", "*", "*"):
                self.static_s4jrpk2b.add(fact)
            elif match_fact(fact, "vom4pfig", "*", "*"):
                self.static_vom4pfig.add(fact)
            elif match_fact(fact, "vjcqgdgn", "*", "*"):
                self.static_vjcqgdgn.add(fact)
            elif match_fact(fact, "fezojtj3", "*"):
                self.static_fezojtj3.add(fact)

        # Build connectivity maps for nh1sy8v9 objects using static relations.
        # These relations represent different "edges" between nh1sy8v9 nodes to approximate paths for moving flags or roles.
        self.neighbors_ga4l9do0 = {}
        self.neighbors_s4jrpk2b = {}
        self.neighbors_vom4pfig = {}
        self.neighbors_vjcqgdgn = {}

        # Helper to add bidirectional edges
        def add_edge(neighbors, a, b):
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)

        for fact in self.static_ga4l9do0:
            _, a, b = parse_fact(fact)
            add_edge(self.neighbors_ga4l9do0, a, b)
        for fact in self.static_s4jrpk2b:
            _, a, b = parse_fact(fact)
            add_edge(self.neighbors_s4jrpk2b, a, b)
        for fact in self.static_vom4pfig:
            _, a, b = parse_fact(fact)
            add_edge(self.neighbors_vom4pfig, a, b)
        for fact in self.static_vjcqgdgn:
            _, a, b = parse_fact(fact)
            add_edge(self.neighbors_vjcqgdgn, a, b)

    def __call__(self, node):
        state = node.state

        # Quick 0 check: if all goal cb6fw2xd facts are in state, heuristic = 0.
        if self.goal_cb6fw2xd <= state:
            return 0

        # Extract current achieved cb6fw2xd facts.
        achieved_cb6fw2xd = {fact for fact in state if match_fact(fact, "cb6fw2xd", "*", "*")}
        # Extract current ypt9mrgu facts: (ypt9mrgu E L)
        ypt9mrgu_facts = [parse_fact(fact)[1:] for fact in state if match_fact(fact, "ypt9mrgu", "*", "*")]
        # Map from entity E to location L (assuming one-to-one per E)
        entity_to_loc = {}
        for e, l in ypt9mrgu_facts:
            entity_to_loc[e] = l

        # Extract set of fguw2142-location flags (indicating which location currently "holds" the flag)
        fguw2142_locations = {parse_fact(fact)[1] for fact in state if match_fact(fact, "fguw2142", "*")}

        # Extract qp2t6u4s facts: indicating entity-value binding: (qp2t6u4s E V)
        qp2t6u4s_facts = [parse_fact(fact)[1:] for fact in state if match_fact(fact, "qp2t6u4s", "*", "*")]
        entity_to_value = {}
        for e, v in qp2t6u4s_facts:
            entity_to_value[e] = v

        # Goal facts are (cb6fw2xd loc val), meaning "loc" must be paired with "val".
        # Start heuristic cost estimate:
        cost = 0

        # For each missing goal cb6fw2xd fact, roughly estimate cost as follows:
        #  - 1 for producing the cb6fw2xd fact (e.g. running action wbsgr2yj, kkqx0eg7, etc.)
        #  - Additional costs for moving ypt9mrgu from current location to the target location.
        #  - Costs for moving the fguw2142 "flag" to allowed intermediate locations.

        # We'll approximate moves on neighbors_vom4pfig and neighbors_vjcqgdgn to simulate necessary moves for fguw2142,
        # and neighbors_s4jrpk2b and neighbors_ga4l9do0 for ypt9mrgu moves.

        # Because the domain is complex, we estimate costs by counting:
        # - missing (cb6fw2xd) facts count,
        # - plus minimal hops from ypt9mrgu current location of the entity to the goal location (using neighbors_s4jrpk2b and neighbors_ga4l9do0),
        # - plus 1 extra step if fguw2142 currently not on the goal location.

        # Build undirected graph combining ga4l9do0 and s4jrpk2b for ypt9mrgu moves.
        combined_neighbors_ypt = {}
        for nmap in (self.neighbors_ga4l9do0, self.neighbors_s4jrpk2b):
            for k, vs in nmap.items():
                combined_neighbors_ypt.setdefault(k, set()).update(vs)

        # Build undirected graph combining vom4pfig and vjcqgdgn for fguw2142 moves.
        combined_neighbors_fguw = {}
        for nmap in (self.neighbors_vom4pfig, self.neighbors_vjcqgdgn):
            for k, vs in nmap.items():
                combined_neighbors_fguw.setdefault(k, set()).update(vs)

        # For shortest path computation, use BFS:
        def shortest_path(graph, start, goal):
            """Compute shortest path length in an undirected graph, or None if no path."""
            if start == goal:
                return 0
            if start not in graph or goal not in graph:
                return None
            visited = {start}
            fringe = [(start, 0)]
            while fringe:
                node, dist = fringe.pop(0)
                for neigh in graph[node]:
                    if neigh == goal:
                        return dist + 1
                    if neigh not in visited:
                        visited.add(neigh)
                        fringe.append((neigh, dist + 1))
            return None

        # Heuristic summation over missing goals.
        for goal_fact in self.goal_cb6fw2xd - achieved_cb6fw2xd:
            _, loc_goal, val_goal = parse_fact(goal_fact)

            # Find entities that currently have the value val_goal bound, if any.
            candidates = [e for e, v in entity_to_value.items() if v == val_goal]

            # If no entity currently bound to val_goal, penalty of 2 for initial binding + move.
            if not candidates:
                cost += 3  # 1 for binding qp2t6u4s (modeled as cheap) + 2 for moves to enable.
                # Cannot estimate path without entity; continue to next.
                continue

            # For each candidate entity, estimate minimal cost to get ypt9mrgu from current loc to goal loc.
            min_move_cost = None
            for entity in candidates:
                loc_current = entity_to_loc.get(entity)
                if loc_current is None:
                    # No current location for entity --- pegged costly
                    dist = 3
                else:
                    dist = shortest_path(combined_neighbors_ypt, loc_current, loc_goal)
                    if dist is None:
                        dist = 4  # no path found, big penalty
                if min_move_cost is None or dist < min_move_cost:
                    min_move_cost = dist

            if min_move_cost is None:  # fallback penalty
                min_move_cost = 4

            # Estimate if fguw2142 flag currently at goal location
            # If yes, no extra cost; else +1 for moving flag.
            fguw_cost = 0 if loc_goal in fguw2142_locations else 1

            # Each goal requires:
            # 1 for generating the cb6fw2xd,
            # plus min_move_cost (moves of ypt9mrgu),
            # plus fguw_cost (moves of fguw2142)
            cost += 1 + (min_move_cost if min_move_cost is not None else 3) + fguw_cost

        # Ensure heuristic is at least 1 for states not solved (to avoid zero before goal)
        return max(cost, 1)
