from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ypt9mrgu pjr7u2i2 j0on2kgr)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary:
    --------
    This heuristic estimates the number of actions required to achieve the goal
    facts `(cb6fw2xd ?nh1sy8v9 ?qekv4465)` for all required objects. The main
    goal predicate concerns the relation `cb6fw2xd` between objects of type nh1sy8v9 
    and qekv4465. Achieving these facts requires relocating k9b7lknr-typed agents (via 
    predicates involving `ypt9mrgu`) to specific nh1sy8v9 locations and assigning appropriate 
    qekv4465 objects to nh1sy8v9 objects, while navigating constraints defined by static 
    relations (`ga4l9do0`, `s4jrpk2b`, `vjcqgdgn`, `vom4pfig` and `fezojtj3`).

    This heuristic focuses on estimating:
    1. How many ns1hy8v9 objects still need the `cb6fw2xd` relation with `qekv4465` objects fulfilled.
    2. How many `ypt9mrgu` moves are necessary to bring each relevant agent close to target locations.
    3. How many replacements of the `qp2t6u4s` connections for k9b7lknr objects must be done.
    4. How many moves are needed to reposition the "flagged" location indicated by predicate `fguw2142`.

    Assumptions:
    ------------
    - The goal facts always target the predicate `cb6fw2xd` between nh1sy8v9 and qekv4465 objects.
    - Objects' static relations form a graph-like structure where movements of `ypt9mrgu` agents occur along edges given by predicates `s4jrpk2b`, `vjcqgdgn`, `vom4pfig`, and `ga4l9do0`.
    - Every `cb6fw2xd` fact requires the relevant nh1sy8v9 object to hold the correct qekv4465 object, which involves action `wvswfzmn` possibly preceded by positioning actions.
    - Exactly one location is flagged with `fguw2142` at a time, which is pivotal for movement and action applicability.
    - Transitioning the `ypt9mrgu` predicate for an agent corresponds roughly to moving it from one nh1sy8v9 location to another; these moves are costly and required for goal achievement.

    Heuristic Initialization:
    -------------------------
    On initialization, the heuristic extracts:
    - The set of goal `(cb6fw2xd nh1sy8v9 qekv4465)` facts the state must satisfy.
    - Static facts like `fezojtj3` (available qekv4465 objects), graph edges represented by predicates:
      `s4jrpk2b`, `vjcqgdgn`, `vom4pfig`, `ga4l9do0`.
    - Objects typed as k9b7lknr (agents) and qekv4465 (items) are identified from static dominance in goal and static facts.
    - It builds maps for quick reachability and adjacency checking within nh1sy8v9 objects for planning movement of agents.
    - Identifies the unique `fguw2142` flagged nh1sy8v9 object representing the "current position of interest".

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1) Check if the current state already satisfies the goal (all required `(cb6fw2xd ...)` facts hold).
       - If so, heuristic = 0.

    2) Identify which `(cb6fw2xd nh1sy8v9 qekv4465)` goal facts are not satisfied yet.

    3) For each unsatisfied goal fact `(cb6fw2xd X Y)`:
       - Estimate the number of actions needed to produce this fact.
         Typically, the actions involve:
         (a) Moving the relevant agent (k9b7lknr) currently at some nh1sy8v9 location Z to X, by sequences of moves along `s4jrpk2b`, `vjcqgdgn` or similar edges.
         (b) Ensuring the k9b7lknr object holds the correct qekv4465 object Y, possibly requiring a replacement action `wvswfzmn`.
         (c) Applying `kkqx0eg7` or `wbsgr2yj`-like actions to produce `(cb6fw2xd X Y)`.

    4) Moving agents involves changing the `ypt9mrgu` predicate location from one nh1sy8v9 node to another.
       Use shortest path approximations over static edges (`s4jrpk2b`, `vjcqgdgn`, `vom4pfig`, `ga4l9do0`) to estimate distances between locations.

    5) Changing held qekv4465 objects (the `qp2t6u4s` facts) involves at least one action per needed replacement.

    6) Manipulating the flagged `fguw2142` location is typically a prerequisite to many moves (agents have to move relative to it),
       so include a penalty for how many `fguw2142` flag moves will be needed.

    7) Sum these minimal costs over all unsatisfied goal facts to form the heuristic estimate.

    This heuristic may overestimate or underestimate parts but aims to reduce expansions by focusing on the key relations
    and dependencies found in the domain specification and goal.

    """

    def __init__(self, task):
        self.goals = task.goals  # goal facts to fulfill
        self.static = task.static  # static facts
        self.facts = task.facts

        # Extract all goal cb6fw2xd facts to fulfill: set of (nh1sy8v9_obj, qekv4465_obj)
        self.goal_cb6fw2xd = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "cb6fw2xd" and len(parts) == 3:
                self.goal_cb6fw2xd.add((parts[1], parts[2]))

        # Extract all static edges between nh1sy8v9 objects for navigation of agents
        # Using predicates: s4jrpk2b, vjcqgdgn, vom4pfig, ga4l9do0
        # These form directed edges from first to second argument.
        self.edges = {}
        # Record all nh1sy8v9 nodes in the static info
        self.nh1sy8v9_nodes = set()
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in {"s4jrpk2b", "vjcqgdgn", "vom4pfig", "ga4l9do0"} and len(parts) == 3:
                src, tgt = parts[1], parts[2]
                self.nh1sy8v9_nodes.add(src)
                self.nh1sy8v9_nodes.add(tgt)
                if src not in self.edges:
                    self.edges[src] = set()
                self.edges[src].add(tgt)
            if pred in {"fezojtj3"} and len(parts) == 2:
                # Collect all qekv4465 objects
                pass
            if pred in {"ypt9mrgu"} and len(parts) == 3:
                # identifies k9b7lknr agent's current location, but in static facts no ypt9mrgu expected
                pass

        # Precompute shortest paths between all nh1sy8v9 nodes for fast distance estimates using BFS
        # Store in a dict of dicts: self.distances[src][tgt] = shortest path length or large number if unreachable
        self.distances = {}
        for node in self.nh1sy8v9_nodes:
            self.distances[node] = self._bfs_distances(node)

    def _bfs_distances(self, start):
        from collections import deque
        dist = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.edges.get(current, []):
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        return dist

    def __call__(self, node):
        state = node.state

        # If goal satisfied, heuristic = 0
        if self.goal_cb6fw2xd <= state:
            return 0

        # Extract current cb6fw2xd facts satisfied
        current_cb6fw2xd = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "cb6fw2xd" and len(parts) == 3:
                current_cb6fw2xd.add((parts[1], parts[2]))

        # Extract locations of k9b7lknr agents: dict agent -> nh1sy8v9 location from ypt9mrgu predicate
        agent_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ypt9mrgu" and len(parts) == 3:
                agent, loc = parts[1], parts[2]
                agent_locations[agent] = loc

        # Extract which qekv4465 object each k9b7lknr agent currently holds (qp2t6u4s predicate)
        agent_objects = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "qp2t6u4s" and len(parts) == 3:
                agent, obj = parts[1], parts[2]
                agent_objects[agent] = obj

        # Extract the currently flagged nh1sy8v9 location (only one fact with fguw2142 predicate)
        flagged_locations = [get_parts(f)[1] for f in state if match(f, "fguw2142", "*")]
        current_flag_loc = flagged_locations[0] if flagged_locations else None

        # Heuristic cost accumulator
        cost = 0

        # For each unsatisfied goal cb6fw2xd fact, estimate cost to achieve it
        unsatisfied_goals = self.goal_cb6fw2xd - current_cb6fw2xd

        for (nh1_obj, qekv_obj) in unsatisfied_goals:
            # Find an agent that can achieve this goal (we assume all agents are viable, pick closest)
            # We try to find minimum agent movement cost + qp2t6u4s replacement + final setting cost

            # Candidate agents and their costs, will pick minimal over them
            agent_costs = []

            for agent, agent_loc in agent_locations.items():
                # Distance for agent to move from current loc to target nh1_obj location
                dist_map = self.distances.get(agent_loc, {})
                move_cost = dist_map.get(nh1_obj, float('inf'))

                # If agent cannot reach location, ignore
                if move_cost == float('inf'):
                    continue

                # Cost to replace currently held object with qekv_obj if different
                current_obj = agent_objects.get(agent, None)
                replace_cost = 0
                if current_obj != qekv_obj:
                    # At least one action of wvswfzmn required to replace held object
                    replace_cost = 1

                # Additional cost to actually produce cb6fw2xd fact presumably 1 action (kkqx0eg7 or wbsgr2yj)
                produce_cost = 1

                # Cost to move the fguw2142 flagged location close if it's not the goal location or agent current
                # Approximate cost as 1 if flagged location differs from agent or target locations (because many actions require fguw2142 move)
                flag_move_cost = 0
                if current_flag_loc not in {agent_loc, nh1_obj}:
                    flag_move_cost = 1

                total_agent_cost = move_cost + replace_cost + produce_cost + flag_move_cost

                agent_costs.append(total_agent_cost)

            if agent_costs:
                min_agent_cost = min(agent_costs)
            else:
                # No agent can reach the location, assign large cost to avoid infinite heuristic
                min_agent_cost = 50  # arbitrary large

            cost += min_agent_cost

        return cost
