from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components from a PDDL fact string like '(predicate obj1 obj2)'."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards '*'.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (hi4l5zmb).

    Summary:
    This heuristic estimates the number of actions needed to achieve goal facts 
    of the form `(cb6fw2xd ?nh1sy8v9 ?qekv4465)`, which represent a key goal condition 
    of the domain. The heuristic focuses on counting goal facts not yet satisfied, 
    and heuristically estimates the minimum number of "moves" and "transfers" 
    of `k9b7lknr` objects (agents/actors) between `nh1sy8v9` locations, as well as 
    assigning `qekv4465` objects, based on static connectivity relations.

    Assumptions:
    - The goal requires certain `(cb6fw2xd ?loc ?obj)` facts to be true.
    - A `(cb6fw2xd ?loc ?obj)` fact can only be achieved by transporting the object `?obj`
      to the location `?loc` via some chain of moves/assignments.
    - The domain predicates `ga4l9do0`, `s4jrpk2b`, `vjcqgdgn`, `vom4pfig` describe different 
      connectivity relations or adjacency graphs over `nh1sy8v9` objects.
    - `ypt9mrgu` relates `k9b7lknr` objects (agents) to their current `nh1sy8v9` location.
    - `fguw2142` marks "free" or special locations/objects involved in moves.
    - The domain actions reflect operations transporting `k9b7lknr` agents or changing 
      assigned objects along these connectivity links.
    - We assume shortest path distances over these adjacency relations can roughly reflect 
      the minimal action sequences required to achieve the goal facts.

    Heuristic Initialization:
    - From the static facts, extract the connectivity relations between `nh1sy8v9` objects:
      - `adj_s4jrpk2b` from `(s4jrpk2b ?from ?to)`
      - `adj_ga4l9do0` from `(ga4l9do0 ?from ?to)`
      - `adj_vjcqgdgn` from `(vjcqgdgn ?from ?to)`
      - `adj_vom4pfig` from `(vom4pfig ?from ?to)`
    - Extract all `k9b7lknr` objects (agents) by looking for `(ypt9mrgu ?agent ?loc)`
    - Extract all `qekv4465` objects (items) from static facts `(fezojtj3 ?obj)`
    - Extract all goal facts of type `(cb6fw2xd ?loc ?obj)`.
    - Precompute shortest path distance maps (using BFS) between `nh1sy8v9` locations for 
      each connectivity relation needed to approximate movement costs.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each goal fact `(cb6fw2xd ?loc ?obj)` not yet in the current state, estimate 
       how many steps are needed to produce it.
    2. To produce `(cb6fw2xd ?loc ?obj)`, presumably an agent `?en18p` must bring `?obj` 
       from some starting location to `?loc`, subject to connectivity constraints.
    3. The heuristic considers:
       - The minimal distance to bring the required agent to the object's starting location.
       - The minimal distance to move the object to the goal location using the adjacency graphs.
    4. Use shortest paths on `ga4l9do0` and `s4jrpk2b`/`vom4pfig`/`vjcqgdgn` graphs to estimate travel costs.
    5. Incorporate presence of `fguw2142` ("free") locations to estimate moves.
    6. Sum up these minimal estimated moves per goal fact to estimate total required actions.
    7. Return 0 if all goal facts satisfied (goal state).

    Note:
    - The heuristic is not admissible.
    - The heuristic avoids costly computations by limiting BFS search and relying on static relations.
    """

    def __init__(self, task):
        """Initialize heuristic by extracting static connectivity info and goal facts."""
        self.goals = task.goals

        # Extract all static connectivity relations as adjacency dicts for quick access:
        # Maps from nh1sy8v9 object to set of connected nh1sy8v9 objects.
        self.adj_s4jrpk2b = {}
        self.adj_ga4l9do0 = {}
        self.adj_vjcqgdgn = {}
        self.adj_vom4pfig = {}

        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "s4jrpk2b":
                a, b = parts[1], parts[2]
                self.adj_s4jrpk2b.setdefault(a, set()).add(b)
            elif pred == "ga4l9do0":
                a, b = parts[1], parts[2]
                self.adj_ga4l9do0.setdefault(a, set()).add(b)
            elif pred == "vjcqgdgn":
                a, b = parts[1], parts[2]
                self.adj_vjcqgdgn.setdefault(a, set()).add(b)
            elif pred == "vom4pfig":
                a, b = parts[1], parts[2]
                self.adj_vom4pfig.setdefault(a, set()).add(b)

        # Extract all goal (cb6fw2xd ...) facts with their parameters
        self.goal_cb6fw2xd = []  # list of (loc: nh1sy8v9, obj: qekv4465)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "cb6fw2xd":
                self.goal_cb6fw2xd.append((parts[1], parts[2]))

        # Extract all known nh1sy8v9 and qekv4465 objects from static facts (for validation)
        self.all_nh1sy8v9 = set()
        self.all_qekv4465 = set()
        for fact in task.static:
            parts = get_parts(fact)
            if len(parts) >= 2:
                if parts[0] in ["s4jrpk2b", "ga4l9do0", "vjcqgdgn", "vom4pfig", "fguw2142", "cb6fw2xd", "fezojtj3", "fguw2142"]:
                    for obj in parts[1:]:
                        # nh1sy8v9 type predicates mostly have first or second arg as nh1sy8v9; fezojtj3 is qekv4465
                        # We accept them conservatively
                        self.all_nh1sy8v9.add(obj)
                if parts[0] == "fezojtj3":
                    self.all_qekv4465.add(parts[1])

    def shortest_path(self, adj, start, goal):
        """Compute shortest path distance from start to goal over adjacency `adj` dict (directed).

        If no path, return None."""
        from collections import deque

        if start == goal:
            return 0
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == goal:
                return dist
            if node in visited:
                continue
            visited.add(node)
            for neigh in adj.get(node, ()):
                if neigh not in visited:
                    queue.append((neigh, dist + 1))
        return None

    def __call__(self, node):
        """
        Estimate the number of remaining actions to reach the goal from the current state.

        Heuristic logic:
        - Count how many goal (cb6fw2xd loc obj) facts remain unsatisfied.
        - For each such goal, estimate minimal actions needed:
          - Consider reachable starting locations of obj (from qp2t6u4s facts).
          - Consider agent (`k9b7lknr` type) currently at some location (ypt9mrgu).
          - Estimate cost to transport obj from its current position to goal loc via
            adjacency `ga4l9do0` graph plus transfers involving `fguw2142`.
          - Estimate cost for agent to move appropriately in `nh1sy8v9` graph(s).
        - Summate minimal costs for all unmet goals.
        """
        state = node.state

        # Fast test: if goal is reached, heuristic = 0
        if self.goals <= state:
            return 0

        # Extract facts for fast queries, build lookup tables:
        # Mapping: qp2t6u4s (agent, obj)
        qp2t6u4s_map = set()
        # Mapping: ypt9mrgu (agent, location)
        agent_location = {}
        # Set: fguw2142 (locations considered free)
        free_locations = set()
        # For quick lookup: current (cb6fw2xd loc obj) in state
        cb6fw2xd_in_state = set()
        # Objects' current known locations to aid estimate
        obj_locations = {}  # obj -> location (qekv4465 -> nh1sy8v9) inferred from cb6fw2xd or qp2t6u4s/others

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "qp2t6u4s" and len(parts) == 3:
                qp2t6u4s_map.add((parts[1], parts[2]))
            elif pred == "ypt9mrgu" and len(parts) == 3:
                agent_location[parts[1]] = parts[2]
            elif pred == "fguw2142" and len(parts) == 2:
                free_locations.add(parts[1])
            elif pred == "cb6fw2xd" and len(parts) == 3:
                cb6fw2xd_in_state.add((parts[1], parts[2]))

        # To estimate object location:
        # If (cb6fw2xd loc obj) in state, then object obj is at loc.
        # Else use qp2t6u4s(agent, obj) and agent_location[agent] to guess object location is agent's location.
        for loc, obj in cb6fw2xd_in_state:
            obj_locations[obj] = loc
        for agent, obj in qp2t6u4s_map:
            if obj not in obj_locations and agent in agent_location:
                obj_locations[obj] = agent_location[agent]

        # For each goal (loc,obj) not yet satisfied, estimate cost.
        total_cost = 0

        for goal_loc, goal_obj in self.goal_cb6fw2xd:
            if (goal_loc, goal_obj) in cb6fw2xd_in_state:
                # This goal fact already achieved, skip.
                continue

            # Estimate cost to bring goal_obj to goal_loc.

            # 1) Find current location of the object
            obj_loc = obj_locations.get(goal_obj, None)
            if obj_loc is None:
                # Unknown object location; pessimistically set cost high (e.g. 5)
                obj_loc = None
                object_move_cost = 5
            else:
                # 2) Estimate shortest path over ga4l9do0 graph from obj_loc to goal_loc.
                d_obj = self.shortest_path(self.adj_ga4l9do0, obj_loc, goal_loc)
                if d_obj is None:
                    # No path known, set cost high
                    object_move_cost = 5
                else:
                    # Model: each edge roughly corresponds to 1 action (or small constant)
                    # Adding some fixed cost for loading/unloading etc.
                    object_move_cost = d_obj + 2  # +2 for loading/unloading cost

            # 3) Estimate agent movement cost: minimal distance from any agent's current location to obj_loc
            min_agent_move = None
            for agent, ag_loc in agent_location.items():
                d_agent = self.shortest_path(self.adj_s4jrpk2b, ag_loc, obj_loc) if obj_loc else None
                if d_agent is not None:
                    if min_agent_move is None or d_agent < min_agent_move:
                        min_agent_move = d_agent
            if min_agent_move is None:
                # No agent location known / reachable, pessimistic cost
                agent_move_cost = 5
            else:
                agent_move_cost = min_agent_move

            # 4) Account for whether the object or agent is currently on a free location (fguw2142)
            # Being on free locations might save moves.
            free_bonus = 0
            if obj_loc in free_locations:
                free_bonus -= 1  # reduce cost
            for ag_loc in agent_location.values():
                if ag_loc in free_locations:
                    free_bonus -= 1

            # 5) Total estimate for this goal:
            # Sum agent move cost + object move cost, subtract free bonus (min zero)
            est_cost = max(agent_move_cost + object_move_cost + free_bonus, 1)

            total_cost += est_cost

        return total_cost
