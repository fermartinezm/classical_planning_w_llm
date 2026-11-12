from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (hrpmxpjg) domain.
    The domain contains:
    - Locations (x6ibpitl) connected by directed edges (miud1izi).
    - Mobile carriers (hctyyy8k, e.g. v1, v2...) that occupy locations via (sjuejzxf carrier location).
      Carriers can move along miud1izi edges using the arsslvyt action.
    - Portable objects (du66z5od, e.g. p1, p2...) that are either located at a location via (sjuejzxf p loc)
      or are onboard a carrier via (un606moq p carrier).
    - Transfer actions (szsevy61/tmoxe3x5) that pick up or drop a portable object when carrier and object
      are co-located; these actions also modify a carrier-related predicate (do9fmoyo), but for heuristic
      purposes we treat pick/drop as single actions each (one to pick up, one to drop).

    The heuristic estimates the number of actions required to satisfy goal facts of the form
    (sjuejzxf p goal_location) for portable objects p. It assumes each pick and drop costs 1 action,
    and each carrier movement along a single miud1izi arc costs 1 action.

    Assumptions
    -----------
    - The goals we care about are of the form (sjuejzxf <object> <location>) (this matches the example
      instances).
    - Portable objects cannot move by themselves and must be transported by carriers.
    - Carriers can move along miud1izi edges (each arc costs 1 arsslvyt action).
    - Each pick (szsevy61) and drop (tmoxe3x5) is treated as a single action. The additional owner-change
      effects required by the real actions (hab0vli4 / do9fmoyo) are ignored in terms of extra action
      costs; i.e., we assume those static relations exist so pick/drop are feasible. This simplifies the
      computation and still yields a useful guidance heuristic.
    - Multiple portable objects are assumed to be independent in cost estimation (we sum per-object
      minimal transport costs). Interactions (e.g., a carrier picking multiple objects in one trip) are
      ignored; this may overestimate or underestimate in some scenarios, but keeps computation fast.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - The goal facts (task.goals).
    - The directed location graph as adjacency lists from static facts of the form (miud1izi locA locB).
      This is used to compute shortest distances (in number of arsslvyt actions) between locations.
    - A set/list of all locations seen in the miud1izi facts (used by BFS).
    - All static hab0vli4 relations are recorded (but not used to modify costs under the assumptions above).

    Step-By-Step Thinking for Computing Heuristic (for a given node/state)
    ---------------------------------------------------------------------
    1. If all goal facts are already satisfied in the given state -> return 0.

    2. Parse the current state to obtain:
       - Where each portable object p currently is:
         * If a fact (sjuejzxf p L) exists -> p is at location L.
         * If a fact (un606moq p C) exists -> p is onboard carrier C.
       - Where each carrier C currently is (sjuejzxf C L).
       - The carrier ownership facts (do9fmoyo C owner) are parsed but not required for cost arithmetic.

    3. Precompute shortest path distances between locations:
       - Using BFS on the directed graph defined by miud1izi facts from static information.
       - Distance dist(A,B) equals the minimal number of arsslvyt actions to move a carrier from A to B.
       - If no path exists between two locations, treat the distance as a large finite penalty (e.g. 1000).

    4. For each goal (sjuejzxf p goal_loc):
       - If p is already at goal_loc in the current state: cost contribution = 0.
       - Else compute the minimal estimated action-cost to get p to goal_loc:
         a) If p is onboard carrier C:
            - cost = distance(carrier_location_of_C, goal_loc) + 1 (one drop action).
         b) If p is at some location Lp:
            - For every carrier C present in the state (with known location Lc):
              estimate cost_v = distance(Lc, Lp) + 1 (pick) + distance(Lp, goal_loc) + 1 (drop).
            - The object's cost is the minimum cost_v over all carriers C.
         - If no carrier exists in the current state the cost is set to a large finite value.

    5. Sum the per-goal minimal costs into a single heuristic value and return it.

    Rationale
    ---------
    This heuristic captures the essential transport structure of the domain:
    move carriers to pickup points (arsslvyt), perform a pick (szsevy61), move to delivery points (arsslvyt),
    and perform a drop (tmoxe3x5). It is not admissible (it may underestimate or overestimate in the presence
    of owner-change constraints and multi-object carriage), but it is fast to compute and provides effective
    guidance for greedy best-first search.
    """

    def __init__(self, task):
        """
        Extract static information and goals from the task.

        - task.goals : set of goal facts (strings)
        - task.static: set of static facts (strings)
        """
        self.goals = task.goals
        static_facts = task.static

        # Build directed adjacency list for locations from miud1izi static facts.
        # miud1izi <from> <to> means carriers can move from <from> to <to> with one arsslvyt.
        self.adj = defaultdict(list)
        self.locations = set()
        self.hab_pairs = set()  # store hab0vli4 facts (kept for completeness)
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "miud1izi" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj[src].append(dst)
                self.locations.add(src)
                self.locations.add(dst)
            elif pred == "hab0vli4" and len(parts) >= 3:
                self.hab_pairs.add((parts[1], parts[2]))

        # Precompute BFS caches for distances on demand (lazy fill).
        self._dist_cache = {}

        # Bound used for unreachable pairs (keeps heuristic finite).
        self._unreachable_penalty = 1000

    def _bfs_distance(self, src, dst):
        """Return shortest path length in directed graph between two locations.
        Uses caching to avoid repeated work."""
        if src == dst:
            return 0
        key = (src, dst)
        if key in self._dist_cache:
            return self._dist_cache[key]

        # If source is unknown in adjacency (isolated), consider unreachable unless src==dst
        if src not in self.locations:
            self._dist_cache[key] = None
            return None

        visited = set()
        queue = deque([(src, 0)])
        visited.add(src)
        found = None
        while queue:
            loc, d = queue.popleft()
            for nb in self.adj.get(loc, ()):
                if nb in visited:
                    continue
                if nb == dst:
                    found = d + 1
                    queue.clear()
                    break
                visited.add(nb)
                queue.append((nb, d + 1))
        self._dist_cache[key] = found
        return found

    def __call__(self, node):
        """
        Compute heuristic estimate for the provided search node.
        """
        state = node.state

        # If all goals already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Parse current state:
        # - portable locations: either (sjuejzxf p loc) or (un606moq p carrier)
        # - carrier locations: (sjuejzxf carrier loc)
        # - carrier owners: (do9fmoyo carrier owner) (kept for completeness)
        portable_location = {}  # p -> location (if on ground)
        portable_onboard = {}  # p -> carrier (if onboard)
        carrier_location = {}  # carrier -> location
        carrier_owner = {}  # carrier -> owner (do9fmoyo)

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "sjuejzxf" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # sjuejzxf applies both to portable (p...) and carriers (v...). Record both.
                # If the object is also recorded as onboard, the onboard mapping takes precedence for portable objects.
                if obj.startswith("p") or obj.startswith("P") or obj.startswith("p"):
                    # treat as portable on ground
                    portable_location[obj] = loc
                else:
                    # treat as carrier location (carriers often start with 'v' in examples)
                    carrier_location[obj] = loc
            elif pred == "un606moq" and len(parts) >= 3:
                p, carrier = parts[1], parts[2]
                portable_onboard[p] = carrier
                # If onboard, ensure any ground mapping is removed (onboard overrides ground presence).
                if p in portable_location:
                    del portable_location[p]
            elif pred == "do9fmoyo" and len(parts) >= 3:
                carrier, owner = parts[1], parts[2]
                carrier_owner[carrier] = owner

        # If there are no carriers available in the state, we cannot move portables -> return large value.
        if not carrier_location:
            # finite but large to indicate difficulty
            return len(self.goals) * self._unreachable_penalty

        total_cost = 0

        # Extract goal mappings for relevant portable objects (goals are strings like '(sjuejzxf p1 l7)')
        goal_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "sjuejzxf" and len(parts) >= 3:
                goal_map[parts[1]] = parts[2]

        for portable, goal_loc in goal_map.items():
            # If portable already at goal in state, skip
            if f"(sjuejzxf {portable} {goal_loc})" in state:
                continue

            best_cost_for_portable = None

            # If portable is onboard some carrier C:
            if portable in portable_onboard:
                carrier = portable_onboard[portable]
                carrier_loc = carrier_location.get(carrier, None)
                if carrier_loc is None:
                    # carrier has no location info; treat as unreachable (penalty)
                    dist_to_goal = None
                else:
                    dist_to_goal = self._bfs_distance(carrier_loc, goal_loc)
                if dist_to_goal is None:
                    est = self._unreachable_penalty
                else:
                    est = dist_to_goal + 1  # moves + one drop
                best_cost_for_portable = est

            # If portable on ground at some location Lp:
            if portable in portable_location:
                Lp = portable_location[portable]
                # For each carrier compute cost: move carrier to Lp + pick + move Lp->goal + drop
                for carrier, carrier_loc in carrier_location.items():
                    dist_to_pick = self._bfs_distance(carrier_loc, Lp)
                    dist_pick_to_goal = self._bfs_distance(Lp, goal_loc)
                    if dist_to_pick is None or dist_pick_to_goal is None:
                        est = self._unreachable_penalty
                    else:
                        est = dist_to_pick + 1 + dist_pick_to_goal + 1
                    if best_cost_for_portable is None or est < best_cost_for_portable:
                        best_cost_for_portable = est

            # If portable neither onboard nor on ground (unexpected), assign large penalty.
            if best_cost_for_portable is None:
                best_cost_for_portable = self._unreachable_penalty

            total_cost += best_cost_for_portable

        return int(total_cost)