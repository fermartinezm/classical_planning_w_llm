from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the tokens of a PDDL fact string without surrounding parentheses."""
    return fact.strip()[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the (obfuscated) Sokoban-like domain ("secret_domain").
    The domain's goals ask that certain boxes satisfy the predicate (adjacent-2 boxX).
    The heuristic estimates the number of actions required to achieve all such
    (adjacent-2 ...) goals from the current state by reasoning about:
      - which boxes already satisfy adjacent-2 (cost 0),
      - whether a box is already in the ephemeral "at" state (ready to be pushed),
      - which robot location is required to perform the push for each box (encoded
        in static facts of the form '(at-robot boxX locationY)'),
      - where the robot currently is ('(at-robot-2 locationZ)'),
      - which locations are connected for robot movement (static '(clear A B)' facts
        define directed move edges A->B that correspond to the move-2/push-2 actions).

    Assumptions
    -----------
    - The planner represents facts as strings such as '(predicate arg1 arg2)'.
    - Goals are only of the form '(adjacent-2 box)' (as in the provided instances).
    - Static facts include:
        * '(at-robot BOX LOCATION)' -- the robot location required to perform
          the push action that achieves adjacent-2 for BOX.
        * '(clear L1 L2)' -- a directed connectivity fact meaning the robot can move
          from L1 to L2 in one move-2 action (and if '(clear L2 L1)' also present,
          the reverse is possible).
      The heuristic uses these static facts as a road map for robot motion.
    - The domain actions behave in the following high-level way (as inferred):
        * If (adjacent BOX L) holds and the robot is at L, one 'move' action can
          make (at BOX).
        * Once (at BOX) holds and the robot is at the required push location
          (the location given by the static '(at-robot BOX LOC)'), one 'push'
          action makes (adjacent-2 BOX).
        * Robot motion between locations is performed by repeated move-2 (or push-2)
          actions along the directed edges given by 'clear' facts; each such step
          costs 1 in the heuristic.
    - The heuristic is not required to be admissible; it aims to be informative and
      cheaply computable to guide greedy best-first search.

    Heuristic Initialization
    ------------------------
    In the constructor we extract from the task:
      - self.goals: the set of goal facts (task.goals).
      - self.goal_boxes: set of boxes (names) that must satisfy adjacent-2 in the goal.
      - self.box_push_location: mapping box -> required robot location for the push,
        gathered from static facts '(at-robot BOX LOCATION)'.
      - self.move_graph: a directed adjacency mapping between locations determined
        by static '(clear X Y)' facts (edge X -> Y allows one robot move).
      - self.locations: set of all seen location names (from static facts).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state the heuristic proceeds as follows:
    1. If every goal fact is already present in the state, return 0.
    2. Parse the state to obtain:
       - robot_loc: the unique location where the robot currently is (from
         '(at-robot-2 LOC)').
       - per-box facts:
           * adjacent_locs[box] = set of locations L such that '(adjacent box L)' is true.
           * at_boxes = set of boxes for which '(at BOX)' is true (box is in the pre-push 'at' state).
    3. For each goal box that is not yet '(adjacent-2 BOX)' in the state, compute
       a small estimated cost to achieve '(adjacent-2 BOX)'. We compute the minimum
       estimate among reasonable micro-plans:
         A. If BOX is already in the 'at' state:
            - If the static mapping gives a required push location S for BOX:
                cost = distance(robot_loc, S) + 1 (one 'push' action).
            - If no mapping exists, optimistically assume push can be done where the
              robot currently is: cost = 1.
         B. If BOX is not in 'at' but has one or more 'adjacent' facts (adjacent to
            some locations L):
            For each such L:
              - cost_to_reach_L = distance(robot_loc, L)
              - one 'move' action will make BOX enter the 'at' state once robot is at L
                => +1
              - then we need to reach the push location S = box_push_location.get(BOX).
                cost_to_reach_S = distance(L, S) (if S missing, assume 0 optimistically)
              - finally one 'push' action => +1
              Total for that L: cost_to_reach_L + 1 + cost_to_reach_S + 1
            Take the minimal over all L.
         C. If no useful information is available (no 'adjacent' facts for the box,
            no 'at' fact, or distances are unreachable), assign a bounded large cost
            to ensure the heuristic is finite but discourages choices that appear hopeless.
    4. Sum the per-box estimates for all unsatisfied goal boxes. This sum is used
       as the heuristic value. The sum is admissible/optimistic in some cases but
       it does overcount shared work (e.g., moving the robot between pushes), which
       is acceptable since admissibility is not required.
    5. Return the integer heuristic estimate.

    Notes on finiteness and robustness
    ----------------------------------
    - Distances are computed using BFS on the directed graph defined by 'clear' facts.
    - If distances are not found (disconnected), the heuristic substitutes a finite
      large cost (self._LARGE_COST) for that option; this guarantees the heuristic
      returns a finite integer for any input state.
    """

    def __init__(self, task):
        # Goals and static info
        self.goals = set(task.goals)  # set of strings
        static = set(task.static)

        # Identify goal boxes: goals expected to be of form '(adjacent-2 box)'
        self.goal_boxes = set()
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "adjacent-2" and len(parts) >= 2:
                self.goal_boxes.add(parts[1])

        # Map box -> required robot location for performing the push.
        self.box_push_location = {}  # box -> location
        # Directed graph for robot movement: location -> set(neighbor_locations)
        self.move_graph = {}
        self.locations = set()

        for fact in static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at-robot" and len(parts) >= 3:
                # static '(at-robot BOX LOCATION)'
                box = parts[1]
                loc = parts[2]
                self.box_push_location[box] = loc
                self.locations.add(loc)
            elif pred == "clear" and len(parts) >= 3:
                a = parts[1]
                b = parts[2]
                self.locations.add(a)
                self.locations.add(b)
                # move-2 can move from a -> b if (clear a b)
                self.move_graph.setdefault(a, set()).add(b)

        # Ensure all locations appear in graph (even isolated)
        for loc in list(self.locations):
            self.move_graph.setdefault(loc, set())

        # Large finite fallback for unreachable options
        self._LARGE_COST = max(100, len(self.locations) * 5 + 20)

    def _bfs_distance(self, start):
        """
        Return a dict mapping location -> distance (in number of move steps)
        from start to every reachable location using the directed move_graph.
        If start is None or not in graph, returns empty dict.
        """
        if start is None or start not in self.move_graph:
            return {}
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.move_graph.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = set(node.state)

        # Checklist item 2: 0 only for goal states
        if self.goals <= state:
            return 0

        # Parse robot location in current state: '(at-robot-2 locationX)'
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot-2" and len(parts) >= 2:
                robot_loc = parts[1]
                break

        # Parse per-box facts in the current state
        adjacent_locs = {}  # box -> set(locations L such that (adjacent box L))
        at_boxes = set()    # boxes that have '(at BOX)' in state

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "adjacent" and len(parts) >= 3:
                box = parts[1]
                loc = parts[2]
                adjacent_locs.setdefault(box, set()).add(loc)
            elif pred == "at" and len(parts) >= 2:
                # domain declares at as unary on boxes: '(at box)'
                box = parts[1]
                at_boxes.add(box)

        # Precompute BFS distances from the robot current location
        dist_from_robot = self._bfs_distance(robot_loc) if robot_loc is not None else {}

        total_cost = 0

        # For each goal box not yet satisfied, compute estimated actions to achieve adjacent-2
        for box in self.goal_boxes:
            goal_fact = f"(adjacent-2 {box})"
            if goal_fact in state:
                continue  # already satisfied

            best_cost = self._LARGE_COST

            # Option A: box is already in 'at' state -> only need to move robot to push location and push
            if box in at_boxes:
                push_loc = self.box_push_location.get(box)
                if push_loc is None:
                    # Optimistic: if we don't know required push location, assume push can be done now
                    best_cost = min(best_cost, 1)
                else:
                    # distance from robot to push_loc
                    d = dist_from_robot.get(push_loc)
                    if d is not None:
                        best_cost = min(best_cost, d + 1)  # move steps + one push
                    else:
                        # maybe push_loc unreachable: use large fallback
                        best_cost = min(best_cost, self._LARGE_COST)

            # Option B: use an adjacent relation: move robot to an adjacent location L, do 'move' to get 'at', then go to push location and 'push'
            adj_locations = adjacent_locs.get(box, set())
            for L in adj_locations:
                # distance robot -> L
                d1 = dist_from_robot.get(L)
                if d1 is None:
                    # robot cannot reach this L, skip it
                    continue
                # after doing 'move' at L, box will be in 'at' state and robot still at L
                # now need to reach push location
                push_loc = self.box_push_location.get(box)
                if push_loc is None:
                    # optimistic: assume push can be done where we are
                    cost = d1 + 1 + 0 + 1  # go to L + move + (no extra move) + push
                    best_cost = min(best_cost, cost)
                else:
                    # distance L -> push_loc
                    # compute BFS from L on the fly (could be cached, but sizes are small)
                    dist_from_L = self._bfs_distance(L)
                    d2 = dist_from_L.get(push_loc)
                    if d2 is not None:
                        cost = d1 + 1 + d2 + 1  # go to L + move + go to push_loc + push
                        best_cost = min(best_cost, cost)
                    else:
                        # L cannot reach required push_loc
                        continue

            # Option C: if no adjacent facts and not at, we can attempt a fallback optimistic plan:
            # - try to move robot to the box's push location (if known) and hope it's possible to get the box into 'at' later.
            push_loc = self.box_push_location.get(box)
            if push_loc is not None:
                d = dist_from_robot.get(push_loc)
                if d is not None:
                    # d moves + 1 push (optimistic because we may need a prior move to set 'at')
                    best_cost = min(best_cost, d + 1)

            # If still unreachable or no constructive option found, assign bounded large cost (ensures finiteness)
            if best_cost == self._LARGE_COST:
                # Encourage search to still try: set cost proportional to remaining unsolved boxes
                best_cost = self._LARGE_COST

            total_cost += best_cost

        # The heuristic returns an integer estimate of remaining actions.
        return int(total_cost)