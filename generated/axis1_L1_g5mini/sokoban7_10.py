from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (Sokoban-like) domain.
    The domain contains:
      - zdufuwne <loc>: the robot/player location.
      - ldlu1rwo <box> <loc>: a box at a location.
      - qsw2k0xh <loc>: a location that is currently free (walkable / targetable).
      - h80z8tvp <u> <v> <label>: a directed, labeled adjacency (used for movement
        and for the special push rule which requires two consecutive edges with
        the same label).

    The heuristic focuses on estimating the number of box-push actions required
    to place all boxes at their goal locations (ldlu1rwo facts in the goal),
    and adds a small estimate for how far the robot is from performing the
    first necessary push. It intentionally ignores complex interactions
    (like boxes blocking other boxes) to remain efficient; therefore the
    heuristic is not admissible but is informative for greedy best-first search.

    Assumptions
    -----------
    - Goals are given as ldlu1rwo <box> <location> facts (one or more boxes).
    - The static relation h80z8tvp lists directed edges between locations with
      labels; a push of a box from v->w is possible if there exists u such
      that h80z8tvp(u,v,label) and h80z8tvp(v,w,label). We call such (v->w)
      a "push-edge".
    - Robot walking moves (without pushing) use single h80z8tvp edges and the
      destination must be marked qsw2k0xh (free) in the current state.
    - For efficiency the heuristic ignores dynamic blocking of push-paths by
      other boxes and computes push-distances on the abstract "push graph".

    Heuristic Initialization
    ------------------------
    The constructor extracts and prepares static structures:
      - self.goals: the set of goal facts (task.goals).
      - self.goal_positions: maps box -> goal_location (from goals).
      - self.outgoing: dict mapping location -> list of (neighbor, label) from
        static h80z8tvp facts.
      - self.incoming: dict mapping location -> list of (predecessor, label).
      - self.push_graph: dict mapping location -> set of successor locations
        that a box can be pushed into (v -> w) whenever there is some u with
        h80z8tvp(u,v,label) and h80z8tvp(v,w,label).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state we compute the heuristic as follows:

    1. Early exit: if all goals are already satisfied in the state -> return 0.

    2. Extract dynamic information from the state:
       - robot_pos: the unique location where (zdufuwne ...) is true (if any).
       - box_positions: dict mapping box -> current_location via ldlu1rwo facts.
       - free_set: the set of locations that are currently free (qsw2k0xh facts).

    3. Compute minimal pushes required for each box:
       - For each box that has a specific goal location (from the goal set),
         perform a BFS on the precomputed push_graph (which is derived only
         from static edges and labels) from the box's current location to its
         goal location. The distance in this BFS is the minimal number of
         push actions required ignoring other boxes.
       - If a box's goal is unreachable in the push_graph, assign a large
         penalty for that box (finite but large) so the heuristic remains
         finite.

    4. Estimate robot travel needed to start pushing:
       - For every box that requires at least one push, collect the set of
         predecessor positions `u` from which the first push (v->w) can be
         executed. A predecessor u qualifies if there exists a label `a` and
         a target w in the current free_set with:
             h80z8tvp(u, v, a) and h80z8tvp(v, w, a)
       - Compute a shortest-path BFS for the robot in the walk graph (directed
         static edges), where transitions are only allowed into locations that
         are currently free (qsw2k0xh). The robot BFS starts from the
         robot_pos (the start may be not-free; we allow the start node but
         subsequent moves require qsw2k0xh on the destination).
       - The robot travel estimate is the minimum BFS distance (in number of
         move actions) from robot_pos to any such predecessor u among all
         boxes that need pushing. If such a u is not reachable, a small
         penalty is added (finite).

    5. Combine costs:
       - total_pushes = sum of minimal pushes for all boxes (or the large
         penalty for unreachable boxes).
       - heuristic_value = total_pushes + robot_travel_estimate
       - If there are no boxes that need pushing, heuristic_value == 0.

    Notes on design choices:
    - The heuristic intentionally ignores multi-box interactions and the
      exact ordering of moves/pushes to remain computationally cheap.
    - The push-graph is computed once from static facts, making per-state
      computation limited to BFS on small graphs (number of locations).
    - Large penalties are finite integers so the heuristic never returns infinity.
    """

    def __init__(self, task):
        """Extract static information and goals from the task."""
        self.goals = task.goals
        static_facts = task.static

        # Goal mapping: box -> target_location for goal ldlu1rwo facts.
        self.goal_positions = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                self.goal_positions[box] = loc

        # Build adjacency lists from static h80z8tvp facts:
        # outgoing[u] = list of (v, label) for h80z8tvp u v label
        # incoming[v] = list of (u, label)
        self.outgoing = {}
        self.incoming = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "h80z8tvp" and len(parts) >= 4:
                u, v, label = parts[1], parts[2], parts[3]
                self.outgoing.setdefault(u, []).append((v, label))
                self.incoming.setdefault(v, []).append((u, label))

        # Build push-graph: a box at v can be pushed to w if there exists label a
        # and some u such that h80z8tvp(u,v,a) and h80z8tvp(v,w,a).
        self.push_graph = {}
        # For each location v, check outgoing (v->w,label) and incoming (u->v,label)
        for v in set(list(self.incoming.keys()) + list(self.outgoing.keys())):
            outs = self.outgoing.get(v, [])
            ins = self.incoming.get(v, [])
            # For each label that appears both in incoming and outgoing, all pairs
            # of (incoming u with label) and (outgoing w with label) allow push v->w.
            # We only need v->w edges for distance calculations.
            label_to_outs = {}
            for (w, lab) in outs:
                label_to_outs.setdefault(lab, []).append(w)
            for (u, lab) in ins:
                if lab in label_to_outs:
                    for w in label_to_outs[lab]:
                        self.push_graph.setdefault(v, set()).add(w)
        # Convert sets to lists for consistent iteration
        for k in list(self.push_graph.keys()):
            self.push_graph[k] = list(self.push_graph[k])

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given search node.

        Returns an integer estimate: number_of_pushes_needed + robot_travel_estimate.
        """
        state = node.state

        # 1) If all goals already satisfied -> heuristic = 0
        if self.goals <= state:
            return 0

        # 2) Extract dynamic facts: robot position, box positions, free tiles
        robot_pos = None
        box_positions = {}  # box -> loc
        free_set = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "zdufuwne" and len(parts) >= 2:
                robot_pos = parts[1]
            elif parts[0] == "ldlu1rwo" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                box_positions[box] = loc
            elif parts[0] == "qsw2k0xh" and len(parts) >= 2:
                free_set.add(parts[1])

        # Safety defaults
        if robot_pos is None:
            # If robot not present in state (shouldn't happen), assume some cost.
            robot_pos = next(iter(box_positions.values())) if box_positions else "unknown"

        # 3) For each box, compute minimal number of pushes to its goal using BFS
        LARGE_PENALTY = 1000  # finite large number to indicate unreachable in push graph
        total_pushes = 0
        boxes_requiring_pushes = []  # store boxes that need at least one push with current loc != goal

        for box, current_loc in box_positions.items():
            goal_loc = self.goal_positions.get(box)
            if goal_loc is None:
                # If box has no specific goal, ignore it (no pushes required for heuristic)
                continue
            if current_loc == goal_loc:
                continue  # already at goal, no pushes needed

            # BFS on push_graph from current_loc to goal_loc
            visited = {current_loc}
            dq = deque([(current_loc, 0)])
            found_distance = None
            while dq:
                loc, dist = dq.popleft()
                if loc == goal_loc:
                    found_distance = dist
                    break
                for succ in self.push_graph.get(loc, []):
                    if succ not in visited:
                        visited.add(succ)
                        dq.append((succ, dist + 1))
            if found_distance is None:
                # mark unreachable with penalty but finite
                pushes = LARGE_PENALTY
            else:
                pushes = found_distance
            total_pushes += pushes
            if pushes > 0:
                boxes_requiring_pushes.append((box, current_loc))

        # If no boxes require pushing (or no boxes with goals), heuristic is 0
        if total_pushes == 0:
            return 0

        # 4) Robot travel estimate: minimal walking moves to reach a valid predecessor
        #    node u for the first push of any box that needs pushes.
        # For a box at v, a predecessor u is valid if there exists label a and w in free_set
        # with h80z8tvp(u, v, a) and h80z8tvp(v, w, a).
        candidate_predecessors = set()
        for _, v in boxes_requiring_pushes:
            outs = self.outgoing.get(v, [])
            # find outgoing (v->w,label) with w free
            outs_free = [(w, lab) for (w, lab) in outs if w in free_set]
            if not outs_free:
                continue
            ins = self.incoming.get(v, [])
            # for each outgoing free edge label, find incoming with same label
            for (w, lab) in outs_free:
                for (u, lab_in) in ins:
                    if lab_in == lab:
                        candidate_predecessors.add(u)

        # If no candidate predecessors found (no possible immediate push given current free tiles),
        # add a small penalty to the heuristic and return sum of pushes + penalty.
        if not candidate_predecessors:
            robot_travel = 2  # small finite penalty
            return total_pushes + robot_travel

        # BFS for robot moves: start from robot_pos. The robot can move along an edge u->v
        # only if v is currently free. The starting node is allowed even if not free.
        dist = {robot_pos: 0}
        dq = deque([robot_pos])
        found_min = None
        while dq:
            u = dq.popleft()
            if u in candidate_predecessors:
                found_min = dist[u]
                break
            # traverse outgoing edges u->v where v is free
            for (v, _) in self.outgoing.get(u, []):
                if v not in dist and v in free_set:
                    dist[v] = dist[u] + 1
                    dq.append(v)

        if found_min is None:
            # No reachable predecessor: small penalty but finite
            robot_travel = 5
        else:
            robot_travel = found_min

        heuristic_value = total_pushes + robot_travel
        return heuristic_value