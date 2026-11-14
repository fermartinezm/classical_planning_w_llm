from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact expressed as a string.
    E.g. "(adjacent box1 location2)" -> ["adjacent", "box1", "location2"]

    This implementation removes the surrounding parentheses and splits on whitespace.
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the `sokoban` (secret_domain) encoding provided.
    The goal predicate in this domain is "adjacent-2 <box>" for each box that must be placed.
    This heuristic estimates how many operator applications are required to achieve
    all missing "adjacent-2" facts by computing, for each unsatisfied box goal, a
    small sequence of necessary actions:
      - move the robot to an appropriate location (using move-2 / push-2),
      - perform the 'move' action when necessary to enable a subsequent push,
      - move the robot to the pushing location if different,
      - perform 'push' to produce adjacent-2 for that box.

    Assumptions
    -----------
    - The static facts include:
        * (at-robot <box> <location>) : gives the robot location required by the push
          operator for that box (we call this the push-position for the box).
        * (clear <loc1> <loc2>) : defines traversability between locations.
          If "clear a b" appears, the robot can move between a and b in both directions
          (because the domain provides two movement actions that use the clear relation
          in different argument positions). Therefore we treat clear facts as undirected edges.
    - The state contains exactly one "(at-robot-2 <location>)" fact describing the current robot location.
    - "adjacent <box> <loc>" in the state indicates a location where the robot can apply
      "move" to create "(at <box>)". The "move" operator itself requires that the robot
      is at that same location.
    - The "push" operator requires the robot to be at the push-position defined by the static
      "(at-robot <box> <location>)" mapping and requires "(at <box>)" to be true.
    - If either an adjacent-location or the push-position is unreachable in the static graph,
      the heuristic assigns a large but finite penalty so the heuristic value remains finite.

    Heuristic Initialization
    ------------------------
    - Store the goal facts (task.goals).
    - Extract from static facts:
        * box_to_push_loc: mapping box -> the location required by the push operator ("at-robot" static facts).
        * location graph: an undirected adjacency list built from "clear" facts (if "clear a b" exists,
          we add an undirected edge between a and b).
    - Prepares small helper structures for fast distance computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic follows these steps:

    1. If the state already satisfies all goal facts (all goal facts in node.state),
       return 0.

    2. Determine the robot's current location by looking for the single "(at-robot-2 ...)" fact
       in the state.

    3. For each box that still needs "(adjacent-2 <box>)" (i.e. boxes referred to in goals but
       without the corresponding adjacent-2 fact in the current state):
       a. Check whether "(at <box>)" is already true in the state. If so, only a push is required
          once the robot is at the box's push-position:
            per-box cost = distance(robot_loc, push_pos) + 1 (for the push action).

       b. Otherwise, check whether there exists an "(adjacent <box> <adj_loc>)" fact in the
          current state. If so, the natural sequence is:
            - Move the robot to adj_loc (to apply "move"): cost distance(robot_loc, adj_loc).
            - Apply "move" to produce "(at <box>)": cost 1.
            - Move (possibly zero distance) from adj_loc to push_pos: cost distance(adj_loc, push_pos).
            - Apply "push" to obtain "adjacent-2": cost 1.
          So per-box cost = dist(robot, adj_loc) + 1 + dist(adj_loc, push_pos) + 1.

       c. If neither "(at <box>)" nor "(adjacent <box> ...)" is present, we conservatively
          assume the box still needs an at-step and a push-step; we estimate:
            per-box cost = dist(robot, push_pos) + 2
          (this is optimistic but finite; it assumes that some adjacency can be created
           without additional impossible actions.)

       d. Distances are computed on the undirected graph formed by static "clear" facts.
          If a required location is not reachable (no path), the distance is treated as a
          large finite penalty (e.g., 1000) to keep the heuristic finite while indicating
          that the box is hard to reach.

    4. The heuristic value is the sum of the per-box costs for all unsatisfied boxes.
       This is a domain-specific, informative estimate that accounts for robot travel
       and the required local actions (move/push). It is not admissible (it may overcount),
       but it is designed to reduce search expansions by being informative and cheap to compute.
    """

    def __init__(self, task):
        """Extract static information (push positions and clear graph) and goals."""
        self.goals = set(task.goals)

        # Map each box -> push-position location from static fact (at-robot box location).
        self.box_to_push_loc = {}
        # Undirected graph of locations built from "clear" facts. Each clear(u,v) provides an undirected edge.
        self.loc_graph = {}

        for fact in task.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at-robot" and len(parts) >= 3:
                # (at-robot boxX locationY)
                box = parts[1]
                loc = parts[2]
                self.box_to_push_loc[box] = loc
            elif pred == "clear" and len(parts) >= 3:
                a = parts[1]
                b = parts[2]
                # add undirected edge a <-> b
                self.loc_graph.setdefault(a, set()).add(b)
                self.loc_graph.setdefault(b, set()).add(a)

        # Cache BFS distances per source location to avoid repeated BFS calls
        self._dist_cache = {}

        # A large finite penalty for unreachable distances (keeps heuristic finite).
        self._BIG = 1000

    def _bfs_distances(self, start):
        """
        Return dict: location -> distance from start, computed with BFS on loc_graph.
        Caches results per start. If start is not present as a node in loc_graph, returns {start:0}.
        """
        if start in self._dist_cache:
            return self._dist_cache[start]

        if start is None:
            d = {}
            self._dist_cache[start] = d
            return d

        if start not in self.loc_graph:
            # start might be isolated (no clear facts incident). distance to itself is 0.
            d = {start: 0}
            self._dist_cache[start] = d
            return d

        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.loc_graph.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[start] = dist
        return dist

    def __call__(self, node):
        """Compute and return the estimated number of actions to reach the goal from node.state."""
        state = node.state

        # If goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Find robot current location from "(at-robot-2 <location>)" in the state.
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot-2" and len(parts) >= 2:
                robot_loc = parts[1]
                break

        # Gather which boxes already have adjacent-2 (goal) in the current state.
        have_adjacent2 = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent-2" and len(parts) >= 2:
                have_adjacent2.add(parts[1])

        # Build quick lookup sets for "at" and "adjacent" facts in the current state.
        at_boxes = set()
        box_adj_locations = {}  # box -> list of adj locations (usually one)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "at" and len(parts) >= 2:
                at_boxes.add(parts[1])
            elif parts[0] == "adjacent" and len(parts) >= 3:
                box = parts[1]
                loc = parts[2]
                box_adj_locations.setdefault(box, []).append(loc)

        total_cost = 0

        # Precompute BFS distances from robot location
        robot_dists = self._bfs_distances(robot_loc)

        # For each goal box (those appearing in self.goals as adjacent-2 facts),
        # if not already satisfied, estimate cost.
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts:
                continue
            if parts[0] != "adjacent-2" or len(parts) < 2:
                # ignore non-adjacent-2 goals (defensive)
                continue
            box = parts[1]
            if box in have_adjacent2:
                continue  # already satisfied

            # Determine push position for this box (from static).
            push_pos = self.box_to_push_loc.get(box, None)

            # Helper to get distance between two locations (uses BFS and _BIG penalty if unreachable).
            def dist(a, b):
                if a == b:
                    return 0
                if a is None or b is None:
                    return self._BIG
                d_from_a = self._bfs_distances(a)
                if b in d_from_a:
                    return d_from_a[b]
                # If b not reachable from a in the static movement graph, give large penalty
                return self._BIG

            # Case A: box already has (at box)
            if box in at_boxes:
                # Only need to move robot to push_pos and execute push.
                if push_pos is None:
                    # If push position unknown, assume at least 1 for push plus some travel
                    per_box_cost = 1 + self._BIG // 10
                else:
                    per_box_cost = dist(robot_loc, push_pos) + 1  # +1 for push
                total_cost += per_box_cost
                continue

            # Case B: there is an adjacent location where 'move' can create (at box)
            adj_locs = box_adj_locations.get(box, [])
            if adj_locs:
                # Choose the adj location that gives the minimal estimate
                best = None
                for adj_loc in adj_locs:
                    # travel to adj_loc from robot, move (1), travel from adj_loc to push_pos, push (1)
                    if push_pos is None:
                        # if push_pos unknown, penalize but still compute travel to adj_loc
                        cost = dist(robot_loc, adj_loc) + 1 + self._BIG // 10 + 1
                    else:
                        cost = dist(robot_loc, adj_loc) + 1 + dist(adj_loc, push_pos) + 1
                    if best is None or cost < best:
                        best = cost
                total_cost += best if best is not None else self._BIG
                continue

            # Case C: no (at box) and no (adjacent box loc) in state.
            # We conservatively estimate: travel to push_pos and pay for two local actions (create-at + push).
            if push_pos is None:
                per_box_cost = 2 + self._BIG // 10
            else:
                per_box_cost = dist(robot_loc, push_pos) + 2
            total_cost += per_box_cost

        # Return integer heuristic value
        return int(total_cost)