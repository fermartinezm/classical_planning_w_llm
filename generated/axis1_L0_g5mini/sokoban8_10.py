from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing the surrounding parentheses
    and splitting the internals on whitespace.

    Example:
      "(at box1 loc_4_4)" -> ["at", "box1", "loc_4_4"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the Sokoban domain. The heuristic estimates
    the number of required actions (moves + pushes) to get all boxes to their
    goal positions. It primarily counts the minimal number of box pushes
    necessary (each push moves a box one cell), and adds an estimate of the
    robot movement required to reach the first push position.

    Assumptions
    -----------
    - Goals are expressed as (at <box> <location>) facts; each box that appears
      in the goals has exactly one target location.
    - Static `adjacent` facts describe valid topological connections between
      locations. A box can be pushed from loc A to loc B iff there is a static
      fact (adjacent A B <dir>). The push requires the robot to be on the
      predecessor cell rloc with a corresponding (adjacent rloc A <dir>) fact.
    - The heuristic ignores dynamic blocking by other boxes when computing the
      minimal number of pushes (i.e., it assumes the box can be moved along a
      free path). This makes the heuristic faster to compute and often
      informative, though not admissible.
    - If a box cannot reach its goal through the static adjacency graph, a
      large finite penalty is returned (rather than infinity) so the heuristic
      remains finite for all states.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts:
    - self.goals: the set of goal facts (for quick 0-check).
    - self.goal_locations: a dict mapping each box to its goal location.
    - self.adj: a dict mapping each location to a list of (neighbor_location, dir)
      for all static adjacent facts. This is used to compute shortest push
      distances (in pushes).
    - self.out_neighbors: a dict mapping each location to the set of outgoing
      neighbors (locations that can be reached by a single push).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts are already satisfied in the state, return 0.
    2. Extract current robot location, the current location of every box, and
       the set of clear locations from the state's facts.
       - get_parts(...) is used to parse facts after removing parentheses.
    3. For each box that is not already at its goal:
       a) Compute the minimal number of pushes required to move the box from
          its current location to its goal location. This is a simple BFS on
          the static adjacency graph (each edge corresponds to a single push).
          If no path exists, assign a large finite penalty for that box.
       b) Sum these minimal push distances for all boxes -> sum_pushes.
    4. Estimate the robot movement cost to begin pushing:
       - For each box that needs pushing, compute the set of valid robot start
         positions rloc (static predecessors) that allow pushing that box in
         some direction (i.e., there exists adjacency(rloc, box, dir) and
         adjacency(box, floc, dir)). From the robot's actual location compute
         BFS distances over the movement graph consisting of currently clear
         locations and static adjacency relations.
       - Find the minimal robot distance to any valid rloc among all boxes;
         call this robot_min_dist. If no such rloc is reachable (e.g., blocked
         by other boxes), we conservatively set robot_min_dist = 0 so we do not
         produce infinite values (the push-sum still provides guiding signal).
    5. The heuristic value is:
         heuristic = sum_pushes + robot_min_dist
       This is cheap to compute (few BFS calls on the static graph) and tends
       to guide greedy best-first search towards reducing the dominant cost
       (pushes).
    """

    # A finite large penalty returned for unreachable situations (keeps heuristic finite).
    LARGE_PENALTY = 10000

    def __init__(self, task):
        """
        Extract static adjacency information and goal mapping from the task.
        """
        # Store goals for the goal-check optimization.
        self.goals = task.goals

        # Static facts (adjacency is static in Sokoban).
        static_facts = task.static

        # adjacency map: loc -> list of (neighbor_loc, dir)
        self.adj = {}
        # outgoing neighbor set: loc -> set(neighbors) (useful for BFS)
        self.out_neighbors = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) >= 4:
                _, l1, l2, d = parts
                self.adj.setdefault(l1, []).append((l2, d))
                self.out_neighbors.setdefault(l1, set()).add(l2)

        # Extract goal location for each box from task.goals
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                _, obj, loc = parts
                # Save mapping for boxes - if there are other at-goals they are stored too.
                self.goal_locations[obj] = loc

    def _bfs_push_distance(self, start, target):
        """
        BFS on the static adjacency graph where each edge corresponds to
        a single push of a box from one location to a neighboring location.
        Returns the number of pushes (edges) in the shortest path or None if
        unreachable.
        """
        if start == target:
            return 0
        visited = {start}
        q = deque([(start, 0)])
        while q:
            loc, dist = q.popleft()
            for neigh, _dir in self.adj.get(loc, ()):
                if neigh in visited:
                    continue
                if neigh == target:
                    return dist + 1
                visited.add(neigh)
                q.append((neigh, dist + 1))
        return None

    def _bfs_robot_distances(self, robot_loc, clear_set):
        """
        BFS to compute shortest robot movement distances (in move actions)
        from robot_loc to all reachable clear locations. The robot can move
        along static adjacent edges, but only enter locations that are currently clear.
        Returns a dict loc -> distance.
        """
        dist = {}
        if robot_loc not in clear_set:
            # In practice robot location should be clear, but add it to allow BFS.
            clear_set = set(clear_set)
            clear_set.add(robot_loc)

        q = deque([(robot_loc, 0)])
        dist[robot_loc] = 0
        while q:
            loc, d = q.popleft()
            for neigh, _dir in self.adj.get(loc, ()):
                if neigh not in clear_set:
                    continue
                if neigh in dist:
                    continue
                dist[neigh] = d + 1
                q.append((neigh, d + 1))
        return dist

    def __call__(self, node):
        """
        Compute heuristic estimate for the given search node (node.state is a frozenset of facts).
        Returns an integer estimate (the lower the better).
        """
        state = node.state

        # 2) Return 0 if state satisfies all goals.
        if self.goals <= state:
            return 0

        # Extract robot location.
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) == 2:
                robot_loc = parts[1]
                break

        # Extract clear locations.
        clear_set = {get_parts(f)[1] for f in state if get_parts(f)[0] == "clear"}

        # Extract box locations (facts of form "(at boxX locY)").
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                _, obj, loc = parts
                box_locations[obj] = loc

        # Sum minimal push distances for all boxes that are not at their goal.
        sum_pushes = 0
        boxes_to_move = []
        for box, curr_loc in box_locations.items():
            goal_loc = self.goal_locations.get(box)
            if goal_loc is None:
                # If a box has no explicit goal, we assume it need not be moved.
                continue
            if curr_loc == goal_loc:
                continue
            # compute minimal number of pushes from curr_loc to goal_loc using static graph
            dist_push = self._bfs_push_distance(curr_loc, goal_loc)
            if dist_push is None:
                # unreachable via pushes through static adjacency -> large penalty
                sum_pushes += self.LARGE_PENALTY
            else:
                sum_pushes += dist_push
            boxes_to_move.append((box, curr_loc, goal_loc))

        # If no boxes need moving, but goals not satisfied (maybe robot needs to move), return small cost.
        if not boxes_to_move:
            # If goals are not subset of state but no boxes need moving, produce a small heuristic.
            return 1

        # Compute robot distances to reachable clear locations.
        robot_min_dist = 0
        if robot_loc is not None:
            robot_distances = self._bfs_robot_distances(robot_loc, clear_set)
            # For each box, identify valid robot starting positions (rloc) which allow a push:
            # rloc must satisfy (adjacent rloc box dir) and (adjacent box floc dir) for some dir.
            # rloc must be clear (robot can stand there).
            best = None
            for (_box, box_loc, _goal) in boxes_to_move:
                # find rloc candidates
                rloc_candidates = []
                # examine all static adjacency facts of form adjacent(rloc, box_loc, dir)
                # i.e., entries in adjacency map that point to box_loc; we need to find such rloc.
                # Since self.adj maps from loc -> [(neigh, dir)], scan keys for neighbours equal to box_loc.
                # For efficiency, iterate over keys in adj (small compared to overall domain sizes here).
                for rloc, neighs in self.adj.items():
                    for neigh, dir_ in neighs:
                        if neigh != box_loc:
                            continue
                        # check there exists floc such that (adjacent box_loc floc dir_) holds
                        flocs = [fl for fl, d2 in self.adj.get(box_loc, ()) if d2 == dir_]
                        if not flocs:
                            continue
                        # rloc is a valid candidate; rloc must be in clear set for robot to stand there
                        if rloc in clear_set:
                            rloc_candidates.append(rloc)
                # From these rloc candidates compute minimal robot distance
                local_min = None
                for rloc in rloc_candidates:
                    d = robot_distances.get(rloc)
                    if d is None:
                        continue
                    if local_min is None or d < local_min:
                        local_min = d
                if local_min is not None:
                    if best is None or local_min < best:
                        best = local_min
            if best is not None:
                robot_min_dist = best
            else:
                # If robot cannot currently reach any valid pushing position (likely blocked by boxes),
                # do not return infinity. Keep robot contribution small so the heuristic remains finite.
                robot_min_dist = 0
        else:
            # If robot location unknown (shouldn't happen), don't add robot cost.
            robot_min_dist = 0

        heuristic_value = int(sum_pushes + robot_min_dist)
        return heuristic_value