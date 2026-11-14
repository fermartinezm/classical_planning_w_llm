from collections import deque
from itertools import permutations
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the Sokoban domain. It estimates the number of
    actions (robot moves + pushes) required to reach the goal by combining:
      - A relaxed estimate of the number of pushes required to move each box to its
        goal location (shortest-path distance on the location graph).
      - A lightweight estimate of robot motion needed to start pushing (distance to
        a box-adjacent square).
    The heuristic is not required to be admissible; it aims to provide a sharp and
    quickly computable estimate to guide a greedy best-first search.

    Assumptions
    -----------
    - Goal facts are of the form "(at <box> <location>)" and specify exactly which
      box must be at which goal location. (This matches the provided instances.)
    - The static facts include "(adjacent <loc1> <loc2> <dir>)" for traversable
      location pairs. We treat adjacency as undirected if both directions are present
      in the static facts (the static input typically contains both directions).
    - For heuristic computation we relax blocking: other boxes are ignored as
      obstacles for robot movement and for box push feasibility. This yields a
      fast (optimistic) estimate of push distances.
    - Distances are unweighted: each robot move and each push counts as cost 1.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
      - Extracts goal locations (mapping box -> goal_location).
      - Builds the static location graph from "(adjacent ...)" facts.
      - Precomputes shortest-path distances (in number of steps) between all
        locations using BFS for each source location. These distances are used
        as the relaxed number of pushes required to move a box from A to B and
        as robot travel distances.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state, the heuristic computes an estimate as follows:

    1. Goal check:
       - If all goal facts are already satisfied in the current state, return 0.

    2. Parse state:
       - Determine the robot's current location from the fact "(at-robot <loc>)".
       - Determine each box's current location from facts "(at <box> <loc>)".

    3. Push-distance for boxes (relaxed):
       - For each box, compute the shortest-path distance (in number of location
         steps) between its current location and its goal location using the
         precomputed location distances. This distance is interpreted as the
         minimal number of pushes required to move the box to the goal (one push
         moves the box by one adjacency step).

    4. Robot travel estimate to start pushing:
       - To perform the first push on a box, the robot must be on a square
         adjacent to that box. For each box, consider all neighbor locations
         of the box (according to the static adjacency graph) and take the robot
         distance to the nearest such neighbor. The minimal such distance across
         all boxes is taken as the (optimistic) robot movement cost required to
         begin pushing. (We ignore subsequent robot repositioning costs for
         efficiency and simplicity.)

    5. Combine costs:
       - The heuristic value is the sum of:
           total_pushes = sum(push_distances for all boxes)
           + robot_start_cost = min_distance_robot_to_some_box_neighbor
       - If any required distance is unavailable (disconnected graph), return a
         large finite penalty cost to indicate difficulty.

    Notes
    -----
    - The heuristic returns an integer >= 0 and returns 0 exactly on goal states.
    - The heuristic is intentionally optimistic about box-push feasibility (it
      ignores the blocking effect of other boxes and temporary clear constraints),
      which makes it fast to compute and still informative for guiding search.
    """

    def __init__(self, task):
        # Extract goal locations: expect goal facts like "(at box1 loc_2_4)"
        self.goal_locations = {}
        for g in task.goals:
            parts = get_parts(g)
            if parts and parts[0] == "at" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts: loc -> list of neighbor locs
        self.adj = {}
        self.locations = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) >= 3:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.adj.setdefault(l1, []).append(l2)
                # Note: We do not automatically add the reverse edge here because
                # the static file typically contains both directions. If it does not,
                # BFS will still operate on the directed edges that exist.

        # Ensure all locations have an entry in adjacency dict
        for loc in self.locations:
            self.adj.setdefault(loc, [])

        # Precompute shortest-path distances between all pairs of locations
        # using BFS from each location. distances[src][dst] = number of steps,
        # or None if unreachable.
        self.distances = {loc: {} for loc in self.locations}
        for src in self.locations:
            dist = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.adj.get(u, []):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            # store distances; unreachable locations absent from dict
            self.distances[src] = dist

    def __call__(self, node):
        state = node.state

        # 1) If already goal, return 0
        # task.goal_reached not available here; check membership of all goal facts
        if all(goal in state for goal in (f"(at {b} {l})" for b, l in self.goal_locations.items())):
            return 0

        # 2) Extract robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) >= 2:
                robot_loc = parts[1]
                break

        # If robot location unknown, be conservative and set a penalty
        if robot_loc is None:
            return 99999

        # 3) Extract box current locations
        box_locs = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # record only if this object has a goal (a box we care about)
                if obj in self.goal_locations:
                    box_locs[obj] = loc

        # If a goal-specified box is missing from the state (should not happen), penalize
        for box in self.goal_locations:
            if box not in box_locs:
                return 99999

        # 4) Compute total push distances (relaxed): for each box, shortest-path distance from box loc to its goal loc
        total_pushes = 0
        for box, goal_loc in self.goal_locations.items():
            cur_loc = box_locs[box]
            # If either location is unknown in the static map -> unreachable
            if cur_loc not in self.distances or goal_loc not in self.distances:
                return 99999
            dist_dict = self.distances[cur_loc]
            if goal_loc not in dist_dict:
                # unreachable
                return 99999
            total_pushes += dist_dict[goal_loc]

        # 5) Estimate robot moves needed to start pushing (distance to some neighbor of any box)
        # For each box, consider all neighbors of the box as candidate robot positions.
        min_robot_to_box_adj = None
        for box, cur_loc in box_locs.items():
            # neighbors of the box
            neighbors = self.adj.get(cur_loc, [])
            if not neighbors:
                # No way to stand adjacent to this box according to static adjacency -> unreachable
                return 99999
            for nb in neighbors:
                # Distance from robot_loc to neighbor
                if robot_loc not in self.distances:
                    return 99999
                dist_to_nb = self.distances[robot_loc].get(nb, None)
                if dist_to_nb is None:
                    # neighbor unreachable from robot; skip
                    continue
                if (min_robot_to_box_adj is None) or (dist_to_nb < min_robot_to_box_adj):
                    min_robot_to_box_adj = dist_to_nb

        if min_robot_to_box_adj is None:
            # robot cannot reach any neighbor of any box
            return 99999

        # 6) Combine estimates: pushes + robot initial travel.
        # Each push is one action; robot moves also count as actions.
        heuristic_value = total_pushes + min_robot_to_box_adj

        # Return as integer cost
        return int(heuristic_value)