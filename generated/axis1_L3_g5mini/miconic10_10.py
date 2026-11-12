from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the space-separated parts of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic tailored for the "sokoban"-like secret_domain. The planner's goals
    require making (adjacent-2 boxX) true for certain boxes. This heuristic
    estimates the number of actions needed to achieve all such goals by
    reasoning about:
      - robot movement between locations (move-2 / push-2),
      - the move action that makes (at box) true (used before pushing),
      - the push action that makes (adjacent-2 box) true.

    Assumptions
    -----------
    - The static facts include mappings of the form "(at-robot box location)"
      telling at which robot location a push for that box must occur (we call
      this the "push location" for the box).
    - Directed "clear" facts "(clear from to)" indicate possible robot moves
      (an edge from `from` to `to`) and are used to compute robot movement
      costs (each step costs 1).
    - The "move" action that creates "(at box)" requires the robot to be at a
      location L such that "(adjacent box L)" holds in the current state; the
      move action costs 1 and does not change the robot location.
    - The "push" action requires "(at box)" and the robot to be at the box's
      push-location (from static facts) and costs 1. Its effect is to make
      "(adjacent-2 box)" true (and remove "(at box)").
    - We ignore interactions between boxes except via robot movement; i.e.,
      we plan a sequence of box-goals greedily (pick next box with lowest
      incremental cost), updating the robot location after each simulated push.
      This gives a more informed (but still fast) estimate than summing
      independent single-box costs.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
      - self.goals: the set of goal facts.
      - self.push_location: mapping box -> location (from static "(at-robot box loc)").
      - self.clear_graph: adjacency list of directed location -> set(locations) from
        static "(clear src dest)" facts.
      - self.all_locations: set of location names detected in static facts.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the node.state already satisfies all goals, return 0.
    2. Identify the robot's current location using the fact "(at-robot-2 <loc>)" in state.
       If missing, we assume robot is nowhere and use a fallback that still returns
       a finite estimate.
    3. For each goal box not yet satisfied (i.e., "(adjacent-2 box)" not in state),
       we compute the minimal incremental cost to achieve that single box goal when
       starting from the current robot location. The cost model:
         - If "(at box)" is already true in the state:
             cost_box = distance(robot_loc -> push_loc) + 1  # for the push action
         - Else:
             Let candidate adjacency locations be all L with "(adjacent box L)" in state.
             For each candidate L:
               cost_candidate = dist(robot_loc -> L) + 1      # move that makes (at box)
                                + dist(L -> push_loc) + 1    # move robot to push_loc + push
             cost_box = min(cost_candidate over all L)
       If a required static mapping or adjacency info is missing or there is no path,
       that candidate is penalized with a large finite cost.
    4. The heuristic simulates performing the cheapest next box achievement greedily:
       - Pick the unsolved box with minimum computed cost_box.
       - Add cost_box to total heuristic cost, update robot_loc to the push_loc of that box
         (because after a push the robot is at the push location), remove box from
         unsolved list, and repeat until all boxes are accounted for.
    5. Return the accumulated cost as the heuristic value.

    This heuristic is not admissible (it is intentionally optimistic/greedy), but it
    aims to produce informative estimates while remaining efficiently computable.
    """

    def __init__(self, task):
        # Goals (set of strings)
        self.goals = task.goals
        static_facts = task.static

        # Map boxes to their push-locations via static "(at-robot box location)" facts.
        self.push_location = {}
        # Directed graph of locations from "(clear src dest)" facts: src -> {dest,...}
        self.clear_graph = {}
        self.all_locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == "at-robot" and len(parts) >= 3:
                box = parts[1]
                loc = parts[2]
                self.push_location[box] = loc
                self.all_locations.add(loc)
            elif predicate == "clear" and len(parts) >= 3:
                src = parts[1]
                dest = parts[2]
                self.clear_graph.setdefault(src, set()).add(dest)
                self.all_locations.add(src)
                self.all_locations.add(dest)

    def _bfs_distances(self, source):
        """
        Compute shortest distances (number of move-steps) from source to all reachable locations
        using the directed edges in self.clear_graph. Locations not in the graph are assumed
        to have no outgoing edges.
        Returns a dict location->distance (int). If source is None or unknown, returns empty dict.
        """
        if source is None:
            return {}
        dist = {}
        q = deque()
        dist[source] = 0
        q.append(source)
        while q:
            u = q.popleft()
            for v in self.clear_graph.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def __call__(self, node):
        state = node.state

        # If goal already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Determine robot current location from state: "(at-robot-2 <loc>)"
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot-2" and len(parts) >= 2:
                robot_loc = parts[1]
                break

        # Extract which boxes must reach adjacent-2 according to goals
        goal_boxes = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "adjacent-2" and len(parts) >= 2:
                goal_boxes.append(parts[1])

        # Identify which goal boxes are already satisfied in the current state
        unsolved = [b for b in goal_boxes if f"(adjacent-2 {b})" not in state]
        if not unsolved:
            return 0

        # Precompute BFS distances from current robot location (will update as robot moves)
        INF = 10 ** 6  # large finite penalty

        total_cost = 0
        current_robot_loc = robot_loc

        # We will greedily solve the unsolved boxes in an order that minimizes incremental cost.
        # To avoid repeated full graph searches for each candidate box, we compute BFS distances
        # from whatever the current_robot_loc is at each iteration.
        while unsolved:
            distances = self._bfs_distances(current_robot_loc) if current_robot_loc is not None else {}
            best_box = None
            best_cost = None
            best_push_loc = None
            # For each unsolved box compute minimal incremental cost
            for box in unsolved:
                # If the box already has (adjacent-2 box) in state (re-check), skip
                if f"(adjacent-2 {box})" in state:
                    candidate_cost = 0
                    candidate_push_loc = current_robot_loc
                else:
                    # Required push location (static)
                    push_loc = self.push_location.get(box)
                    if push_loc is None:
                        # Missing static mapping -> penalize heavily but finite
                        candidate_cost = INF
                        candidate_push_loc = current_robot_loc
                    else:
                        # If box already has "(at box)" in state, then only need to get to push_loc and push
                        if f"(at {box})" in state:
                            d_to_push = distances.get(push_loc, INF)
                            candidate_cost = d_to_push + 1  # push cost
                            candidate_push_loc = push_loc
                        else:
                            # Need to perform a "move" that makes (at box) true.
                            # The move requires the robot to be at some location L such that "(adjacent box L)" is in state.
                            # Find all such L in the current state.
                            candidate_cost = INF
                            candidate_push_loc = current_robot_loc
                            # scan state for adjacency facts for this box
                            found_any_adj = False
                            for fact in state:
                                parts = get_parts(fact)
                                if parts and parts[0] == "adjacent" and len(parts) >= 3:
                                    # predicate: adjacent box location (domain uses (adjacent ?v1 - box ?v2 - location))
                                    b = parts[1]
                                    loc_L = parts[2]
                                    if b != box:
                                        continue
                                    found_any_adj = True
                                    # Cost to go to L from current robot loc
                                    d_to_L = distances.get(loc_L, INF)
                                    # After doing the move (cost 1), robot remains at L
                                    # Then robot must travel from L to push_loc (cost = distance from L to push_loc)
                                    # We need distances from L to push_loc: compute BFS from L (cheap if graph small)
                                    dist_from_L = self._bfs_distances(loc_L)
                                    d_L_to_push = dist_from_L.get(push_loc, INF)
                                    cost_candidate = d_to_L + 1 + d_L_to_push + 1  # move + travel + push
                                    if cost_candidate < candidate_cost:
                                        candidate_cost = cost_candidate
                                        candidate_push_loc = push_loc
                            if not found_any_adj:
                                # No adjacency info available in state for this box, penalize
                                candidate_cost = INF
                                candidate_push_loc = current_robot_loc

                # select best box (minimum incremental cost)
                if best_cost is None or candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_box = box
                    best_push_loc = candidate_push_loc

            # If best_cost is INF (unreachable), return a large but finite heuristic:
            if best_cost is None or best_cost >= INF:
                # Heuristic cannot find reachable actions for remaining boxes; return a finite
                # but large estimate proportional to number of unsolved boxes.
                return INF + len(unsolved) * 10

            # "Apply" the best box optimistic plan: add cost and update robot location to push location
            total_cost += best_cost
            current_robot_loc = best_push_loc
            # Mark box solved and continue
            unsolved.remove(best_box)

        # Return the accumulated optimistic cost estimate
        return int(total_cost)