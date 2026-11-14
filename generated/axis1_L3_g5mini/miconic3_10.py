from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Remove surrounding parentheses and split a PDDL fact into its components.
    E.g. "(adjacent box1 location2)" -> ["adjacent", "box1", "location2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "sokoban"-like secret_domain. The goal
    predicates in this domain are of the form (adjacent-2 <box>) for boxes.
    Achieving (adjacent-2 <box>) requires performing domain actions that
    typically consist of: bringing the robot to some location, applying a
    'move' action to turn (adjacent <box> <loc>) into (at <box>), and then a
    'push' action to turn (at <box>) into (adjacent-2 <box>). Robot movement
    between locations is governed by static (clear <loc1> <loc2>) facts and
    uses (at-robot-2 <loc>) facts in the state.

    Assumptions
    -----------
    - Goals are only (adjacent-2 <box>) facts (the heuristic reads task.goals).
    - Static facts include:
        * (at-robot <box> <loc>)  -> the location where the robot must stand to push that box.
        * (clear <loc1> <loc2>)   -> a directed edge enabling robot movement from loc1 to loc2.
      These are extracted in the constructor.
    - The state contains at most one (at-robot-2 <loc>) fact describing the robot's current location.
    - Every box that can reach the goal is initially either in an 'adjacent' state
      (has an (adjacent <box> <loc>) fact) or an 'at' state ((at <box>)).
      If a box is in neither state (and not already adjacent-2), this implementation
      treats it as effectively unreachable and assigns a large finite penalty.
    - The heuristic is NOT required to be admissible; it aims to be informative and
      fast to compute.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts:
    - self.goals: the set of goal facts (task.goals).
    - self.at_robot_map: a dictionary mapping each box -> location from static facts
      of the form "(at-robot box location)". This is the location where the robot
      must stand to execute the push for that box.
    - self.clear_graph: a directed adjacency dictionary mapping each location ->
      set(neighbor_locations) based on static "(clear loc1 loc2)" facts. This
      graph is used to compute shortest robot move distances (BFS).
    - A constant UNREACHABLE_COST used as a large finite penalty for states
      where a required fact cannot be produced given available static info.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goals are already satisfied (self.goals <= state) return 0.
    2. Determine the robot's current location by finding the single
       "(at-robot-2 <loc>)" fact in the state. If not present, assume robot
       location unknown and use a conservative large penalty when movement is needed.
    3. For each goal box (a box that must satisfy (adjacent-2 <box>)):
       - If the box already has (adjacent-2 <box>) in the state, contribution = 0.
       - Otherwise, inspect the state's facts for the box:
         a) If the box has (at <box>) in the state:
            - We only need one push action, provided the robot stands at the
              required push-location (from static (at-robot <box> <loc>)).
            - Cost_i = distance(robot_loc, push_loc) + 1 (for the push).
         b) If the box has (adjacent <box> <loc_move>) in the state:
            - We must first do a move action while the robot stands at loc_move:
              that yields (at <box>). Then we must go to the push location and push.
            - Cost_i = distance(robot_loc, loc_move) + 1 (move)
                       + distance(loc_move, push_loc) + 1 (push).
         c) If the box is in neither (at) nor (adjacent) and not already
            (adjacent-2), it is treated as unreachable in this straightforward
            model; assign a large finite UNREACHABLE_COST to that box.
       - Distances between locations are computed as shortest-path lengths in the
         directed clear graph using BFS. If a location is unreachable from the
         robot's current location (or between needed locations), use UNREACHABLE_COST.
    4. Sum the per-box costs to get the heuristic estimate. The robot location is
       not advanced between boxes, i.e., the heuristic sums independent minimal
       costs per box (this is admissibility-violating but fast and usually
       informative for greedy best-first search).
    """

    def __init__(self, task):
        # Goals are provided as strings of facts.
        self.goals = task.goals
        static_facts = task.static

        # Map from box -> location where robot must stand to push that box.
        self.at_robot_map = {}
        # Directed adjacency graph for robot movement between locations.
        self.clear_graph = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == "at-robot" and len(parts) >= 3:
                # (at-robot <box> <loc>)
                box = parts[1]
                loc = parts[2]
                self.at_robot_map[box] = loc
            elif predicate == "clear" and len(parts) >= 3:
                # (clear <loc1> <loc2>) : directed edge loc1 -> loc2
                loc1 = parts[1]
                loc2 = parts[2]
                self.clear_graph.setdefault(loc1, set()).add(loc2)

        # Ensure nodes with no outgoing edges still appear in graph (for BFS).
        # We do not need to enumerate all locations here; BFS will handle missing nodes.
        self.UNREACHABLE_COST = 1000

    def _bfs_distance(self, start, goal):
        """Return shortest directed distance from start to goal in clear_graph (integer).
           If start == goal return 0. If unreachable return UNREACHABLE_COST.
        """
        if start is None or goal is None:
            return self.UNREACHABLE_COST
        if start == goal:
            return 0
        # Quick check: if start not in graph, it might still be equal handled above;
        # but there may be outgoing edges from other nodes only. We'll still attempt BFS.
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            neighbors = self.clear_graph.get(node, ())
            for nb in neighbors:
                if nb == goal:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        return self.UNREACHABLE_COST

    def __call__(self, node):
        """
        Compute an estimate (integer) of the number of actions required to reach a goal state.
        """
        state = node.state

        # If already a goal state, heuristic is zero.
        if self.goals <= state:
            return 0

        # Extract robot current location from state: (at-robot-2 <loc>).
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot-2" and len(parts) >= 2:
                robot_loc = parts[1]
                break

        total_cost = 0

        # For each goal fact (we expect goals of form (adjacent-2 <box>))
        for goal_fact in self.goals:
            gparts = get_parts(goal_fact)
            if not gparts:
                continue
            if gparts[0] != "adjacent-2" or len(gparts) < 2:
                # The heuristic focuses on adjacent-2 goals; for other goals,
                # conservatively add a small penalty (should not happen in this domain).
                total_cost += 1
                continue
            box = gparts[1]

            # If box already satisfies adjacent-2 in the current state -> cost 0.
            if f"(adjacent-2 {box})" in state:
                continue

            # Find whether box is currently 'at' or 'adjacent <box> <loc>'
            box_at = f"(at {box})" in state

            adjacent_loc = None
            # Find an (adjacent box loc) fact for this box (if any).
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "adjacent" and len(parts) >= 3 and parts[1] == box:
                    adjacent_loc = parts[2]
                    break

            # Determine where robot must stand to push this box (from static).
            push_loc = self.at_robot_map.get(box)

            # Compute cost according to the box's current status.
            if box_at:
                # Need a single push (1) plus robot movement to push_loc.
                if push_loc is None:
                    # Missing static mapping -> cannot determine push location reliably.
                    total_cost += self.UNREACHABLE_COST
                    continue
                d = self._bfs_distance(robot_loc, push_loc)
                total_cost += d + 1  # d moves + 1 push
            elif adjacent_loc is not None:
                # Need to move while robot stands at adjacent_loc (1),
                # then move from adjacent_loc to push_loc, then push (1).
                if push_loc is None:
                    total_cost += self.UNREACHABLE_COST
                    continue
                d1 = self._bfs_distance(robot_loc, adjacent_loc)
                d2 = self._bfs_distance(adjacent_loc, push_loc)
                total_cost += d1 + 1 + d2 + 1  # travel to adjacent_loc + move + travel + push
            else:
                # Box is neither 'at' nor 'adjacent' nor already adjacent-2:
                # in the simple model there is no operator to create 'adjacent' from scratch,
                # so treat as effectively unreachable with a large finite penalty.
                total_cost += self.UNREACHABLE_COST

        return total_cost