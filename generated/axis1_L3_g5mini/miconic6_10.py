from collections import deque
from heuristics.heuristic_base import Heuristic
import math


def get_parts(fact):
    """Remove surrounding parentheses and split into components."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the `sokoban`-like secret_domain. The domain's
    goal is to make predicates (adjacent-2 boxX) true for a set of boxes. Achieving
    adjacent-2(box) is done by a push action that requires the robot to be at a
    specific location (an `at-robot box location` fact) and the box to be in the
    `at box` state. The `at box` state can be obtained with a move action when the
    robot is positioned at a location L for which `(adjacent box L)` is true.
    The robot's current position is represented by a single fact `(at-robot-2 loc)`.
    Robot navigation between locations is governed by directed `clear` edges
    (static facts of the form `(clear loc1 loc2)`) and is achieved by `move-2`
    / `push-2` operators, each costing one action per edge traversal.

    Assumptions
    -----------
    - The cost (number of actions) for:
        * moving the robot along one `clear` edge is 1 (one move-2 or push-2).
        * the `move` operator that creates `(at box)` is 1.
        * the `push` operator that creates `(adjacent-2 box)` is 1.
    - The `at-robot box loc` facts that indicate valid push locations for a box
      may appear either in the static facts or in the current state; we consider
      both sources.
    - The `adjacent box loc` facts specifying locations from which a `move`
      can pick up a box are part of the current state.
    - If a necessary location (push or adjacency) is unreachable using `clear`
      edges from the robot's current position, the box is treated as (very)
      expensive to achieve but the heuristic remains finite.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal boxes from the task goals (those that must have
      `(adjacent-2 box)` true).
    - Build a directed graph of robot-navigation edges from static `clear`
      facts (edges loc -> loc2 whenever `(clear loc loc2)` is in static).
    - Collect `at-robot` facts from static (these denote where the robot can be
      to perform a `push` for a given box). (During evaluation we also consider
      any `at-robot` facts present in the current state.)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node (state), the heuristic estimates the number of
    actions to make all goal boxes satisfy `(adjacent-2 box)`:

    1. If all goal facts are already present in the state, return 0.

    2. Parse the current robot location from the unique fact `(at-robot-2 loc)`.

    3. For each goal box that is not yet `adjacent-2`:
       - Determine adjacency locations:
         * adjacency_locs = { L | "(adjacent box L)" in current state }.
         These are locations where a `move` can be applied to set `(at box)`.
       - Determine push locations:
         * push_locs = { P | "(at-robot box P)" in (static U current_state) }.
         These are locations where the robot must be positioned to apply `push`.
       - Determine whether the box already has `(at box)` in the current state.

       The minimal action sequence to make `(adjacent-2 box)` true is one of:
         A) If box already has `(at box)`:
            * Move the robot to some P in push_locs (cost = distance(robot_loc,P))
              then `push` (cost +1).
         B) Otherwise (box not `at`):
            * Move robot to some A in adjacency_locs (cost = dist(robot_loc,A)),
              apply `move` (cost +1) -> box becomes `(at box)` while robot remains
              at A, then move (via clear edges) to some P in push_locs
              (cost += dist(A,P)), then apply `push` (cost +1).
            * A special case of B is when there exists L that is both an
              adjacency location and a push location; then cost = dist(robot,L)
              + 1 (move) + 1 (push) = dist(robot,L) + 2.

       - If adjacency_locs or push_locs are empty, treat the box as expensive
         (we add a large finite penalty to keep the heuristic finite).

    4. Since multiple boxes must be achieved, we produce a sequence estimate:
       - We use a greedy sequencing approach: starting from the current robot
         location, repeatedly pick the next box with the smallest incremental
         estimated cost to complete (including robot-navigation required),
         update the robot's position to the push-location used for that box,
         and sum these incremental costs. This gives a tighter estimate than
         simply summing independent costs because it accounts for robot reuse
         between box tasks.

    5. Return the summed estimated action cost as the heuristic value.

    This heuristic is not guaranteed to be admissible (it may overestimate),
    but it is efficient to compute (BFS on the relatively small location graph)
    and captures the essential structure of the task: navigation cost plus the
    local pick-and-push actions required per box.
    """

    def __init__(self, task):
        # Goals: goal facts like "(adjacent-2 box1)". We extract the box names.
        self.goal_boxes = []
        for g in task.goals:
            parts = get_parts(g)
            if parts[0] == "adjacent-2" and len(parts) == 2:
                self.goal_boxes.append(parts[1])

        # Build directed graph for locations from static `clear` facts.
        # clear facts have the form "(clear loc1 loc2)".
        self.clear_graph = {}
        static = task.static
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "clear" and len(parts) == 3:
                a, b = parts[1], parts[2]
                self.clear_graph.setdefault(a, set()).add(b)

        # Collect push-location facts of the form "(at-robot box loc)" that
        # are static (we will also consider runtime state facts).
        # Map: box -> set(locations)
        self.static_push_locations = {}
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.static_push_locations.setdefault(box, set()).add(loc)

        # A large finite cost for unreachable situations (keep heuristic finite).
        self.UNREACHABLE_PENALTY = 1000

    def _bfs_distance(self, start, goal):
        """Compute directed shortest-path distance in the clear_graph from start to goal."""
        if start == goal:
            return 0
        if start not in self.clear_graph:
            return math.inf
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            node, dist = dq.popleft()
            for nbr in self.clear_graph.get(node, ()):
                if nbr == goal:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    dq.append((nbr, dist + 1))
        return math.inf

    def __call__(self, node):
        state = node.state

        # If all goals already satisfied -> heuristic 0
        if all(f"(adjacent-2 {box})" in state for box in self.goal_boxes):
            return 0

        # Extract current robot location from "(at-robot-2 loc)".
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot-2" and len(parts) == 2:
                robot_loc = parts[1]
                break
        if robot_loc is None:
            # No robot location in state: assign a penalty but keep finite.
            return len(self.goal_boxes) * self.UNREACHABLE_PENALTY

        # Build push-locations from static + current state for convenience:
        # push_locs(box) = set of locations where a push for box can be applied.
        push_locs_current = {}
        # Start with static known locations
        for box, locs in self.static_push_locations.items():
            push_locs_current.setdefault(box, set()).update(locs)
        # Add any at-robot facts present in the current state too.
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                push_locs_current.setdefault(box, set()).add(loc)

        # Helper to compute the minimal cost to make adjacent-2 true for a single box
        # starting from a given robot location. Also returns the robot's resulting
        # location after completion (we assume it remains at the push location P).
        def cost_to_complete_box(box, start_robot_loc):
            # If already goal-satisfied:
            if f"(adjacent-2 {box})" in state:
                return 0, start_robot_loc

            # adjacency locations from which move can pick up the box:
            adjacency_locs = {
                get_parts(fact)[1]
                for fact in state
                if get_parts(fact)[0] == "adjacent" and get_parts(fact)[1] == box and len(get_parts(fact)) == 3
            }
            # push locations (from static or state):
            push_locs = push_locs_current.get(box, set()).copy()

            # Check if box already has (at box) in current state:
            has_at = f"(at {box})" in state

            best_cost = math.inf
            best_end_loc = None

            if push_locs:
                # Case A: box already at -> just need to move to some push location and push.
                if has_at:
                    for P in push_locs:
                        d = self._bfs_distance(start_robot_loc, P)
                        if math.isinf(d):
                            continue
                        c = d + 1  # +1 for push
                        if c < best_cost:
                            best_cost = c
                            best_end_loc = P

                # Case B: need to pick up box first (use some adjacency location A)
                if adjacency_locs:
                    for A in adjacency_locs:
                        d1 = self._bfs_distance(start_robot_loc, A)
                        if math.isinf(d1):
                            continue
                        # apply move at A -> cost +1, resulting robot position is still A
                        # Then go to some push location P and push
                        for P in push_locs:
                            d2 = self._bfs_distance(A, P)
                            if math.isinf(d2):
                                continue
                            c = d1 + 1 + d2 + 1  # dist to A + move + dist A->P + push
                            if c < best_cost:
                                best_cost = c
                                best_end_loc = P

                # Special case: if there exists a location L that is both adjacency and push,
                # then cost is simply dist(start,L) + 1(move) +1(push) = dist+2.
                inter = adjacency_locs & push_locs
                for L in inter:
                    d = self._bfs_distance(start_robot_loc, L)
                    if not math.isinf(d):
                        c = d + 2
                        if c < best_cost:
                            best_cost = c
                            best_end_loc = L

            # If there were no push_locs or adjacency_locs reachable, mark as very expensive
            if math.isinf(best_cost):
                return self.UNREACHABLE_PENALTY, start_robot_loc

            return int(best_cost), best_end_loc

        # Greedy sequencing: from current robot location, repeatedly pick the next
        # unfinished box with minimal incremental cost, update robot location, sum.
        remaining = [b for b in self.goal_boxes if f"(adjacent-2 {b})" not in state]
        total_cost = 0
        cur_robot = robot_loc

        while remaining:
            best_box = None
            best_c = math.inf
            best_new_loc = None
            for b in remaining:
                c, end_loc = cost_to_complete_box(b, cur_robot)
                if c < best_c:
                    best_c = c
                    best_box = b
                    best_new_loc = end_loc
            if best_box is None:
                # Shouldn't happen, but guard against it.
                total_cost += self.UNREACHABLE_PENALTY
                break
            total_cost += best_c
            # Update robot location to the push location used (or remain same if None)
            if best_new_loc is not None:
                cur_robot = best_new_loc
            # Remove box from remaining
            remaining.remove(best_box)

        return int(total_cost)