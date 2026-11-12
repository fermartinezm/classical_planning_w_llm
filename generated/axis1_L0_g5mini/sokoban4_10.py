from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact string by stripping the surrounding
    parentheses and splitting on whitespace.

    Example:
      "(adjacent loc_1 loc_2 right)" -> ["adjacent", "loc_1", "loc_2", "right"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the Sokoban domain. The heuristic estimates
    the remaining number of actions by combining:
      - An estimate of the number of pushes required to bring each box from
        its current location to its goal location (computed as the shortest
        directed path length on the static adjacency graph).
      - A small estimate of the robot's travel cost to reach the first push
        position (computed as the shortest directed distance from the robot's
        current location to any cell from which it can perform the first push).

    The heuristic is not required to be admissible; it is designed to be
    informative and fast to compute in order to guide greedy best-first search
    and reduce node expansions.

    Assumptions
    -----------
    - The static "adjacent" facts define directed edges between location nodes.
      A box can be pushed from loc A to loc B iff (adjacent A B <dir>) exists.
    - Goals are of the form "(at boxN loc_X)". The heuristic uses the goal set
      provided in task.goals.
    - The robot and boxes are represented in the state using:
        "(at-robot loc_X)" and "(at boxN loc_Y)".
    - Robot movement costs and push costs are both counted as 1 per action.
    - To keep computation fast, the robot distance is computed on the static
      adjacency graph and ignores dynamic blocking by boxes. This gives a
      heuristic that may be optimistic about robot travel but is cheap.

    Heuristic Initialization
    ------------------------
    Constructor extracts and stores:
      - self.goals: the set of goal facts (task.goals).
      - self.goal_locations: mapping box_name -> goal_location parsed from goals.
      - self.adjacency: a directed adjacency dictionary mapping location -> list
        of successor locations (from static facts).
      - self.locations: set of all locations seen in adjacency facts.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a state (set of fact strings):
    1. Goal check:
       - If all goal facts are already true in the state, return 0.

    2. Parse the state:
       - Extract robot location from the "(at-robot ...)" fact.
       - Extract current locations for boxes (facts of form "(at boxN loc)").

    3. For each box that is not yet at its goal:
       - Compute the minimal number of pushes required to move the box from its
         current location to its goal location. This is the length (in edges)
         of the shortest directed path from current_loc to goal_loc on the
         adjacency graph. Each edge corresponds to one push of that box.
       - If a box's goal is unreachable on the static graph, return a large
         penalty (finite but large) to deprioritize this state.

    4. Robot travel estimate:
       - For each box (not yet at goal), compute the set of predecessor
         locations "rloc" from which a push could move the box one step
         (i.e., all rloc such that adjacent rloc -> box_loc exists).
       - Compute shortest directed distances from the robot's current location
         to all rlocs (BFS on the static adjacency graph). The minimal
         distance among all boxes' predecessor locations is taken as the
         robot travel estimate (it represents the moves needed before the
         first push). If there are no pushes required (all boxes at goal),
         this is zero.

    5. Combine estimates:
       - heuristic_value = total_pushes_required + robot_travel_estimate
       - Both push and move costs are counted as 1 per action. The heuristic
         deliberately ignores some subtleties (robot movements between pushes,
         dynamic blocking) to remain fast.

    Reasoning about properties:
    - The heuristic returns 0 exactly when all goal facts are true (checked
      directly).
    - If a box's goal is unreachable in the static adjacency graph the
      heuristic returns a large finite number, avoiding infinity while making
      such states low priority.
    - Computation is efficient: BFS is used on the static adjacency graph;
      typical Sokoban boards are small enough for this to be fast.
    """

    def __init__(self, task):
        """
        Extract static adjacency and goal information from the task.

        Parameters
        ----------
        task : planning Task
            The planning task object provided by the planner framework.
        """
        # The set of goal facts that must hold in goal states.
        self.goals = task.goals

        # Parse goal facts for per-box goal locations: "(at box1 loc_2_4)"
        self.goal_locations = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "at" and len(parts) >= 3:
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Build directed adjacency graph from static facts.
        self.adjacency = {}
        self.locations = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.locations.add(a)
                self.locations.add(b)
                self.adjacency.setdefault(a, []).append(b)

        # Ensure every known location appears in adjacency dict (even with empty list)
        for loc in list(self.locations):
            self.adjacency.setdefault(loc, [])

        # A reasonably large penalty used if some goal is unreachable.
        # Chosen finite so heuristic always returns a numeric value.
        self.unreachable_penalty = 1000

    def _bfs_distance(self, start, targets):
        """
        BFS on directed adjacency graph from start to nearest target.
        Returns the distance (#edges) to the closest node in 'targets' (set),
        or None if unreachable.
        """
        if start in targets:
            return 0
        q = deque()
        q.append((start, 0))
        seen = {start}
        while q:
            node, dist = q.popleft()
            for succ in self.adjacency.get(node, []):
                if succ in seen:
                    continue
                if succ in targets:
                    return dist + 1
                seen.add(succ)
                q.append((succ, dist + 1))
        return None

    def _bfs_distance_single(self, start, goal):
        """
        BFS on directed adjacency graph from start to goal.
        Returns distance (#edges) or None if unreachable.
        """
        if start == goal:
            return 0
        q = deque()
        q.append((start, 0))
        seen = {start}
        while q:
            node, dist = q.popleft()
            for succ in self.adjacency.get(node, []):
                if succ in seen:
                    continue
                if succ == goal:
                    return dist + 1
                seen.add(succ)
                q.append((succ, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given search node.

        Parameters
        ----------
        node : search Node
            The node provided by the planner; node.state is a frozenset of fact strings.

        Returns
        -------
        int
            Estimated number of actions (pushes + a small robot travel estimate)
            to reach a goal state from node.state.
        """
        state = node.state

        # If all goal facts are satisfied, return 0.
        if self.goals <= state:
            return 0

        # Extract robot location.
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) >= 2:
                robot_loc = parts[1]
                break

        # Extract current locations of boxes (facts of the form "(at boxX locY)").
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                # We only care about boxes that appear in the goal set (goal boxes).
                if obj in self.goal_locations:
                    box_locations[obj] = loc

        # If no boxes are known (should not happen in normal instances), fallback.
        if not box_locations:
            # No boxes to move -> return 0 if goals met else small penalty.
            return 0 if self.goals <= state else 1

        total_pushes = 0
        # For robot travel estimate, collect all predecessor locations (locations
        # from which a push into the current box location is possible).
        predecessor_positions = set()

        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)
            # If the box is missing in the current state, treat as unreachable.
            if current_loc is None:
                return self.unreachable_penalty

            # If box already at its goal, no pushes required for this box.
            if current_loc == goal_loc:
                continue

            # Compute minimal number of pushes (edges) needed to move the box to its goal.
            dist = self._bfs_distance_single(current_loc, goal_loc)
            if dist is None:
                # Unreachable goal for this box in the static graph -> large penalty.
                return self.unreachable_penalty
            total_pushes += dist

            # Collect predecessor positions rloc such that adjacent rloc -> current_loc.
            # Those are positions from which the robot can push the box for the first move.
            # We scan adjacency to find such rlocs.
            for loc, succs in self.adjacency.items():
                if current_loc in succs:
                    predecessor_positions.add(loc)

        # If all boxes are already at goals (no pushes required), return 0.
        if total_pushes == 0:
            return 0

        # Compute robot travel estimate: minimal directed distance from robot_loc to any predecessor position.
        robot_travel = 0
        if robot_loc is None:
            # If robot location unknown, give a small penalty.
            robot_travel = 1
        else:
            if not predecessor_positions:
                robot_travel = 0
            else:
                d = self._bfs_distance(robot_loc, predecessor_positions)
                if d is None:
                    # Robot cannot reach any predecessor position in static graph -> large penalty.
                    return self.unreachable_penalty
                robot_travel = d

        # Final heuristic: sum of required pushes plus initial robot travel to the first push.
        return total_pushes + robot_travel