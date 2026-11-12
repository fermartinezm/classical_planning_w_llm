from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the 'sokoban' (secret_domain) domain.
    The goal in this domain is to make predicate (adjacent-2 BOX) true for
    all goal boxes. Achieving (adjacent-2 BOX) requires the sequence of
    actions that (informally) moves the robot to a location that allows
    a 'move' action to make (at BOX) true and then applies 'push' to
    produce (adjacent-2 BOX). The robot can move between locations using
    directed edges encoded by (clear LOC1 LOC2) static facts (via move-2 /
    push-2 operators). Static facts also include (at-robot BOX LOC_ROBOT),
    which specifies the robot-relative location required to push that box.

    The heuristic estimates the number of actions needed to satisfy all
    (adjacent-2 BOX) goal facts by (a) counting necessary move-2 steps to
    travel the robot between locations, (b) counting one 'move' action to
    set (at BOX) when required, and (c) counting one 'push' action to
    establish (adjacent-2 BOX). The heuristic simulates a greedy sequence
    of fixing goal boxes one-by-one, always choosing the currently
    available adjacent location for a box (from the current state) that
    minimizes travel cost from the robot's current location.

    Assumptions
    -----------
    - Every goal is of the form (adjacent-2 BOX).
    - Static facts contain mappings (at-robot BOX LOC) that specify the
      robot location required to perform the push for BOX. If such a
      mapping is missing for a box, the heuristic assumes the push can be
      performed at the same location where a 'move' would be performed.
    - Connectivity between locations is encoded by static facts
      (clear LOC1 LOC2). These are treated as directed edges (LOC1 -> LOC2)
      because the move-2 operator requires (clear ?from ?to).
    - If there is no path between two locations (according to static clear
      facts), the heuristic uses a large finite penalty to keep values finite.
    - The heuristic is not admissible; it is tuned to guide greedy best-first search.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
    - Stores the set of goal facts for quick goal checks.
    - Builds a directed graph of locations from static (clear ...) facts.
    - Builds a mapping from each box to its required robot location for pushing
      from static (at-robot BOX LOC) facts.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state:
    1. If all goal facts are already true in the state, return 0.
    2. Parse the state to find:
       - robot_loc: the current robot location from (at-robot-2 LOC).
       - boxes that already satisfy (adjacent-2 BOX).
       - for each box, the set of locations L for which (adjacent BOX L) is
         currently true (these are the candidate locations where a 'move'
         action can be applied to produce (at BOX)).
       - which boxes already have (at BOX) true (so 'move' is not required).
    3. For each goal box not yet in adjacent-2:
       - Identify at_robot_loc for that box from static (at-robot BOX LOC).
         If not available, fall back to using a candidate move location or
         the robot's current location.
       - If (at BOX) is already true in the state, the only required action is
         to get the robot to at_robot_loc (if not already there) and do one
         'push' action.
       - Otherwise, the canonical sequence considered is:
         a) Travel from current robot location to a chosen move_loc (one of
            the (adjacent BOX move_loc) facts in the state).
         b) Apply 'move' (cost 1) to set (at BOX).
         c) Travel from move_loc to at_robot_loc (if different).
         d) Apply 'push' (cost 1) to set (adjacent-2 BOX).
         The chosen move_loc is the one that minimizes travel distances for the
         current robot location. Travel distances are computed with BFS on the
         directed graph built from (clear LOC1 LOC2).
    4. After "fixing" a box the heuristic simulates the robot ending up at the
       at_robot_loc for that box (because push requires that location). The
       robot's simulated location is then used as the start point for the next
       remaining box. This models a sequential strategy to make the heuristic
       informative about travel overhead between boxes.
    5. If BFS reports no path between two required locations, a large finite
       penalty (PENALTY) is used for that travel leg to keep the heuristic
       finite while still discouraging unreachable routes.
    """

    def __init__(self, task):
        # Goals and static information
        self.goals = task.goals  # set of goal fact strings
        static_facts = task.static  # frozenset of static fact strings

        # Build directed graph of locations from (clear LOC1 LOC2) static facts.
        self.graph = {}  # node -> list of neighbour nodes (directed)
        self.all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "clear" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.all_locations.add(src)
                self.all_locations.add(dst)
                self.graph.setdefault(src, []).append(dst)

        # Map each box to the robot location required for push from static (at-robot BOX LOC).
        self.at_robot_loc = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at-robot" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                self.at_robot_loc[box] = loc

        # Finite penalty for unreachable travel legs (keeps heuristic finite).
        self.PENALTY = 1000

        # Precomputed BFS distance cache for speed: (src, dst) -> distance
        self._dist_cache = {}

    def _bfs_distance(self, src, dst):
        """Compute directed shortest-path distance from src to dst using BFS.
        Returns a non-negative integer or a large penalty if unreachable.
        Results are cached for repeated queries."""
        if src == dst:
            return 0
        key = (src, dst)
        if key in self._dist_cache:
            return self._dist_cache[key]

        # If src or dst are unknown, return penalty (keeps heuristic finite).
        if src not in self.graph and src not in self.all_locations:
            self._dist_cache[key] = self.PENALTY
            return self.PENALTY
        if dst not in self.graph and dst not in self.all_locations:
            self._dist_cache[key] = self.PENALTY
            return self.PENALTY

        visited = {src}
        q = deque([(src, 0)])
        while q:
            node, d = q.popleft()
            for nei in self.graph.get(node, []):
                if nei == dst:
                    self._dist_cache[key] = d + 1
                    return d + 1
                if nei not in visited:
                    visited.add(nei)
                    q.append((nei, d + 1))

        # If not reachable via directed edges, return penalty
        self._dist_cache[key] = self.PENALTY
        return self.PENALTY

    def __call__(self, node):
        """
        Compute the heuristic estimate (an integer) for node.state.
        See docstring for the step-by-step computation.
        """
        state = node.state

        # If all goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse current robot location from state: (at-robot-2 LOC)
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot-2" and len(parts) >= 2:
                robot_loc = parts[1]
                break
        # If robot location unknown (should not happen in valid states), assume arbitrary known location
        if robot_loc is None and self.all_locations:
            robot_loc = next(iter(self.all_locations))
        elif robot_loc is None:
            # No locations known at all: return a conservative finite estimate (number of remaining goals * 2)
            remaining = sum(1 for g in self.goals if g not in state)
            return remaining * 2

        # Build per-box current information from the state.
        # adjacent_map: box -> list of locations L where (adjacent box L) is true in current state
        adjacent_map = {}
        at_set = set()  # boxes for which (at BOX) is true
        adjacent2_set = set()  # boxes for which (adjacent-2 BOX) is already true

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "adjacent" and len(parts) >= 3:
                box, loc = parts[1], parts[2]
                adjacent_map.setdefault(box, []).append(loc)
            elif pred == "at" and len(parts) >= 2:
                box = parts[1]
                at_set.add(box)
            elif pred == "adjacent-2" and len(parts) >= 2:
                box = parts[1]
                adjacent2_set.add(box)

        total_cost = 0
        # Extract boxes from goal facts: assume goals are like "(adjacent-2 boxX)"
        goal_boxes = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "adjacent-2" and len(parts) >= 2:
                goal_boxes.append(parts[1])

        # Process boxes greedily, simulating the robot ending up at at_robot_loc after each push.
        current_robot_loc = robot_loc
        for box in goal_boxes:
            # Skip boxes already in adjacent-2 in current state
            if box in adjacent2_set:
                continue

            # Determine the at_robot location required for push (static)
            at_robot_loc = self.at_robot_loc.get(box, None)

            # Candidate move locations (where a 'move' action can be applied) are from adjacent_map.
            candidate_move_locs = adjacent_map.get(box, [])

            # If the box already satisfies (at BOX), we can skip the 'move' action.
            if box in at_set:
                # Need to travel to at_robot_loc (if known) and do one push action.
                if at_robot_loc is None:
                    # No static mapping; assume push can be done where the box is (use current robot location)
                    travel_to_push = 0 if current_robot_loc == current_robot_loc else 0
                    push_cost = 1
                    step_cost = travel_to_push + push_cost
                else:
                    travel_to_push = self._bfs_distance(current_robot_loc, at_robot_loc)
                    push_cost = 1
                    step_cost = travel_to_push + push_cost

                # Add step cost and update robot location after push (robot stays at at_robot_loc)
                total_cost += step_cost
                if at_robot_loc is not None:
                    current_robot_loc = at_robot_loc
                # If at_robot_loc is unknown, leave robot where it was.
                continue

            # Box is not at robot; a 'move' is needed followed by a 'push'.
            # If no candidate move locations are present, fall back to at_robot_loc or current location.
            if not candidate_move_locs:
                # Use at_robot_loc if available as both move and push location.
                if at_robot_loc:
                    candidate_move_locs = [at_robot_loc]
                else:
                    # Use current robot location as trivial fallback
                    candidate_move_locs = [current_robot_loc]

            # For each candidate move location, compute total cost:
            # travel(current_robot_loc -> move_loc) + 1 (move) + travel(move_loc -> at_robot_loc) + 1 (push)
            best = None
            for move_loc in candidate_move_locs:
                # Travel to move_loc
                d1 = self._bfs_distance(current_robot_loc, move_loc)
                # If at_robot_loc exists, travel from move_loc to at_robot_loc; else assume zero
                if at_robot_loc is not None:
                    d2 = self._bfs_distance(move_loc, at_robot_loc)
                else:
                    # If we don't know the required push location, assume push can be done at move_loc
                    d2 = 0
                    at_robot_loc = at_robot_loc or move_loc

                cost = d1 + 1 + d2 + 1  # travel + move + travel + push
                if best is None or cost < best[0]:
                    best = (cost, move_loc, at_robot_loc)

            if best is None:
                # Defensive fallback
                total_cost += self.PENALTY
                # leave robot location unchanged
            else:
                step_cost, chosen_move_loc, chosen_at_robot_loc = best
                total_cost += step_cost
                # After push, robot is at the at_robot_loc for this box (if known).
                if chosen_at_robot_loc is not None:
                    current_robot_loc = chosen_at_robot_loc

        # Return integer heuristic estimate
        return int(total_cost)