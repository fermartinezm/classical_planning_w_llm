from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing surrounding parentheses
    and splitting on whitespace.

    Example:
      "(adjacent box1 location2)" -> ["adjacent", "box1", "location2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the simplified Sokoban-like domain (secret_domain).
    The heuristic estimates how many actions are required to make every goal
    fact of the form (adjacent-2 BOX) true. It combines:
      - Navigation cost for the robot to reach the locations where it can
        operate on each box (based on static `clear` edges), and
      - Local box-operation cost to transform the box into the goal predicate
        (primarily the sequence "move" then "push", or just "push" if the box
        is already in a grabbed state `(at BOX)`).

    Assumptions
    -----------
    - Goals are of the form (adjacent-2 BOX). Each such BOX must be acted on
      by the robot to achieve the goal.
    - Static facts contain mappings of the form (at-robot BOX LOCATION) that
      indicate the robot location required to perform the push operation for BOX.
      If this mapping is missing for a box, the heuristic falls back to using
      any location that appears as "adjacent" for that box in the current state,
      or otherwise assumes the robot must go "somewhere" (the heuristic remains finite).
    - Robot navigation is possible along directed edges (clear L1 L2) using
      the `move-2` (and possibly `push-2`) actions. The heuristic uses these
      `clear` relations as a directed graph to compute movement costs.
    - Applying the local sequence to change a single box to `adjacent-2` is
      optimistically approximated as either:
         - cost 1 (push) if the box is already in the grabbed state `(at BOX)`,
         - cost 2 (move + push) otherwise.
      (More complex interactions between boxes are ignored for efficiency.)

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts and stores:
    - self.goals: set of goal fact strings (task.goals).
    - self.goal_boxes: list of boxes that must satisfy (adjacent-2 BOX).
    - self.box_required_robot_loc: mapping BOX -> LOCATION from static facts
      matching "(at-robot BOX LOCATION)".
    - self.clear_graph: a dictionary representing directed edges LOCATION -> set(NEIGHBORS)
      derived from static facts "(clear L1 L2)".
    This static preprocessing allows fast distance computations during heuristic calls.

    Step-By-Step Thinking for Computing Heuristic (per state)
    --------------------------------------------------------
    1. If all goal facts are already present in the state, return 0.
    2. Identify the robot's current location from the single state fact
       "(at-robot-2 LOCATION)". If multiple such facts exist, the heuristic uses
       the first; if none exists, the heuristic assumes the robot is not present
       and uses an arbitrary location (still returns finite values).
    3. Determine which boxes still need work (their (adjacent-2 BOX) goal is
       not in the current state).
    4. For each outstanding box, decide:
       - The LOCATION where the robot must be to perform the push. This is
         primarily taken from static (at-robot BOX LOCATION). If missing,
         fall back to any currently-adjacent location "(adjacent BOX L)" found
         in the state; if none exist, the heuristic still assigns a placeholder
         location so it remains finite.
       - The local operation overhead:
           * If "(at BOX)" is present in the state, we assume the box is
             already grabbed and only one push is required -> overhead = 1.
           * Otherwise, assume a move to grab and then a push -> overhead = 2.
    5. Treat the problem of visiting all required robot-locations as a simple
       greedy route-planning problem:
       - Start at the robot's current location.
       - Repeatedly go to the nearest remaining required LOCATION (according
         to the `clear` graph shortest-path distances), add that navigation
         cost to the heuristic, then add the local operation overhead for
         one box that uses that LOCATION. If multiple boxes require the same
         location, they are handled sequentially without extra navigation cost
         between them (i.e., only their per-box overhead is added).
    6. Return the total sum: navigation moves + per-box overheads. The result
       is an integer estimate of actions remaining; it is not admissible but
       is efficient to compute and captures robot travel cost and local box
       manipulation cost, which are the dominant contributors in this domain.
    """

    def __init__(self, task):
        # Goals and static facts
        self.goals = task.goals  # set of goal strings, e.g., "(adjacent-2 box1)"
        static_facts = task.static  # frozenset of static fact strings

        # Extract boxes that must have (adjacent-2 BOX) in the goal.
        self.goal_boxes = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "adjacent-2" and len(parts) >= 2:
                self.goal_boxes.append(parts[1])

        # Map each box -> required robot location for pushing, from static (at-robot BOX LOC).
        self.box_required_robot_loc = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot" and len(parts) >= 3:
                _, box, loc = parts[:3]
                self.box_required_robot_loc[box] = loc

        # Build directed graph of locations using the static `clear` facts:
        # (clear L1 L2) allows a directed move from L1 -> L2.
        self.clear_graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "clear" and len(parts) >= 3:
                _, l1, l2 = parts[:3]
                self.clear_graph.setdefault(l1, set()).add(l2)

        # Precompute list of all locations known from clear graph for fallback.
        self.all_locations = set(self.clear_graph.keys())
        for neighs in self.clear_graph.values():
            self.all_locations.update(neighs)

    def _bfs_distance(self, start, goal):
        """
        Compute directed shortest-path distance (number of move-2 actions)
        from start to goal using the clear_graph. If unreachable, return
        a conservative finite estimate (len(all_locations) + 5).
        """
        if start == goal:
            return 0
        if start not in self.clear_graph and start not in self.all_locations:
            # Unknown start: return a finite fallback
            return len(self.all_locations) + 5
        visited = {start}
        q = deque([(start, 0)])
        while q:
            loc, d = q.popleft()
            # neighbors may be empty set
            for nb in self.clear_graph.get(loc, ()):
                if nb == goal:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        # If not found, return a conservative finite upper bound
        return len(self.all_locations) + 5

    def __call__(self, node):
        """
        Compute the heuristic estimate for `node`.

        Returns an integer estimate of actions remaining to satisfy all (adjacent-2 BOX)
        goal facts.
        """
        state = node.state  # frozenset of fact strings

        # 1) If goal already reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Find robot current location from state fact "(at-robot-2 LOC)"
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at-robot-2" and len(parts) >= 2:
                robot_loc = parts[1]
                break
        # If robot location is unknown, pick an arbitrary known location so the heuristic remains finite.
        if robot_loc is None:
            robot_loc = next(iter(self.all_locations)) if self.all_locations else "unknown_loc"

        # 3) Which boxes still need to reach adjacent-2?
        outstanding_boxes = []
        for box in self.goal_boxes:
            goal_fact = f"(adjacent-2 {box})"
            if goal_fact not in state:
                outstanding_boxes.append(box)

        # 4) Precompute which boxes are already grabbed "(at BOX)"
        grabbed_boxes = {get_parts(f)[1] for f in state if get_parts(f)[0] == "at" and len(get_parts(f)) >= 2}

        # 5) For each outstanding box, determine target robot location and local overhead.
        #    We'll collect a multimap target_loc -> list of overheads for boxes that share the same target.
        loc_to_overheads = {}
        # Also collect adjacency facts from state to potentially choose fallbacks
        box_adjacent_locs = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) >= 3:
                _, box, loc = parts[:3]
                box_adjacent_locs.setdefault(box, set()).add(loc)

        for box in outstanding_boxes:
            # Prefer static mapping (at-robot BOX LOC)
            target_loc = self.box_required_robot_loc.get(box, None)
            if target_loc is None:
                # Fallback: if the box has any 'adjacent' locations in the current state, use one of them.
                adj_locs = box_adjacent_locs.get(box)
                if adj_locs:
                    # pick arbitrary one (heuristic remains optimistic)
                    target_loc = next(iter(adj_locs))
                else:
                    # final fallback: pick an arbitrary known location so distances remain finite
                    target_loc = next(iter(self.all_locations)) if self.all_locations else "unknown_loc"

            overhead = 1 if box in grabbed_boxes else 2  # 1 push if already grabbed, else move+push
            loc_to_overheads.setdefault(target_loc, []).append(overhead)

        # 6) Greedy route: start at robot_loc, repeatedly go to nearest target location,
        #    add navigation cost (BFS distance) and then add all overheads for boxes at that location.
        total_cost = 0
        remaining_locations = dict(loc_to_overheads)  # shallow copy

        current_loc = robot_loc
        # Precompute distance cache to avoid repeated BFS calls
        dist_cache = {}

        while remaining_locations:
            # Find nearest target location from current_loc
            best_loc = None
            best_dist = None
            for loc in remaining_locations.keys():
                key = (current_loc, loc)
                if key in dist_cache:
                    d = dist_cache[key]
                else:
                    d = self._bfs_distance(current_loc, loc)
                    dist_cache[key] = d
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_loc = loc

            # Add navigation cost (best_dist) and then per-box overheads at that location
            if best_dist is None:
                # No reachable locations found; add conservative finite cost
                total_cost += len(self.all_locations) + 5
            else:
                total_cost += best_dist

            # Once at best_loc, perform all box operations for boxes requiring that location.
            overheads = remaining_locations.pop(best_loc)
            total_cost += sum(overheads)

            # Robot is now at best_loc
            current_loc = best_loc

        # The heuristic value must be finite. Return integer.
        return int(total_cost)