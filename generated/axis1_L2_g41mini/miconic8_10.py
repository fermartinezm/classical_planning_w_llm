from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(pxch7pi8 obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to reach a goal state where
    all goal facts (typically of the form (guby8wxu ?jxhpkqr5)) hold.
    It focuses on how many jxhpkqr5 objects still need to be transformed from state (eacj8ftl)
    to the goal state (guby8wxu). The heuristic encodes necessary steps to "activate" these objects
    by moving or changing properties of related ql6b2h31 objects.

    Assumptions
    -----------
    - The goal facts require certain jxhpkqr5 objects to have the predicate (guby8wxu).
    - Objects of type jxhpkqr5 transition from having (eacj8ftl) true to (guby8wxu) true.
    - The transition depends on them being "unlocked" from the predicate (pxch7pi8) and (pahbmn6t)
      relations, and also depending on the location state represented by (v0yy74n7).
    - Movements between ql6b2h31 objects are restricted by (pvhp8m0p) relations forming a graph.
    - The heuristic ignores operator costs, focusing on estimating a minimum number of necessary actions.
    - The heuristic is not guaranteed admissible but is efficient and informative.

    Heuristic Initialization
    ------------------------
    - Extracts goals that require (guby8wxu ?jxhpkqr5) to hold at the end.
    - Extracts static facts describing:
      - (pvhp8m0p ?loc1 ?loc2): directed links between locations (ql6b2h31 objects),
      - (pahbmn6t ?jxhpkqr5 ?loc): associations between jxhpkqr5 objects and locations,
      - (pxch7pi8 ?jxhpkqr5 ?loc): initial locked state of jxhpkqr5 objects with locations.
    - Builds adjacency maps from static (pvhp8m0p) facts for efficient navigation/estimation.
    - Records which jxhpkqr5 objects must reach goal state (guby8wxu).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each jxhpkqr5 object required to be "activated" (i.e. in (guby8wxu) goal):

    1. If the jxhpkqr5 object already satisfies (guby8wxu) in the current state,
       it requires no further actions.

    2. Otherwise, it must be "activated". Activation steps roughly correspond to:
       a. If currently (eacj8ftl ?jxhpkqr5) holds, it means the object is "ready" to be activated
          via the jg64qouf action that requires the object to be at a location with (v0yy74n7).
          Activation thus requires moving the "v0yy74n7" property to the location matching
          this object’s associated location (from (pahbmn6t ?jxhpkqr5 ?loc)).

       b. If not (eacj8ftl) but (pxch7pi8) holds, the object needs to be "unlocked" first via the makpz2tc action:
          this requires the "v0yy74n7" predicate to be at the object’s locked location.
          So the locker location must be reached by moving the "v0yy74n7".

    3. The heuristic estimates the minimal number of moves of the "v0yy74n7" predicate between locations
       to reach the required position for unlocking or activation, plus the number of required
       unlock/activate actions (typically one per step).

    4. To estimate minimal moves, the heuristic uses a shortest path in the directed graph defined by (pvhp8m0p).

    5. Summation over all jxhpkqr5 objects not completed gives the heuristic value.

    6. Finally, the heuristic returns 0 for states fulfilling all goals.

    Note:
    - Moving "v0yy74n7" between locations is the main action cost driver.
    - Each movement corresponds to one of the jt9wnwgs/v90zoaro actions that move "v0yy74n7".
    - Unlocking and activating each jxhpkqr5 requires one action each after positioning "v0yy74n7".
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract which jxhpkqr5 objects must hold (guby8wxu) in goal
        self.objects_goal_guby8wxu = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "guby8wxu" and len(parts) == 2:
                self.objects_goal_guby8wxu.add(parts[1])

        # Build adjacency map for location navigation from static (pvhp8m0p) facts
        self.adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p" and len(parts) == 3:
                src, tgt = parts[1], parts[2]
                self.adj.setdefault(src, set()).add(tgt)

        # Map jxhpkqr5 to their locked locations from (pxch7pi8) static facts
        self.locked_location = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pxch7pi8" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.locked_location[obj] = loc

        # Map jxhpkqr5 to their associated location (for activation) from (pahbmn6t) static facts
        self.activation_location = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pahbmn6t" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.activation_location[obj] = loc

    def _shortest_path_length(self, start, goal):
        """
        Compute shortest path length in directed graph self.adj from start to goal.
        Returns integer length or float('inf') if no path found.
        Uses BFS for efficiency.
        """
        if start == goal:
            return 0
        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.adj.get(current, ()):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float("inf")

    def __call__(self, node):
        state = node.state

        # Shortcut: if goals achieved, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract which jxhpkqr5 objects currently hold (guby8wxu)
        objs_guby8wxu = {get_parts(f)[1] for f in state if match(f, "guby8wxu", "*")}

        # Extract which jxhpkqr5 currently hold (eacj8ftl)
        objs_eacj8ftl = {get_parts(f)[1] for f in state if match(f, "eacj8ftl", "*")}

        # Extract current location of "v0yy74n7"
        loc_v0yy74n7 = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "v0yy74n7" and len(parts) == 2:
                loc_v0yy74n7 = parts[1]
                break
        # If unknown, no moves possible; treat as infinite distance.
        if loc_v0yy74n7 is None:
            return float("inf")

        heuristic_cost = 0

        for obj in self.objects_goal_guby8wxu:
            if obj in objs_guby8wxu:
                # Object already activated: no cost for this object.
                continue

            # Steps needed to activate this object:
            # 1) If it still has (pxch7pi8) (locked), unlocking requires v0yy74n7 at locked location + makpz2tc action
            # 2) Then if it has (eacj8ftl), activation requires v0yy74n7 at activation location + jg64qouf action

            # Check if locked (pxch7pi8) holds
            has_locked = False
            if obj in self.locked_location:
                locked_loc = self.locked_location[obj]
                if match(f'(pxch7pi8 {obj} {locked_loc})', "pxch7pi8", obj, locked_loc):
                    # Need to confirm if this fact exists in current state (or else it was deleted)
                    # Because pxch7pi8 is static, it is in static only - so check if unlocked by absence in state
                    # The operator makpz2tc deletes pxch7pi8 predicate, so if it's still present in state, locked remains
                    if f"(pxch7pi8 {obj} {locked_loc})" in state:
                        has_locked = True
            else:
                has_locked = False

            # Check if object is eacj8ftl (ready to activate)
            is_ready = obj in objs_eacj8ftl

            # Moves required for unlocking
            if has_locked:
                dist_to_locked = self._shortest_path_length(loc_v0yy74n7, locked_loc)
                if dist_to_locked == float("inf"):
                    # Unreachable location means heuristic infinitely large -> large number
                    return float("inf")

                # Move cost plus one unlock action (makpz2tc)
                # + move cost = dist_to_locked moves of v0yy74n7 location
                heuristic_cost += dist_to_locked + 1

                # After unlocking, v0yy74n7 is at locked_loc
                loc_v0yy74n7 = locked_loc

                # Now pxch7pi8 is removed, so has_locked is false and is_ready must be true
                # but in state it might not be; we must consider activation still necessary

                # Mark as ready after unlocking for estimation:
                is_ready = True

            # Moves required for activation
            if is_ready:
                # Activation location where jg64qouf must be applied:
                activation_loc = self.activation_location.get(obj)
                if activation_loc is None:
                    # Defensive: if no activation location known, heuristic infinite.
                    return float("inf")

                dist_to_activation = self._shortest_path_length(loc_v0yy74n7, activation_loc)
                if dist_to_activation == float("inf"):
                    return float("inf")

                # Move cost plus one activation action (jg64qouf)
                heuristic_cost += dist_to_activation + 1

                # v0yy74n7 ends at activation_loc, update
                loc_v0yy74n7 = activation_loc

            else:
                # Neither locked nor ready (possibly missing facts), treat as requiring unlocking and activation both:
                # But facts should be consistent; as safe fallback:
                # Try to move to locked_location if known, else to activation_location, then add two actions
                unlock_loc = self.locked_location.get(obj)
                activation_loc = self.activation_location.get(obj)

                # Pick where to go first: prefer locked_loc if present, else activation_loc
                first_loc = unlock_loc if unlock_loc is not None else activation_loc
                if first_loc is None:
                    return float("inf")

                dist_to_first = self._shortest_path_length(loc_v0yy74n7, first_loc)
                if dist_to_first == float("inf"):
                    return float("inf")

                # Add moves + two actions (unlock + activate)
                heuristic_cost += dist_to_first + 2

                # Update loc_v0yy74n7 position
                loc_v0yy74n7 = first_loc

        return heuristic_cost
