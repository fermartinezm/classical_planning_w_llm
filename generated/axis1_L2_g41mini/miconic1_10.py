from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(pxch7pi8 ps0y2bwz qklmztf3)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the 'secret_domain' domain (unkhrp2l).

    Summary:
    This heuristic estimates the number of actions required to achieve all goal facts
    of the form (guby8wxu ?obj), which represent the final desired state of objects.
    To reach (guby8wxu ?obj) from the initial state, the object ?obj must go through
    a sequence of states involving predicates (pxch7pi8 ?obj ?loc), (eacj8ftl ?obj),
    and finally (guby8wxu ?obj). Additionally, the location indicator (v0yy74n7 ?loc)
    can change via actions using the adjacency relationship (pvhp8m0p ?from ?to).

    Assumptions:
    - Each object of type jxhpkqr5 can only be in one v0yy74n7 location at a time.
    - The goal requires each goal object to have predicate (guby8wxu ?obj).
    - The main chain to achieve (guby8wxu ?obj) is:
          1) (pxch7pi8 ?obj ?loc1) in the initial state
          2) Using action makpz2tc, convert (pxch7pi8 ?obj ?loc) into (eacj8ftl ?obj)
          3) Using action jg64qouf on location with (v0yy74n7 ?loc), move to achieve (guby8wxu ?obj)
    - Locations ql6b2h31 are connected by the (pvhp8m0p ?from ?to) predicate forming a directed graph.
    - Actions jt9wnwgs and v90zoaro move the "v0yy74n7" marker between locations following (pvhp8m0p) arcs.
    - The heuristic counts the minimal estimated "distance" the marker v0yy74n7 needs to move to reach
      the location for finalizing each object, plus the steps required to convert object predicates.

    Heuristic Initialization:
    - Extract goal objects of type jxhpkqr5 that require (guby8wxu ?obj).
    - Build adjacency lists for location objects (ql6b2h31) based on static (pvhp8m0p) facts.
    - For each object, record its initial known location from (pxch7pi8 ?obj ?loc).
    - Precompute shortest paths between all locations using BFS for efficient distance calculation.

    Step-By-Step Thinking for Computing Heuristic:
    1) Identify all goal objects (?obj) that need (guby8wxu ?obj).
    2) For each goal object:
       a) Check if (guby8wxu ?obj) already holds in the current state → cost 0 for this object.
       b) Otherwise:
          i) If (eacj8ftl ?obj) is true, the object is "ready" to be finalized via jg64qouf on some location.
             - The marker (v0yy74n7 ?loc) must be at that location to apply jg64qouf.
             - For heuristic, estimate the minimal steps to move the marker from its current location to
               a location that satisfies (pahbmn6t ?obj ?loc) (mandatory precondition of jg64qouf).
          ii) Else if (pxch7pi8 ?obj ?loc) is true, the object needs to be transformed first by action makpz2tc,
               which requires (v0yy74n7 ?loc) to be at that location.
               Then (eacj8ftl ?obj) becomes true (handled as above).
          iii) Otherwise, the object is unreachable for this heuristic (count large cost).
    3) There's only one "active" marker (only one location at a time with (v0yy74n7 ?loc)), so 
       estimate the cost as the sum over objects of:
          - The steps to move the marker to required location for the intermediate steps (with shortest paths)
          - The fixed cost of applying makpz2tc or jg64qouf (1 action each)
    4) For marker position, take the current location from state.
    5) For multiple goal objects, approximate cost by summing individual minimal costs ignoring marker travel order
       (since ordering is complex), but improve by considering minimal marker travel distance from its current
       location to all required location visits for finalization and transformation steps.
    6) Return the total estimated number of actions needed; zero if all goals achieved.

    Note: This heuristic is not guaranteed to be admissible but aims to guide greedy best-first search efficiently.
    """

    def __init__(self, task):
        # Store goals and static facts.
        self.goals = task.goals
        self.static = task.static

        # Extract goal objects that must satisfy (guby8wxu ?obj).
        self.goal_objects = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "guby8wxu" and len(parts) == 2:
                self.goal_objects.add(parts[1])

        # Extract adjacency graph of locations from static facts: (pvhp8m0p ?from ?to)
        self.adjacency = {}
        self.locations = set()
        for fact in self.static:
            if match(fact, "pvhp8m0p", "*", "*"):
                _, frm, to = get_parts(fact)
                self.locations.add(frm)
                self.locations.add(to)
                self.adjacency.setdefault(frm, set()).add(to)
        # Ensure all locations appear even if no outgoing edges
        for loc in list(self.locations):
            self.adjacency.setdefault(loc, set())

        # Precompute shortest distances between all pairs of locations using BFS
        self.shortest_dist = {}
        for loc in self.locations:
            self.shortest_dist[loc] = self._bfs_shortest_paths(loc)

        # For each object (jxhpkqr5) and location from static facts, build inverted indices:
        # Objects starting at locations (pxch7pi8 ?obj ?loc)
        self.obj_init_locations = {}
        for fact in self.static:
            if match(fact, "pxch7pi8", "*", "*"):
                _, obj, loc = get_parts(fact)
                self.obj_init_locations[obj] = loc

        # Objects related to locations by (pahbmn6t ?obj ?loc) needed for the final action jg64qouf
        self.obj_pahbmn6t_locs = {}
        for fact in self.static:
            if match(fact, "pahbmn6t", "*", "*"):
                _, obj, loc = get_parts(fact)
                self.obj_pahbmn6t_locs.setdefault(obj, set()).add(loc)

        # Objects related to locations by (pxch7pi8 ?obj ?loc) (used for pre-final action makpz2tc)
        # These static facts are the same as obj_init_locations but might be repeated in static, keep for generality
        self.obj_pxpch7pi8_locs = {}
        for fact in self.static:
            if match(fact, "pxch7pi8", "*", "*"):
                _, obj, loc = get_parts(fact)
                self.obj_pxpch7pi8_locs.setdefault(obj, set()).add(loc)

    def _bfs_shortest_paths(self, start):
        """Return a dict of shortest distances from start location to all reachable locations."""
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state

        # Shortcut: if goal already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Determine current v0yy74n7 location (exactly one location has this predicate true).
        current_marker_loc = None
        for fact in state:
            if match(fact, "v0yy74n7", "*"):
                current_marker_loc = get_parts(fact)[1]
                break

        # For each goal object, estimate cost to achieve (guby8wxu ?obj).

        total_cost = 0

        for obj in self.goal_objects:

            # 1) Check if already at goal: (guby8wxu ?obj) in state → cost 0 for this object
            if f"(guby8wxu {obj})" in state:
                continue  # no cost for this object

            # 2) Check if (eacj8ftl ?obj) true: object ready for final action jg64qouf at some location
            has_eacj8ftl = f"(eacj8ftl {obj})" in state

            # 3) Check if (pxch7pi8 ?obj ?loc) true: object needs to be transformed first by makpz2tc action
            pxch7pi8_locs = {
                get_parts(fact)[2]
                for fact in state
                if match(fact, "pxch7pi8", obj, "*")
            }

            # Final location(s) needed for finalization (jg64qouf requires (pahbmn6t ?obj ?loc))
            final_locs = self.obj_pahbmn6t_locs.get(obj, set())
            if not final_locs:
                # If no final location static info, fallback to found pxch7pi8_locs or empty
                final_locs = pxch7pi8_locs if pxch7pi8_locs else set()

            # If no location info available, large penalty:
            if not final_locs and not pxch7pi8_locs:
                total_cost += 50  # unreachable, large penalty
                continue

            # Current location of marker needed for distance calculation.
            if current_marker_loc is None:
                # If cannot find marker location, defaultPenalty
                total_cost += 20
                continue

            # Define helper to find minimal distance from marker to a set of target locations
            def min_dist_to_locs(marker_loc, targets):
                distances = self.shortest_dist.get(marker_loc, {})
                dist_candidates = [distances[loc] for loc in targets if loc in distances]
                if dist_candidates:
                    return min(dist_candidates)
                else:
                    # No path to any target location
                    return 50

            # Cost constants per action (nominal 1 step each)
            COST_MOVE_MARKER = 1
            COST_MAKPZ2TC = 1
            COST_JG64QOUF = 1

            if has_eacj8ftl:
                # The object is ready for finalization.
                # The marker must be moved to a location where (pahbmn6t ?obj ?loc) holds.
                dist = min_dist_to_locs(current_marker_loc, final_locs)
                total_cost += dist * COST_MOVE_MARKER + COST_JG64QOUF

            elif pxch7pi8_locs:
                # Object needs transformation first by makpz2tc at one of pxch7pi8 locations
                dist_to_transform = min_dist_to_locs(current_marker_loc, pxch7pi8_locs)
                # After transformation, object gets (eacj8ftl ?obj), so then must finalize at locations from final_locs
                # Estimate distance from transform location to final location:
                min_intermediate = 50
                for transform_loc in pxch7pi8_locs:
                    dist_transform_to_final = min_dist_to_locs(transform_loc, final_locs)
                    if dist_transform_to_final < min_intermediate:
                        min_intermediate = dist_transform_to_final

                cost = (
                    dist_to_transform * COST_MOVE_MARKER
                    + COST_MAKPZ2TC
                    + min_intermediate * COST_MOVE_MARKER
                    + COST_JG64QOUF
                )
                total_cost += cost

            else:
                # Neither eacj8ftl nor pxch7pi8 present for object: large penalty as heuristic fallback
                total_cost += 50

        # Return estimated heuristic cost (must be >= 0)
        return total_cost
