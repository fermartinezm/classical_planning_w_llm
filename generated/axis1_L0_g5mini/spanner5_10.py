from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'spanner' domain.
    Estimates the number of actions required to tighten all loose nuts from a
    given state. The estimate accounts for:
      - walking along links (each walk action counts as 1),
      - picking up usable spanners (pickup_spanner, cost 1),
      - tightening nuts (tighten_nut, cost 1).
    The heuristic performs a fast greedy route construction that simulates the
    man picking up spanners (if needed) and walking to nuts to tighten them.
    It is not guaranteed to be admissible, but it is efficiently computable
    and informative for guiding greedy best-first search.

    Assumptions
    -----------
    - There is at least one "man" in the task. Men are identified as objects
      that appear as the subject of (at ...) facts but are not nuts or
      spanners (we detect spanners via 'usable' facts and nuts via 'loose'/'tightened').
    - The planner represents facts as strings like "(at bob shed)".
    - Links given in static facts represent directed walk edges (walk uses
      predicate (link start end)). Distances are computed respecting direction.
    - A usable spanner can be used exactly once (tighten_nut makes it not usable).
    - The man may carry multiple spanners simultaneously (the domain permits it).

    Heuristic Initialization
    ------------------------
    The constructor extracts static information from the task:
      - adjacency list of locations (directed) from static link facts. This is
        used to compute shortest path (in number of walk actions) between any
        two locations via BFS.
      - the set of goal facts (for quick goal checks).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state, the heuristic proceeds as follows:

    1. Parse the state facts to obtain:
       - the set and locations of loose nuts (each nut's location),
       - the set of usable spanners and whether they are currently carried or
         lying at a location,
       - the location of the man (if multiple men exist we pick one arbitrarily;
         the same logic extends if multiple men are present but the heuristic
         uses a single-man approximation),
       - which spanners are already carried by the man and are usable.

    2. If there are no loose nuts, return 0 (goal reached).

    3. Compute shortest-path distances (in number of walk actions) between
       locations using BFS on the directed link graph extracted from static facts.

    4. Build a greedy simulation to estimate the required sequence of actions:
       - Maintain:
         * current_location: where the man is,
         * carried_usable: how many usable spanners the man currently carries,
         * available_spanners: usable spanners that are still on the ground and
           not yet picked,
         * remaining_nuts: set of nut locations to be tightened.
       - While there are remaining nuts:
         a) If carried_usable > 0: go to the nearest remaining nut (by BFS
            distance), walk there (add distance), perform tighten (add 1),
            decrement carried_usable, mark that nut as done, set current_location
            to that nut's location.
         b) Otherwise (no usable spanners carried): go to the nearest available
            usable spanner, walk there (add distance), pick it up (add 1),
            increment carried_usable, remove that spanner from available_spanners,
            set current_location to the spanner's location.
       - The greedy choice (nearest next target) gives an informed but fast
         approximation of routing cost.

    5. The heuristic value returned is:
         total_walk_steps + total_pickup_actions + total_tighten_actions
       where total_tighten_actions equals the number of loose nuts.

    Notes on unsolvable / unreachable situations
    --------------------------------------------
    - If the heuristic detects that some nut or needed spanner is unreachable
      (no path in the directed link graph), it uses a large penalty value for
      that distance. For truly unsolvable states (e.g., not enough usable
      spanners), the heuristic will also grow large. The heuristic remains
      finite for solvable states.
    """

    def __init__(self, task):
        """Extract static information (links) and goals from the task."""
        self.goals = task.goals
        static_facts = task.static

        # Build directed adjacency list from (link from to) facts.
        self.adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "link" and len(parts) >= 3:
                u, v = parts[1], parts[2]
                self.adj.setdefault(u, []).append(v)

        # Collect the set of all locations known from the links.
        locs = set()
        for u, vs in self.adj.items():
            locs.add(u)
            for v in vs:
                locs.add(v)
        self.locations = locs

        # Precompute shortest path distances between every pair of locations
        # lazily on demand to keep initialization fast. We will compute BFS
        # distances when needed and cache them.
        self._dist_cache = {}

    def _bfs_distance(self, start, goal):
        """Return directed shortest path length (walk actions) from start to goal.
        Return a large number if unreachable. Distances are cached."""
        if start == goal:
            return 0
        key = (start, goal)
        if key in self._dist_cache:
            return self._dist_cache[key]

        # If either start or goal not in known locations, unreachable.
        if start not in self.locations or goal not in self.locations:
            self._dist_cache[key] = float("inf")
            return float("inf")

        # BFS
        q = deque()
        q.append((start, 0))
        visited = {start}
        found = float("inf")
        while q:
            node, d = q.popleft()
            for nei in self.adj.get(node, []):
                if nei in visited:
                    continue
                if nei == goal:
                    found = d + 1
                    q.clear()  # stop BFS
                    break
                visited.add(nei)
                q.append((nei, d + 1))
        self._dist_cache[key] = found
        return found

    def __call__(self, node):
        """
        Compute the heuristic estimate for the node's state.

        The heuristic is the estimated number of actions (walk, pickup_spanner,
        tighten_nut) to tighten all remaining loose nuts.
        """
        state = node.state

        # Parse the state into useful maps/sets.
        at_map = {}  # object -> location (for objects that have an at fact)
        carrying = {}  # man -> set(spanner)
        carried_by = {}  # spanner -> man
        usable_spanners = set()  # spanners that are currently usable
        loose_nuts = set()
        tightened_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                at_map[obj] = loc
            elif pred == "carrying" and len(parts) >= 3:
                man, sp = parts[1], parts[2]
                carrying.setdefault(man, set()).add(sp)
                carried_by[sp] = man
            elif pred == "usable" and len(parts) >= 2:
                usable_spanners.add(parts[1])
            elif pred == "loose" and len(parts) >= 2:
                loose_nuts.add(parts[1])
            elif pred == "tightened" and len(parts) >= 2:
                tightened_nuts.add(parts[1])

        # If no loose nuts remain -> goal reached.
        if not loose_nuts:
            return 0

        # Determine men: objects that appear as subject of 'at' but are not nuts or
        # spanners (we detect spanners via usable facts and nuts via loose/tightened).
        spanner_names = set(usable_spanners) | set(carried_by.keys())
        nut_names = loose_nuts | tightened_nuts

        men = []
        for obj, loc in at_map.items():
            if obj not in spanner_names and obj not in nut_names and loc in self.locations:
                men.append(obj)
        # If we still have carrying facts that indicate men not in at_map, ensure they are included.
        for man in carrying.keys():
            if man not in men:
                # If there is an explicit at for this man, it was already included.
                # Otherwise include it to avoid empty men list.
                men.append(man)

        if not men:
            # Fallback: try to infer a man by excluding known spanners and nuts and
            # location names from all objects appearing in state.
            all_objs = {get_parts(f)[1] for f in state if len(get_parts(f)) >= 2}
            candidates = [o for o in all_objs if o not in spanner_names and o not in nut_names and o not in self.locations]
            if candidates:
                men = [candidates[0]]

        # Use the first man as the actor for this heuristic approximation.
        man = men[0] if men else None

        # Determine man's current location.
        man_location = None
        if man and man in at_map:
            man_location = at_map[man]
        else:
            # If man isn't in at_map but carries things, try to retrieve location via any carried item's at (shouldn't happen in valid states),
            # otherwise treat as unknown -> unreachable.
            if man and man in carrying:
                # If he carries some spanner, that spanner shouldn't have an at fact.
                # Fallback: set to an arbitrary location from static to avoid crash.
                man_location = next(iter(self.locations)) if self.locations else None
            else:
                man_location = next(iter(self.locations)) if self.locations else None

        # Build usable spanners that are on the ground: map spanner -> location
        usable_on_ground = {}
        for sp in usable_spanners:
            if sp in carried_by:
                # carried usable spanner
                continue
            if sp in at_map:
                usable_on_ground[sp] = at_map[sp]
            else:
                # usable spanner but no at fact and not carried -> unknown; treat as unavailable on ground
                pass

        # Count usable spanners already carried by man
        carried_usable = 0
        if man and man in carrying:
            for sp in carrying[man]:
                if sp in usable_spanners:
                    carried_usable += 1

        # Prepare sets of nut locations (only loose nuts)
        nut_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] in loose_nuts:
                nut_locations[parts[1]] = parts[2]

        # If some loose nut has no at fact, treat it as unreachable -> large heuristic
        for nut in loose_nuts:
            if nut not in nut_locations:
                # Use large estimate
                return 100000

        # Greedy simulation:
        remaining_nuts = set(nut_locations.keys())
        # available spanners (on ground) as sp -> loc
        available_spanners = dict(usable_on_ground)
        # Note: carried usable spanners are considered available for immediate use (counted by carried_usable).
        curr_loc = man_location
        total_walk = 0
        pickups = 0
        tightens = 0  # will count equal to number of nuts processed

        # If there are zero usable spanners anywhere (not carried and none on ground),
        # then it's impossible to tighten; return large estimate.
        total_usable_available = carried_usable + len(available_spanners)
        if total_usable_available < len(remaining_nuts):
            # Not enough usable spanners to tighten all remaining nuts.
            # Return a large estimate but finite.
            # We still proceed with greedy to provide a smarter large value.
            pass

        # Greedy loop: while nuts remain, either use carried spanner for nearest nut,
        # or go pick nearest spanner.
        while remaining_nuts:
            if carried_usable > 0:
                # find nearest nut
                best_nut = None
                best_dist = float("inf")
                for nut in remaining_nuts:
                    nut_loc = nut_locations[nut]
                    d = self._bfs_distance(curr_loc, nut_loc)
                    if d < best_dist:
                        best_dist = d
                        best_nut = nut
                if best_nut is None:
                    # unreachable - break with large penalty
                    total_walk += 100000
                    break
                # walk to nut (best_dist may be inf)
                total_walk += best_dist if best_dist != float("inf") else 100000
                # tighten action
                tightens += 1
                carried_usable -= 1
                remaining_nuts.remove(best_nut)
                # update current location
                curr_loc = nut_locations[best_nut]
            else:
                # Need to pick up a usable spanner first.
                if not available_spanners:
                    # No spanners on ground left; cannot proceed normally.
                    # Give large penalty and stop to avoid infinite loop.
                    total_walk += 100000
                    break
                # choose nearest available spanner
                best_sp = None
                best_dist = float("inf")
                for sp, loc in available_spanners.items():
                    d = self._bfs_distance(curr_loc, loc)
                    if d < best_dist:
                        best_dist = d
                        best_sp = sp
                if best_sp is None:
                    total_walk += 100000
                    break
                # walk to spanner
                total_walk += best_dist if best_dist != float("inf") else 100000
                # pickup action
                pickups += 1
                carried_usable += 1
                # remove spanner from available
                curr_loc = available_spanners.pop(best_sp)

        # The number of tighten actions must equal number of loose nuts (but in case of early break we may be off).
        # Ensure we count all tightens required (if loop terminated prematurely add remaining nuts count).
        tightens += 0  # already counted in loop; if we added large penalty above, it's included.

        heuristic_value = total_walk + pickups + tightens
        # As a fallback ensure heuristic is at least number of remaining nuts (if greedy failed)
        if heuristic_value < len(loose_nuts):
            heuristic_value = len(loose_nuts)

        # Cap heuristic to a large finite number for safety
        if heuristic_value == float("inf"):
            heuristic_value = 100000

        return int(heuristic_value)