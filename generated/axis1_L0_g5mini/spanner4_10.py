from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10**6


def get_parts(fact):
    """Remove surrounding parentheses and split into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the "spanner" domain estimating the number of actions required
    to tighten all loose nuts. It reasons about distances on the location graph,
    the current position of the man(s), and the locations of usable spanners
    (including spanners currently being carried). The heuristic greedily assigns
    available usable spanners to loose nuts, simulating a single agent moving
    from one nut to the next and consuming spanners as they are used.

    Assumptions
    -----------
    - The search tasks typically involve one man, but the heuristic attempts to
      handle multiple men by selecting a man location found in the state (first
      reasonable candidate). The planner itself handles multiple agents; this
      heuristic is primarily designed for the single-man case.
    - A spanner can be used exactly once (after tighten it becomes not usable).
    - Only spanners that are currently usable (predicate (usable S)) can be
      used to tighten nuts.
    - If a usable spanner is being carried by a man, the spanner is considered
      available at the man's current location and no pickup action is needed
      for that spanner.
    - If there are fewer usable spanners than loose nuts the heuristic returns a
      large finite value (representing effectively unreachable under current
      resources).
    - Links are undirected (walk is allowed along either direction of a link).

    Heuristic Initialization
    ------------------------
    The constructor extracts static "link" facts and constructs an undirected
    adjacency map over locations. It also stores the task goals (not strictly
    needed here aside from possibly identifying nuts via goal predicates).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Parse the current state to extract:
       - All loose nuts (facts "(loose N)").
       - All usable spanners (facts "(usable S)").
       - All carrying relations "(carrying M S)" so that carried spanners are
         treated as available at the corresponding man's location.
       - All at-relations "(at X L)" to map objects (men, spanners, nuts) to locations.
    2. Build a graph of locations from static "(link L1 L2)" facts and be able
       to compute shortest path distances (in number of walk actions) between
       any two locations via BFS.
    3. Create the multiset of available usable spanners with their current
       locations. A spanner that is both (usable S) and (carrying M S) is
       available at M's location (and pickup is not required). A usable spanner
       that is not carried is available at its "at" location (requires pickup).
    4. Identify a starting man location:
       - If any carrying facts exist, use the location of the man from the
         carrying fact(s) (prefer the first found man).
       - Otherwise, pick a reasonable candidate from the "at" facts that is not
         a nut name or a spanner name (typically the unique man).
    5. Greedily plan a sequence to tighten all loose nuts:
       - For the current man location, for each remaining loose nut and each
         available spanner compute the (optimistic) cost:
             cost = dist(man_loc, sp_loc)
                    + pickup_cost (0 if spanner is already carried by man, else 1)
                    + dist(sp_loc, nut_loc)
                    + 1 (tighten action)
         - Choose the spanner-nut pair with minimal cost, add the cost to the
           heuristic total, remove that spanner from availability and set the
           man's location to the nut location (after tightening).
       - Repeat until all loose nuts are assigned or until no usable spanners
         remain.
    6. If there are not enough usable spanners to tighten all loose nuts, the
       heuristic returns a large finite value (INF) to indicate impossibility
       with current resources; otherwise it returns the computed total cost.
    """

    def __init__(self, task):
        # store goals (may be useful); static facts used to compute links
        self.goals = task.goals
        static_facts = task.static

        # build undirected adjacency map of locations from "(link a b)" facts
        self.adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

    def _bfs_distance(self, start, goal):
        """Return shortest distance in edges between start and goal locations."""
        if start == goal:
            return 0
        if start not in self.adj or goal not in self.adj:
            return INF
        q = deque([(start, 0)])
        visited = {start}
        while q:
            loc, d = q.popleft()
            for nb in self.adj.get(loc, ()):
                if nb == goal:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        return INF

    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # Parse state into useful maps
        at_map = {}  # object -> location for "(at OBJ LOC)"
        carrying = []  # list of (man, spanner)
        usable_spanners = set()
        loose_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                at_map[obj] = loc
            elif pred == "carrying" and len(parts) >= 3:
                man, sp = parts[1], parts[2]
                carrying.append((man, sp))
            elif pred == "usable" and len(parts) >= 2:
                usable_spanners.add(parts[1])
            elif pred == "loose" and len(parts) >= 2:
                loose_nuts.add(parts[1])

        # If no loose nuts, we're at goal (heuristic 0)
        if not loose_nuts:
            return 0

        # Determine candidate men and starting location(s)
        men_locations = {}
        for man, sp in carrying:
            # man must have an "at" fact for location
            if man in at_map:
                men_locations[man] = at_map[man]

        # If no carrying facts gave man locations, try to infer men from at_map:
        if not men_locations:
            # Identify known nuts and spanners so we can exclude them when selecting man
            known_nuts = set(loose_nuts)
            known_spanners = set(usable_spanners)
            # also include spanners that appear in carrying facts (even if not usable)
            for _, sp in carrying:
                known_spanners.add(sp)
            # Choose any object from at_map that is not a nut and not a spanner:
            for obj, loc in at_map.items():
                if obj not in known_nuts and obj not in known_spanners:
                    men_locations[obj] = loc
                    break
        # If still empty (very unlikely), pick any location from at_map as start
        if not men_locations:
            # choose any object's location (fallback)
            if at_map:
                some_loc = next(iter(at_map.values()))
            else:
                # no location info available: return large heuristic
                return INF
            men_locations["__unknown_man__"] = some_loc

        # For our greedy single-agent simulation pick a single man (first entry)
        man_name, curr_loc = next(iter(men_locations.items()))

        # Build list of available usable spanners with locations and carried status
        # available_spanners: dict spanner -> (location, carried_by_man_bool)
        available_spanners = {}
        # map which spanner is currently carried by which man (if any)
        sp_carried_by = {sp: m for (m, sp) in carrying}
        for sp in usable_spanners:
            if sp in sp_carried_by:
                # if the spanner is carried, get the carrier location (if known)
                carrier = sp_carried_by[sp]
                loc = at_map.get(carrier, None)
                if loc is not None:
                    available_spanners[sp] = (loc, carrier == man_name)
            else:
                # otherwise the spanner should be in at_map
                if sp in at_map:
                    available_spanners[sp] = (at_map[sp], False)
                else:
                    # usable spanner but no known location: ignore it (can't use)
                    pass

        # Nuts must be at locations (from at_map)
        nut_locations = {}
        for nut in loose_nuts:
            if nut in at_map:
                nut_locations[nut] = at_map[nut]
            else:
                # loose nut with no known location -> unreachable in our model
                return INF

        # If not enough usable spanners to cover the loose nuts, return large value
        if len(available_spanners) < len(nut_locations):
            return INF

        total_cost = 0

        # Greedily assign spanners to nuts minimizing incremental cost from curr_loc
        remaining_nuts = set(nut_locations.keys())
        available = dict(available_spanners)  # mutable copy

        while remaining_nuts:
            best = None  # tuple (cost, nut, sp)
            for nut in remaining_nuts:
                nut_loc = nut_locations[nut]
                for sp, (sp_loc, sp_is_carried_by_man) in available.items():
                    # distance from current man location to spanner location
                    d1 = self._bfs_distance(curr_loc, sp_loc)
                    # pickup cost: 0 if the man already carries that spanner, else 1
                    pickup = 1
                    if sp_is_carried_by_man and sp_loc == curr_loc:
                        pickup = 0
                    # distance from spanner location to nut location
                    d2 = self._bfs_distance(sp_loc, nut_loc)
                    cost = d1 + pickup + d2 + 1  # +1 for tighten action
                    if cost >= INF:
                        continue
                    if best is None or cost < best[0]:
                        best = (cost, nut, sp)
            if best is None:
                # no feasible assignment (disconnected graph etc.)
                return INF
            cost, chosen_nut, chosen_sp = best
            total_cost += cost
            # update current man location to the nut location (man ends up at nut)
            curr_loc = nut_locations[chosen_nut]
            # remove used nut and spanner
            remaining_nuts.remove(chosen_nut)
            del available[chosen_sp]

            # After using a carried spanner it becomes unusable; our removal models that.

        # Return computed heuristic cost (integer)
        return int(total_cost)