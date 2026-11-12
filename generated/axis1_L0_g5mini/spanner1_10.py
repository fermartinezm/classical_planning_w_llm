from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split the fact into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the "spanner" domain. The heuristic estimates
    the number of actions required to tighten all nuts required by the goal.
    It reasons about the location of the single man, the locations of usable
    spanners (including those currently carried), and the locations of loose
    nuts. The estimate is a fast, greedy, sequential assignment of spanners to
    nuts minimizing per-nut cost (walks + pickup + tighten). The heuristic is
    not guaranteed to be admissible but aims to be informative and cheaply
    computable so that greedy best-first search expands fewer nodes.

    Assumptions
    -----------
    - There is (effectively) a single human/man relevant for planning.
    - A spanner can be used at most once because 'tighten_nut' makes the
      spanner unusable.
    - The planner's state encodes:
        * (at <obj> <loc>) for objects/agents,
        * (carrying <man> <spanner>) for carried spanners,
        * (usable <spanner>) for spanners that can be used,
        * (loose <nut>) and (tightened <nut>) for nut status.
    - Walking edges (link facts) are treated as undirected for cost
      estimation (optimistic and efficient to compute).
    - Distances between locations are measured as number of walk actions
      (each 'walk' counts as one action). Pickup and tighten each count
      as one action.

    Heuristic Initialization
    ------------------------
    - Save the set of goal facts (task.goals).
    - Parse static link facts (task.static) to construct an undirected graph
      of locations for fast BFS distance queries.
    - No other static typing information is required.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts are already true in the state, return 0.
    2. Parse state facts to discover:
       - Which nuts still need to be tightened (those goals not yet satisfied
         and present as loose in the state).
       - Which spanners are usable (presence of (usable <s>)) and their
         locations (either carried by the man or at some location).
       - The man's current location (found by an (at <x> <loc>) fact where
         <x> is not a known spanner or nut name).
    3. Precompute shortest path distances between locations on demand using
       BFS on the undirected link graph. If two locations are disconnected,
       treat the distance as a large finite number (so heuristic remains
       finite for solvable situations where links exist).
    4. Greedily assign usable spanners to remaining nuts one by one:
       - For each remaining nut, compute the minimal cost to tighten it using
         any currently available usable spanner. Two cases for a given
         spanner:
           a) spanner is already carried by the man:
              cost = distance(man_loc, nut_loc) + 1 (tighten)
           b) spanner is not carried:
              cost = distance(man_loc, spanner_loc) + 1 (pickup) +
                     distance(spanner_loc, nut_loc) + 1 (tighten)
       - Pick the (nut, spanner) pair with the smallest cost, add that cost
         to the total, update the man's location to the nut's location, and
         mark the chosen spanner as consumed (no longer usable).
       - Repeat until all nuts are paired/handled.
    5. Return the summed cost as the heuristic estimate.

    Notes
    -----
    - The greedy assignment is a heuristic approximation and may not be
      optimal globally (it ignores opportunities to pick multiple spanners
      on one route or reorder tightening for less travel). This trade-off
      yields a fast, informative estimate that is effective in practice
      for guiding greedy best-first search.
    """

    def __init__(self, task):
        """Extract goals and build undirected adjacency from static link facts."""
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Build undirected adjacency for locations from (link a b) facts.
        self.adj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "link" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

    def bfs_distance(self, start, goal):
        """Return shortest-path distance (number of walks) between two locations.
        If start == goal return 0. If unreachable return a large finite cost."""
        if start == goal:
            return 0
        if start not in self.adj or goal not in self.adj:
            # If a location does not appear in adjacency, it may be isolated;
            # return large finite value to keep heuristic finite for solvable states.
            return 1000
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            for nb in self.adj.get(node, ()):
                if nb == goal:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        return 1000

    def __call__(self, node):
        """Compute heuristic estimate as described in the docstring."""
        state = set(node.state)

        # 1) If all goals are satisfied, return 0.
        if self.goals <= state:
            return 0

        # Parse state facts into helpful structures:
        # - loose_nuts: set of nut names that are loose and needed by goals
        # - tightened_nuts: set of already tightened nuts
        # - usable_spanners: set of spanner names that are usable
        # - carrying relations and at-locations
        loose_nuts = set()
        tightened_nuts = set()
        usable_spanners = set()
        carrying = {}  # carrying[spanner] = man_name
        at = {}  # at[object] = location

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "loose" and len(parts) >= 2:
                loose_nuts.add(parts[1])
            elif pred == "tightened" and len(parts) >= 2:
                tightened_nuts.add(parts[1])
            elif pred == "usable" and len(parts) >= 2:
                usable_spanners.add(parts[1])
            elif pred == "carrying" and len(parts) >= 3:
                man_name, sp = parts[1], parts[2]
                carrying[sp] = man_name
            elif pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                at[obj] = loc

        # Determine which nuts still need to be tightened according to goals.
        # Goals typically contain (tightened nutX). We consider nuts in goals that
        # are not yet true in the state as targets.
        target_nuts = []
        for g in self.goals:
            gp = get_parts(g)
            if gp and gp[0] == "tightened" and len(gp) >= 2:
                nut = gp[1]
                # If already tightened in state, skip
                if f"(tightened {nut})" in state:
                    continue
                # Only consider nuts that are loose (or at least present) in state
                # If a nut is not loose but not tightened either, treat as loose:
                target_nuts.append(nut)

        # If no nuts to tighten, heuristic is zero
        if not target_nuts:
            return 0

        # Identify all spanner names we can detect:
        # from usable facts, from carrying facts, and from at facts for objects
        spanner_names = set(usable_spanners) | set(carrying.keys())
        # Also try to detect spanners from at facts whose object name begins with "spanner"
        for obj in list(at.keys()):
            if obj.startswith("spanner"):
                spanner_names.add(obj)

        # Determine the man name: prefer the subject in carrying facts; otherwise
        # choose an 'at' subject that is neither a known spanner nor a known nut.
        man_name = None
        if carrying:
            # any carrying fact gives the man name (value in carrying dict)
            # but carrying maps spanner->man, so pick any man
            man_name = next(iter(carrying.values()))
        else:
            # find a subject in 'at' that is not recognized as a spanner or nut
            for obj, loc in at.items():
                if obj not in spanner_names and obj not in loose_nuts and obj not in tightened_nuts:
                    man_name = obj
                    break
            # If not found, fallback to the first 'at' subject (safe if only one man)
            if man_name is None and at:
                man_name = next(iter(at.keys()))

        man_loc = at.get(man_name, None)

        # Build a list of available usable spanners with their current locations and carried flags.
        # Each spanner record: (name, loc, carried_boolean)
        spanner_records = {}
        for s in spanner_names:
            s_usable = s in usable_spanners
            if not s_usable:
                continue  # only consider usable spanners for assignment
            carried = s in carrying
            if carried:
                # carried by the man; its location is man's location (if known)
                loc = man_loc
            else:
                loc = at.get(s, None)
            spanner_records[s] = {"loc": loc, "carried": carried}

        # If there are fewer usable spanners than nuts to tighten, the problem is
        # unsolvable (given domain dynamics). Still return a finite heuristic:
        # overshoot by assuming additional pickups would be required (large cost).
        if len(spanner_records) < len(target_nuts):
            # return a high estimate but finite
            # Base: for each missing spanner assume 1000 cost.
            missing = len(target_nuts) - len(spanner_records)
            base_estimate = missing * 1000
        else:
            base_estimate = 0

        # Greedy assignment simulation: at each step pick the (nut, spanner) pair
        # that yields minimal incremental cost, update man location and consume spanner.
        remaining_spanners = dict(spanner_records)  # mutable copy
        remaining_nuts = list(target_nuts)
        total_cost = base_estimate

        # If we don't know man's location, assume an arbitrary location that is
        # cheap: pick the location of the closest spanner or nut if available.
        if man_loc is None:
            possible_locs = set()
            for sr in remaining_spanners.values():
                if sr["loc"]:
                    possible_locs.add(sr["loc"])
            for n in remaining_nuts:
                nloc = at.get(n)
                if nloc:
                    possible_locs.add(nloc)
            # If still empty, default to a single arbitrary location name to avoid errors.
            man_loc = next(iter(possible_locs), "unknown_loc")
            # ensure adjacency contains this loc to avoid BFS issues
            if man_loc not in self.adj:
                self.adj.setdefault(man_loc, set())

        # Perform greedy matching
        while remaining_nuts and remaining_spanners:
            best = None  # (cost, nut, spanner, new_man_loc)
            for nut in remaining_nuts:
                nut_loc = at.get(nut)
                if nut_loc is None:
                    # If nut has no known location, treat as expensive
                    nut_dist_penalty = 1000
                else:
                    nut_dist_penalty = 0
                for sname, rec in remaining_spanners.items():
                    s_loc = rec["loc"]
                    carried = rec["carried"]
                    if carried:
                        # cost: walk from current man location to nut, then tighten
                        d = self.bfs_distance(man_loc, nut_loc) if nut_loc else 1000
                        cost = d + 1  # tighten
                    else:
                        # cost: walk to spanner, pickup, walk to nut, tighten
                        d1 = self.bfs_distance(man_loc, s_loc) if s_loc else 1000
                        d2 = self.bfs_distance(s_loc, nut_loc) if (s_loc and nut_loc) else 1000
                        cost = d1 + 1 + d2 + 1  # pickup + tighten
                    # add nut location missing penalty if applicable
                    cost += nut_dist_penalty
                    if best is None or cost < best[0]:
                        best = (cost, nut, sname, nut_loc)
            if best is None:
                # No feasible pair found (likely because of missing locations); bail out.
                total_cost += 1000 * len(remaining_nuts)
                break
            cost, chosen_nut, chosen_spanner, nut_loc = best
            total_cost += cost
            # update man location to nut location after tightening
            if nut_loc is not None:
                man_loc = nut_loc
            # remove chosen nut and spanner
            remaining_nuts.remove(chosen_nut)
            del remaining_spanners[chosen_spanner]

        # If there remain nuts but no usable spanners, add a large penalty per nut
        if remaining_nuts and not remaining_spanners:
            total_cost += 1000 * len(remaining_nuts)

        # final integer heuristic
        return int(total_cost)