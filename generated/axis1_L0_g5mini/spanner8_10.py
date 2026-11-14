from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the parts of a fact string '(pred a b ...)' as a list ['pred','a','b',...]."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic estimating the number of actions required to tighten all remaining loose nuts
    in the spanner domain. An action sequence to tighten a nut requires bringing a usable
    spanner to the nut and performing the tighten_nut action. The heuristic estimates the
    cost (in number of STRIPS actions) by greedily assigning available usable spanners to
    loose nuts in an order that minimizes incremental cost from the current man's position.

    Assumptions
    -----------
    - There is at least one man (typical instances have a single man).
    - Walking along a single 'link' fact costs 1 action (each walk is one action).
    - Picking up a spanner (pickup_spanner) costs 1 action.
    - Tightening a nut (tighten_nut) costs 1 action.
    - A usable spanner is consumed (becomes not usable) after tightening a nut.
    - The planner representation uses facts as strings like '(at bob shed)'.
    - Links are directed as given in static facts (we use directed BFS).
    - If portions of the graph are unreachable, the heuristic returns a large finite value
      (not infinity) so the heuristic remains finite in solvable states and bounded otherwise.

    Heuristic Initialization
    ------------------------
    - self.goals: stored set of goal facts (typically '(tightened nutX)', etc.).
    - static facts are parsed and stored:
      - self.locations: set of location names extracted from (link ...) static facts.
      - self.adj: adjacency dict mapping location -> list of successor locations for walk.
    - No further heavy preprocessing is performed.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goal facts are already satisfied in the current state, return 0.
    2. Identify which nuts still need tightening by checking goal tightened facts missing from state.
       For each such nut, obtain its current location from a state fact '(at nut location)'.
    3. Identify all usable spanners from state facts '(usable s)'. For each usable spanner determine
       whether it is currently carried by the man (a '(carrying man s)' fact) or lying at some
       location '(at s loc)'.
    4. Locate the man: prefer a subject of any '(carrying ...)' fact; if none exist, find a subject
       of an '(at X L)' fact that is not a location, not a known spanner and not a known nut.
       If the man cannot be robustly determined, the heuristic falls back to a large estimate.
    5. Greedy assignment loop:
       - While there remain loose nuts, compute for every (nut, usable-spanner) pairing the
         incremental action cost to (a) move the man from his current location to the spanner (if not
         already carried), (b) pick up the spanner (if required), (c) move from the spanner location to
         the nut, and (d) perform the tighten action.
       - Movement cost is computed via BFS shortest-path distances on the directed link graph.
       - Choose the nut-spanner pair with the minimal incremental cost, apply that assignment:
         update the man's location to the nut's location, remove the spanner from the available list
         (it becomes unusable), and add the incremental cost to the running heuristic total.
       - If any required move is impossible (no path in the static link graph), that pairing is
         considered infeasible.
    6. If at least one feasible assignment exists per remaining nut, the sum of incremental costs is
       returned. If no feasible assignments exist for remaining nuts (disconnected graph), a large
       finite estimate is returned.
    """

    def __init__(self, task):
        self.goals = task.goals
        static = task.static

        # Extract locations and build directed adjacency from (link a b) static facts.
        self.locations = set()
        self.adj = {}
        for fact in static:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) >= 3:
                a, b = parts[1], parts[2]
                self.locations.add(a)
                self.locations.add(b)
                if a not in self.adj:
                    self.adj[a] = []
                self.adj[a].append(b)

        # Ensure every location key exists in adjacency dictionary
        for loc in list(self.locations):
            self.adj.setdefault(loc, [])

        # A large finite value used when distances are unreachable.
        self._BIG = 10 ** 6

    def _bfs_distance(self, src, dst):
        """Return shortest directed distance (number of walk actions) from src to dst using self.adj.
           If src==dst return 0. If either node not in graph or dst unreachable return a very large number.
        """
        if src == dst:
            return 0
        if src not in self.adj:
            return self._BIG
        visited = {src}
        q = deque([(src, 0)])
        while q:
            node, d = q.popleft()
            for succ in self.adj.get(node, []):
                if succ == dst:
                    return d + 1
                if succ not in visited:
                    visited.add(succ)
                    q.append((succ, d + 1))
        return self._BIG

    def __call__(self, node):
        state = node.state

        # Quick goal check
        if self.goals <= state:
            return 0

        # Extract spanners (usable), nuts (loose/tightened), carrying facts, and locations of objects.
        usable_spanners = set()
        tightened_nuts = set()
        loose_nuts = set()
        carrying_facts = set()
        at_facts = {}  # maps object -> location (for objects that have at facts)
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "usable" and len(parts) >= 2:
                usable_spanners.add(parts[1])
            elif pred == "tightened" and len(parts) >= 2:
                tightened_nuts.add(parts[1])
            elif pred == "loose" and len(parts) >= 2:
                loose_nuts.add(parts[1])
            elif pred == "carrying" and len(parts) >= 3:
                carrying_facts.add((parts[1], parts[2]))  # (man, spanner)
            elif pred == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                at_facts[obj] = loc

        # Identify nuts that still need tightening (based on goal facts).
        nuts_to_tighten = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "tightened" and len(parts) >= 2:
                nut = parts[1]
                if g not in state:  # not yet tightened
                    # determine nut location
                    nut_loc = at_facts.get(nut)
                    if nut_loc is None:
                        # If the nut is not at any location in the state we cannot reason well;
                        # treat as unreachable (large heuristic).
                        return self._BIG
                    nuts_to_tighten.append((nut, nut_loc))

        # Build list of available usable spanners with their current location:
        # If a usable spanner is carried by the man, mark its location as 'carried' (special).
        # Otherwise, get its location from at_facts.
        available_spanners = {}
        # Determine man name(s). Prefer subjects of carrying facts; else heuristically detect from at facts.
        man_name = None
        if carrying_facts:
            # get first carrying fact's man
            man_name = next(iter(carrying_facts))[0]
        else:
            # heuristic: find a subject of an (at X L) fact where X is not a location
            # and not a known spanner or nut.
            subjects = set()
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == "at" and len(parts) >= 3:
                    subjects.add(parts[1])
            # remove known locations, spanners, nuts
            candidates = subjects - self.locations - set(usable_spanners) - loose_nuts - tightened_nuts
            if candidates:
                man_name = next(iter(candidates))

        if man_name is None:
            # Could not identify man robustly; return large but finite heuristic.
            return self._BIG

        # Populate available_spanners mapping: spanner -> ('carried') or location
        for s in usable_spanners:
            # check if carried
            if (man_name, s) in carrying_facts:
                available_spanners[s] = "carried"
            else:
                loc = at_facts.get(s)
                if loc is not None:
                    available_spanners[s] = loc
                else:
                    # spanner usable but neither carried nor at any location in the state -> ignore
                    # (conservative: cannot use it)
                    continue

        # If there are fewer usable spanners than nuts, it's still possible that some spanners are currently not marked usable
        # but can become usable? In this domain tighten_nut makes spanner not usable and no action makes it usable again,
        # so number of usable spanners must be >= nuts to tighten them. If fewer, the problem is unsolvable; return large.
        if len(available_spanners) < len(nuts_to_tighten):
            # There might be additional spanners that are not marked usable now, but they cannot be used.
            # Conservative estimate: still attempt with what we have; if insufficient, return large.
            # If no usable spanners at all and nuts remain, impossible.
            return self._BIG

        # Start greedy assignment:
        total_cost = 0
        current_man_loc = at_facts.get(man_name)
        if current_man_loc is None:
            # Man location unknown -> conservative large estimate
            return self._BIG

        # Convert spanner inventory to mutable dict copy
        spanner_locations = dict(available_spanners)  # spanner -> 'carried' or loc

        remaining_nuts = list(nuts_to_tighten)

        while remaining_nuts:
            best_cost = None
            best_choice = None  # tuple (nut_index, spanner)
            best_nut_loc = None

            # For each nut, find the cheapest spanner to use
            for ni, (nut, nut_loc) in enumerate(remaining_nuts):
                for s, sloc in spanner_locations.items():
                    # Compute cost to use spanner s for nut at nut_loc from current_man_loc
                    if sloc == "carried":
                        # No pickup needed; just walk to nut and tighten
                        walk_cost = self._bfs_distance(current_man_loc, nut_loc)
                        if walk_cost >= self._BIG:
                            cost = self._BIG
                        else:
                            cost = walk_cost + 1  # +1 tighten
                    else:
                        # Need to walk to spanner location, pickup, walk to nut, tighten
                        d1 = self._bfs_distance(current_man_loc, sloc)
                        d2 = self._bfs_distance(sloc, nut_loc)
                        if d1 >= self._BIG or d2 >= self._BIG:
                            cost = self._BIG
                        else:
                            cost = d1 + 1 + d2 + 1  # walk to spanner + pickup + walk to nut + tighten

                    # Track best
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_choice = (ni, s)
                        best_nut_loc = nut_loc

            if best_choice is None or best_cost is None or best_cost >= self._BIG:
                # No feasible assignment for remaining nuts (disconnected). Return large.
                return self._BIG

            # Apply best assignment
            nut_index, chosen_spanner = best_choice
            total_cost += best_cost

            # Update man's location to the nut's location (after performing tighten)
            current_man_loc = best_nut_loc

            # Remove chosen spanner from available set (it becomes not usable)
            del spanner_locations[chosen_spanner]

            # Remove the assigned nut from remaining list
            remaining_nuts.pop(nut_index)

            # Also, any remaining spanners recorded at 'carried' remain 'carried' (man may still carry others).
            # However, if a spanner was recorded at some fixed location but the man moved past it earlier (and
            # might pick it up later), that will be accounted for in the bfs distances in subsequent iterations.

        return int(total_cost)