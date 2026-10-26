from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


# ---------------------------------------------------------------------------
# Small helpers for dealing with PDDL facts that are represented as strings.
# ---------------------------------------------------------------------------

def get_parts(fact):
    """
    Return the fact split into tokens, without the surrounding brackets.

    Example
    -------
    "(at bob shed)"   ->  ["at", "bob", "shed"]
    """
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Test whether a fact matches the given pattern.  Wildcards ``*`` are allowed.

    Example
    -------
    match("(at bob shed)", "at", "bob", "*")        ->  True
    match("(loose nut1)",   "loose", "*")           ->  True
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


# ---------------------------------------------------------------------------
# Domain–dependent heuristic for the modified Spanner domain
# ---------------------------------------------------------------------------

class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Greedy distance‐based estimate for the *spanner-modified* domain.  The
    heuristic assumes that a (single) mechanic first acquires any *usable*
    spanner and then visits every location that still contains a *loose* nut,
    tightening each nut once it arrives.  The cost returned is the number of
    operator applications that a straightforward greedy strategy would need.

    Assumptions
    -----------
    • Only one mechanic (called *bob* in the benchmark problems) exists.  
    • Walking between two directly linked locations costs one action.
    • Carrying a spanner never prevents the mechanic from carrying additional
      spanners; picking up the first usable spanner therefore suffices.
    • All *link* relations are undirected; i.e. the mechanic can walk in both
      directions.
    • Every *tighten_nut* action is independent once the mechanic and a usable
      spanner are co-located with the nut.

    Heuristic Initialization
    ------------------------
    1.   Extract the unique mechanic, all spanners and all nuts from the
         initial state supplied in the task description.
    2.   Build an undirected graph from the static *link* facts.
    3.   Pre-compute and cache the shortest path length between every pair of
         locations by breadth-first search.  These distances are reused for
         every subsequent heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node with state *s*:

    1.  If all goal facts are already satisfied, return **0**.
    2.  Determine the mechanic’s current location *loc_bob*.
    3.  Check whether the mechanic already carries a (usable) spanner.
        • If **yes**, the current position stays *loc_bob*.
        • If **no**, choose the usable spanner that is *closest* to
          *loc_bob*.
          – Add the walking distance to that spanner.
          – Add **1** for the *pickup_spanner* action.
          – Set the current position to the spanner’s location.
    4.  Collect every nut that is still *loose* together with its physical
        location.
    5.  Starting from the current position, repeatedly visit the nearest
        still-loose nut (“nearest neighbour” tour):
        • Add the shortest path length to the nut’s location (walk actions).
        • Add **1** for the *tighten_nut* action.
        • Update the current position to the nut’s location and continue
          until no loose nuts remain.
    6.  The sum of all action counts produced in steps 3–5 is the heuristic
        value that is returned.

    The heuristic is *inadmissible* (it may over-estimate) but inexpensive to
    compute while providing useful guidance: it accounts for the real road
    network, the necessity of acquiring a spanner, and the number as well as
    spatial distribution of still-loose nuts.
    """

    # ---------------------------------------------------------------------
    # Construction utilities
    # ---------------------------------------------------------------------

    def __init__(self, task):
        """Pre-compute static information that never changes during search."""
        # --- Identify mechanic, spanners and nuts --------------------------------
        initial = task.initial_state
        self.spanners = {get_parts(f)[1] for f in initial if match(f, "usable", "*")}
        self.nuts     = {get_parts(f)[1] for f in initial if match(f, "loose", "*")}

        # try to find the (unique) mechanic
        carriers = [get_parts(f)[1] for f in initial if match(f, "carrying", "*", "*")]
        if carriers:
            self.mechanic = carriers[0]
        else:
            # no carrying fact in the initial state – fall back to 'at' facts
            candidates = [get_parts(f)[1]
                          for f in initial
                          if match(f, "at", "*", "*")
                          and get_parts(f)[1] not in self.spanners
                          and get_parts(f)[1] not in self.nuts]
            self.mechanic = candidates[0]  # assume at least one candidate exists

        # --- Build undirected graph of locations ---------------------------------
        self.adj = {}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.adj.setdefault(l1, set()).add(l2)
                self.adj.setdefault(l2, set()).add(l1)

        # Ensure that locations that appear only in *at* facts are still registered
        for fact in initial:
            if match(fact, "at", "*", "*"):
                loc = get_parts(fact)[2]
                self.adj.setdefault(loc, set())

        # --- Pre-compute all-pairs shortest path lengths -------------------------
        self.dist = {}                                   # (l1, l2) -> int
        for src in self.adj:
            self._bfs_fill(src)

        # Store goals for the fast goal-test inside __call__
        self.goals = task.goals

    # ---------------------------------------------------------------------
    # Helper: breadth-first search for distances from a single source
    # ---------------------------------------------------------------------

    def _bfs_fill(self, source):
        """Fill distance table entries for *source* → every other location."""
        queue = deque([(source, 0)])
        visited = {source}
        while queue:
            loc, d = queue.popleft()
            self.dist[(source, loc)] = d
            self.dist[(loc, source)] = d               # symmetry
            for nb in self.adj[loc]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, d + 1))

    def _distance(self, a, b):
        """Return the cached shortest walking distance between a and b."""
        if a == b:
            return 0
        # Both locations are assumed to exist; unreachable pairs never occur
        return self.dist.get((a, b), float("inf"))

    # ---------------------------------------------------------------------
    # Main heuristic function
    # ---------------------------------------------------------------------

    def __call__(self, node):
        state = node.state

        # --- Early exit: goal already satisfied ----------------------------------
        if self.goals <= state:
            return 0

        # --- Current mechanic position ------------------------------------------
        bob_loc = next(get_parts(f)[2] for f in state
                       if match(f, "at", self.mechanic, "*"))

        # --- Does the mechanic already carry a usable spanner? -------------------
        has_spanner = any(match(f, "carrying", self.mechanic, "*") for f in state)

        cost = 0
        current_loc = bob_loc

        if not has_spanner:
            # Gather all *usable* spanners that are lying around
            candidates = [(get_parts(f)[1], get_parts(f)[2])
                          for f in state
                          if match(f, "at", "*", "*")
                          and get_parts(f)[1] in self.spanners]
            if candidates:
                # Choose the spanner with minimal walking distance
                spanner, sp_loc = min(candidates,
                                      key=lambda p: self._distance(bob_loc, p[1]))
                cost += self._distance(bob_loc, sp_loc)   # walk to spanner
                cost += 1                                 # pickup_spanner
                current_loc = sp_loc
            else:
                # Should not happen in solvable instances; keep heuristic finite
                current_loc = bob_loc

        # --- Collect all still-loose nuts with their locations -------------------
        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                # find the nut's location
                loc = next(get_parts(f2)[2] for f2 in state
                           if match(f2, "at", nut, "*"))
                loose_nuts.append(loc)

        # --- Greedy tour: always go to nearest un-tightened nut ------------------
        while loose_nuts:
            # pick the nut closest to the current location
            idx, dist = min(enumerate(loose_nuts),
                            key=lambda pair: self._distance(current_loc, pair[1]))
            target_loc = loose_nuts.pop(idx)
            cost += dist                               # walk distance
            cost += 1                                  # tighten_nut
            current_loc = target_loc

        return cost
