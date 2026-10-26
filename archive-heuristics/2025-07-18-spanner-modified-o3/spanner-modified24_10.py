from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Split a fact string '(p a b …)' into its individual components.
    """
    return fact[1:-1].split()


class SpannerModifiedHeuristic(Heuristic):
    """
    Summary
    -------
    Estimates the number of actions still required to reach a state
    in which every nut is tightened.  The estimate is domain–specific
    for the *spanner-modified* domain and is tuned for greedy best-first
    search – it is *not* meant to be admissible.

    Assumptions
    -----------
    1. There is exactly one man (as in all benchmark instances).
    2. A *usable* spanner can tighten an arbitrary number of nuts,
       hence the planner only needs *one* of them.
    3. The undirected road map is given by facts of the form
       ``(link l1 l2)`` in the *static* section.
    4. All actions have unit cost.  Therefore the number of actions is
       a suitable surrogate for plan length.

    Heuristic Initialization
    ------------------------
    During construction the heuristic:
    •	Extracts the set of *usable* spanners, all nuts and the single man
      from the *initial state*.
    •	Builds an undirected adjacency list from the static ``link`` facts.
    •	Pre-computes nothing else – shortest paths are obtained on demand
      with a lightweight breadth-first search (the maps are tiny).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a state *s* the heuristic value *h(s)* is computed as follows:

    1.	Locate the man M and all still *loose* nuts.
    2.	If there are no loose nuts the goal is reached ⇒ *h(s)=0*.
    3.	If M is **already carrying** a usable spanner, no pick-up is needed.
       Otherwise:
       a.	For every usable spanner on the ground determine the shortest
          walking distance d(M, S).
       b.	The cheapest of those distances plus one action for
          ``pickup_spanner`` is the *spanner cost*.
       c.	After the pick-up M stands at the chosen spanner’s location.
    4.	Ignore the exact order in which nuts are visited and approximate
       the remaining travelling effort by the weight of a *minimum
       spanning tree* (MST) over the set
       { current-location ∪ locations(loose-nuts) } with edge weights
       equal to shortest walking distances.
       The MST gives a good admissible estimate of the length of a path
       that visits all nuts once.
    5.	Add one action for every nut that still has to be tightened
       (the ``tighten_nut`` action itself).
    6.	The heuristic value is

           spanner_cost + MST_cost + |loose_nuts|.

    The result is 0 for goal states, finite for any solvable state and
    cheap to compute.
    """

    # -------------------------------- Construction ---------------------------

    def __init__(self, task):
        """
        Extract static information (map and object categories) from the task.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial = task.initial_state

        # --- identify objects by scanning the initial state ------------------
        self.usable_spanners = set()
        self.nuts = set()
        for fact in self.initial:
            pred, *args = get_parts(fact)
            if pred == "usable":
                self.usable_spanners.add(args[0])
            elif pred in ("loose", "tightened"):
                self.nuts.add(args[0])

        # Any locatable that is neither spanner nor nut must be the man
        candidates = {
            get_parts(f)[0]
            for f in self.initial
            if get_parts(f)[0] not in self.usable_spanners
            and get_parts(f)[0] not in self.nuts
            and get_parts(f)[0] != "link"
        }
        # Instances contain exactly one man
        self.man = next(iter(candidates))

        # --- build undirected road map --------------------------------------
        self.adj = defaultdict(list)
        for fact in self.static:
            pred, l1, l2 = get_parts(fact)
            if pred == "link":
                self.adj[l1].append(l2)
                self.adj[l2].append(l1)  # treat links as undirected

    # ---------------------------- helper methods -----------------------------

    def _bfs_distances(self, start):
        """
        Breadth-first search returning shortest path lengths from *start*
        to every reachable location.
        """
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in self.adj[cur]:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return dist

    @staticmethod
    def _mst_length(nodes, distance_lookup):
        """
        Prim’s algorithm on the complete graph induced by *nodes* using
        distances from *distance_lookup[u][v]*.  Returns the MST weight.
        """
        if len(nodes) <= 1:
            return 0

        visited = {nodes[0]}
        total = 0
        while len(visited) < len(nodes):
            best_d = float("inf")
            best_v = None
            for u in visited:
                for v in nodes:
                    if v in visited:
                        continue
                    d = distance_lookup[u][v]
                    if d < best_d:
                        best_d, best_v = d, v
            visited.add(best_v)
            total += best_d
        return total

    # -------------------------------- heuristic ------------------------------

    def __call__(self, node):
        state = node.state

        # ----------- collect dynamic information available in the state ------
        man_location = None
        spanners_on_ground = []  # list of (spanner, location)
        carrying_spanner = False

        loose_nuts = set()
        nut_locations = dict()

        for fact in state:
            pred, *args = get_parts(fact)

            if pred == "at":
                obj, loc = args
                if obj == self.man:
                    man_location = loc
                elif obj in self.usable_spanners:
                    spanners_on_ground.append((obj, loc))
                elif obj in self.nuts:
                    nut_locations[obj] = loc
            elif pred == "carrying" and args[0] == self.man:
                sp = args[1]
                if sp in self.usable_spanners:
                    carrying_spanner = True
            elif pred == "loose":
                loose_nuts.add(args[0])

        # Goal test: all nuts tightened
        if not loose_nuts:
            return 0

        # Pre-compute distances from every location of interest on-demand
        interesting_locations = {man_location}
        interesting_locations.update(loc for _, loc in spanners_on_ground)
        interesting_locations.update(
            nut_locations[n] for n in loose_nuts if n in nut_locations
        )

        distance_lookup = {
            loc: self._bfs_distances(loc) for loc in interesting_locations
        }

        # -------------------------- cost of obtaining a spanner --------------
        spanner_cost = 0
        current_location_after_pickup = man_location

        if not carrying_spanner:
            best_cost = float("inf")
            best_loc = None
            for _, sp_loc in spanners_on_ground:
                d = distance_lookup[man_location].get(sp_loc, float("inf"))
                if d < best_cost:
                    best_cost = d
                    best_loc = sp_loc
            # The problem instances always contain a reachable spanner
            spanner_cost = best_cost + 1  # +1 for the pickup action
            current_location_after_pickup = best_loc

        # ------------------ travelling + tightening remaining nuts -----------
        nut_locs = [
            nut_locations[n] for n in loose_nuts if n in nut_locations
        ]
        # In solvable instances every loose nut has a known location
        nodes = [current_location_after_pickup] + nut_locs
        mst_cost = self._mst_length(nodes, distance_lookup)

        tighten_actions = len(loose_nuts)

        return spanner_cost + mst_cost + tighten_actions
