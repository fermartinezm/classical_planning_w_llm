from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob gate)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Summary
    -------
    This heuristic estimates the minimum number of actions required for the man to tighten all loose nuts.
    It takes into account the actions needed to:
      - Walk to a spanner's location (if not carrying a usable spanner),
      - Pick up a spanner,
      - Walk to a loose nut's location,
      - Tighten the nut with the spanner,
      - Repeat for all loose nuts.
    The heuristic uses the link graph to estimate shortest walking distances between locations.
    The heuristic is not admissible but aims to be informative and efficiently computable to minimize expanded nodes.

    Assumptions
    -----------
    - There is exactly one man.
    - Only one spanner can be carried at a time.
    - Each tighten_nut action consumes the usability of the spanner, so a new spanner must be picked up before tightening another nut.
    - All nuts start loose and must become tightened.
    - Walking costs one action per step along links.
    - Picking up a spanner and tightening a nut cost one action each.
    - The spanner usability must be restored by picking another usable spanner (i.e., pickup + tighten per nut).
      Carrying multiple spanners is not possible (the domain does not allow carrying more than one).

    Heuristic Initialization
    ------------------------
    - Parse static facts to build an undirected graph of linked locations.
    - Precompute shortest path distances between all locations of interest using BFS (since all links cost 1).
      Locations of interest include:
        * all locations from static facts,
        * all spanner initial locations,
        * all nut locations,
        * the initial location of the man.
    - Extract all nuts (loose nuts in the current state),
      all spanners that are usable and their locations,
      and man location and whether he is carrying a spanner.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state (set of facts):

    1. Check if all nuts are tightened (goal reached). If so, heuristic = 0.

    2. Extract:
       - the man's current location,
       - whether the man is carrying a usable spanner,
       - the set of loose nuts and their locations,
       - the set of usable spanners and their locations (those not carried).

    3. For each loose nut, compute the minimal walk distance from the man's location (or from next spanner location if pick-up needed) to the spanner location, then from spanner location to the nut location.

       Since the man can carry only one spanner at a time and each tighten_nut consumes usability, the heuristic sums over all nuts of:
           - (if not carrying usable spanner) minimal cost to walk to some usable spanner + 1 action to pickup spanner
           - walk to nut location + 1 action to tighten nut

       Because the man must pick a new spanner for each nut (assuming no spanner usability restoration), the heuristic approximates the total cost as:
           sum over nuts of (pick spanner + walk to spanner + walk nut + tighten)

    4. Since the heuristic must be efficiently computable, we take a greedy approach:
       - Estimate for each nut the minimal total cost of: 
         (if not carrying spanner: from current man location to spanner + pickup) + from spanner to nut + tighten
       - Then use a nearest-nut visiting approximation:
         * start at man_location,
         * for each nut:
            - if carrying usable spanner: directly go to nut and tighten
            - otherwise, go to closest spanner, pick it up, then go to nut and tighten
         * update man_location accordingly after each nut is tightened (the nut location)
         * after tightening nut, spanner is used up and no longer usable (lose carrying usable spanner status)
       - This simulates a chain of moving between nuts while always picking up spanners as needed.

    5. Output the sum of all involved actions and move steps as heuristic value.

    This heuristic does not guarantee admissibility, but it incorporates domain structure:
    walking costs among locations, usage of spanners, and the nuts to tighten.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all locations and build adjacency graph from (link l1 l2)
        self.adj = dict()
        locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.adj.setdefault(l1, set()).add(l2)
                self.adj.setdefault(l2, set()).add(l1)

        self.locations = locations

        # Precompute all-pairs shortest paths using BFS for efficiency only among known locations.
        # We compute a dictionary of dicts: distance[l1][l2] = shortest number of steps from l1 to l2
        self.distance = {}
        for loc in self.locations:
            self.distance[loc] = self._bfs_distances(loc)

    def _bfs_distances(self, start):
        """Compute shortest distances from start to all reachable locations using BFS."""
        distances = {start: 0}
        frontier = [start]
        while frontier:
            current = frontier.pop(0)
            current_dist = distances[current]
            for neighbor in self.adj.get(current, ()):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    frontier.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state

        # Quick goal check: If all tightened nuts are in state, heuristic = 0
        if self.goals <= state:
            return 0

        # Extract man, spanners, nuts status and locations from state
        # Find man location (there is exactly one man)
        man_location = None
        man_carrying_spanner = False
        loose_nuts = {}  # nut -> location
        usable_spanners = {}  # spanner -> location

        # Objects sets to identify types via naming conventions (optional)
        # Instead, we rely solely on facts patterns.

        # Since types are omitted here, we parse all facts:
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            # man location: (at man loc)
            if pred == "at":
                obj, loc = parts[1], parts[2]
                # we don't know man name, but from domain it's man type objects only,
                # assume man is the unique object with predicate 'carrying' in state or from task?
                # safer: find object used as "carrying ?m ?s" indicating man.
                # We'll find man name first: look for any carrying ?m ?spanner
                pass  # Defer man name retrieval below

        # Retrieve man name by scanning carrying facts:
        man_set = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "carrying":
                man_set.add(parts[1])
        man_name = None
        if len(man_set) == 1:
            man_name = next(iter(man_set))
        else:
            # If no carrying fact, try to find man in domain objects by at predicates 
            # assuming man is unique object that appears as first argument of carrying in domain
            # fallback: find any object of type man from the problem is not provided here,
            # so we fallback to find 'at ?m ?loc' where ?m is a single object of type man
            # Here, we take first 'at' predicate whose first arg is not spanner or nut
            # This might be fragile but no other option given current info
            # We'll just pick the first 'at' whose first argument is not spanner or nut name
            # but we don't have object types here, so as fallback pick the unique 'at' predicate where the first argument equals 'bob' or a known man name, else pick unique man if obvious.
            # Simplify: pick the single object that appears in carrying or multiple carrying is rare.
            # If no carrying at all, the man is not carrying spanner for now.

            # for now let's guess man_name is the object that appears in any (at ? ?), but exclude spanners and nuts.
            # From domain: man and spanner and nut are different types.
            # Check facts to get all spanners and nuts in state

            spanners = set()
            nuts = set()
            carrying_objs = set()
            for f in state:
                p = get_parts(f)
                if p[0] == "at":
                    o = p[1]
                    if o.startswith("spanner"):
                        spanners.add(o)
                    elif o.startswith("nut"):
                        nuts.add(o)
                elif p[0] == "carrying":
                    carrying_objs.add(p[1])  # man names

            # Pick man_name as object in (at ?loc) but not spanner or nut
            candidates = set()
            for f in state:
                p = get_parts(f)
                if p[0] == "at":
                    o = p[1]
                    if o not in spanners and o not in nuts:
                        candidates.add(o)
            if len(candidates) == 1:
                man_name = next(iter(candidates))
            elif carrying_objs:
                man_name = next(iter(carrying_objs))
            else:
                # worst case fallback: pick first candidate if any
                man_name = next(iter(candidates)) if candidates else None

        # Now find man location and whether carrying a usable spanner
        spanners_in_state = set()
        nuts_in_state = set()
        usable_spanners_set = set()
        carried_spanners = set()
        carrying_usablespanner = False

        for fact in state:
            parts = get_parts(fact)
            p = parts[0]

            if p == "at":
                obj, loc = parts[1], parts[2]
                if obj == man_name:
                    man_location = loc
                if obj.startswith("spanner"):
                    spanners_in_state.add(obj)
                if obj.startswith("nut"):
                    nuts_in_state.add(obj)

            elif p == "usable":
                s = parts[1]
                usable_spanners_set.add(s)
            elif p == "carrying":
                m, s = parts[1], parts[2]
                if m == man_name:
                    carried_spanners.add(s)
            elif p == "loose":
                n = parts[1]
                if n in nuts_in_state:
                    # Nut is currently loose
                    loose_nuts[n] = None  # location to fill later

        # Find locations of loose nuts:
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj in loose_nuts:
                    loose_nuts[obj] = loc

        # Find locations of usable spanners that are not carried:
        spanner_locations = dict()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj in usable_spanners_set:
                    spanner_locations[obj] = loc

        # Add usable spanners carried by man (usage will be consumed)
        # Check if the man is currently carrying a usable spanner
        carrying_usable_spanner = False
        for s in carried_spanners:
            if s in usable_spanners_set:
                carrying_usable_spanner = True
                # Remove this spanner from spanner_locations as it's carried
                spanner_locations.pop(s, None)

        # Safety if man_location is still None (should not happen)
        if man_location is None:
            # fallback: pick arbitrary location from links or 0 heuristic
            man_location = next(iter(self.locations)) if self.locations else "unknown"

        # If no loose nuts, heuristic is zero
        if not loose_nuts:
            return 0

        # Precompute distances helper to avoid missing keys
        def dist(a, b):
            if a not in self.distance or b not in self.distance[a]:
                return 1000  # large penalty if no path
            return self.distance[a][b]

        # Heuristic calculation:

        # Strategy:
        # We simulate a greedy tightening sequence:
        # Start with man location and whether carrying usable spanner.
        # For each nut (in any order), compute minimal cost to:
        # - If no usable spanner carried: walk to closest usable spanner and pick it up (pickup = 1 action)
        # - Walk to nut location from spanner (or current man location if carrying)
        # - Tighten nut (1 action)
        # Then update man_location to nut location,
        # losing usable spanner status after each tighten (spanner becomes unusable).
        #
        # Since finding optimal ordering is expensive, we use a simple greedy nearest-nut approach:
        # Pick the next nut that results in minimal incremental heuristic cost.

        unvisited_nuts = set(loose_nuts.keys())
        current_man_location = man_location
        have_usable_spanner = carrying_usable_spanner
        total_cost = 0

        while unvisited_nuts:
            best_nut = None
            best_cost = None
            best_spanner_loc = None  # For pickup if needed

            for nut in unvisited_nuts:
                nut_loc = loose_nuts[nut]
                if have_usable_spanner:
                    # Just walk to nut and tighten
                    cost = dist(current_man_location, nut_loc) + 1
                    # no pickup cost, spanner consumed hereafter
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_nut = nut
                        best_spanner_loc = None
                else:
                    # Need spanner pickup:
                    # Find closest usable spanner location from current_man_location
                    if not spanner_locations:
                        # no spanner available -> unsolvable state; assign large heuristic
                        # but must be finite, so large number
                        return 10000 + len(unvisited_nuts) * 10
                    min_spanner_dist = None
                    closest_spanner = None
                    for s, loc in spanner_locations.items():
                        d = dist(current_man_location, loc)
                        if min_spanner_dist is None or d < min_spanner_dist:
                            min_spanner_dist = d
                            closest_spanner = loc
                    # cost = walk to spanner + pickup (1) + walk to nut + tighten (1)
                    cost = min_spanner_dist + 1 + dist(closest_spanner, nut_loc) + 1
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_nut = nut
                        best_spanner_loc = closest_spanner

            # Apply best nut choice:
            if best_nut is None:
                # No available nuts? Break.
                break

            total_cost += best_cost
            current_man_location = loose_nuts[best_nut]
            unvisited_nuts.remove(best_nut)
            # After tightening a nut, usable spanner is consumed
            have_usable_spanner = False
            # Remove spanner location if it was used this step (picked up)
            if best_spanner_loc is not None:
                # Remove spanner(s) at this location from available spanners
                to_remove = []
                for s, loc in spanner_locations.items():
                    if loc == best_spanner_loc:
                        to_remove.append(s)
                for s in to_remove:
                    spanner_locations.pop(s)

        return int(total_cost)
