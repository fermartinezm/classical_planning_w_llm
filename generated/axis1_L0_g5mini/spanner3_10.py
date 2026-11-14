from heuristics.heuristic_base import Heuristic
from collections import deque, defaultdict

INF = 10**6


def get_parts(fact):
    """Return the tokens inside a PDDL fact string, e.g. "(at bob shed)" -> ["at","bob","shed"]."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the "spanner" domain. It estimates the number
    of actions required to tighten all loose nuts in the current state. The estimate
    attempts to model the real cost structure of the domain (walks, pickup_spanner
    actions, and tighten_nut actions) while remaining efficiently computable.

    Assumptions
    -----------
    - The planner's internal state encodes facts as strings like "(predicate arg1 arg2)".
    - Locations are the nodes that appear in static "link" facts. Links are treated
      as directed edges (matching the domain's walk precondition).
    - A spanner is usable iff the fact "(usable SPANNER)" is present. Spanners that
      are carried by the man are represented by "(carrying MAN SPANNER)" facts and
      remain usable only if "(usable SPANNER)" also appears in the state.
    - There may be multiple spanners carried by the man (the domain allows multiple
      carrying facts for the same man).
    - Tightening a nut consumes one usable spanner (the spanner becomes not usable).
    - There is typically a single man; the heuristic finds the man by identifying
      the object that appears in an "(at MAN LOCATION)" fact but is not itself a
      location, spanner, or nut.

    Heuristic Initialization
    ------------------------
    - Extracts the directed adjacency graph of locations from static "link" facts.
    - Precomputes (via BFS from each location) the shortest directed distance (in
      number of walk actions) between all pairs of locations. Distances that do not
      exist remain effectively infinite (INF).
    - Stores the goal facts to quickly detect goal states.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    The heuristic simulates a simple greedy plan for tightening all loose nuts.
    It keeps track of:
      - Where the man currently is.
      - Which usable spanners exist and where they are (either at some location or carried).
      - Which nuts remain loose and where they are.

    The simulated plan iteratively selects a single nut to tighten next using a
    cheap greedy choice:
      1. If the man currently carries a usable spanner, one option is to walk from
         the man's location to the nut and tighten it (cost = walk_distance + 1).
      2. Another option is to walk to a location that contains one or more usable
         spanners, pick them up (1 action per pickup), walk to the nut, and tighten
         it. When the heuristic chooses to visit a spanner location it assumes the
         man will pick up all usable spanners at that location (this is realistic
         because picking extra spanners while already there avoids future travel).
         The cost for visiting a spanner location L to serve a nut at N is thus:
             dist(man, L) + pickups_at_L + dist(L, N) + 1 (tighten)
         where pickups_at_L is the number of usable spanners at L picked up.
      3. Among all these options (using a carried spanner or using a spanner from
         any location), the heuristic chooses the option with minimum incremental
         cost. It then updates:
         - man location = nut location (we assume the man ends up at the nut after tightening),
         - consumes one usable spanner (the one used becomes unusable),
         - if the man visited a spanner location, all usable spanners at that
           location are added to the carried pool (and counted as picked up).
      4. Repeat until all loose nuts are accounted for or until it becomes clear
         that some nuts are unreachable (in which case the heuristic returns a
         large finite value INF).

    Rationale and properties
    ------------------------
    - The heuristic models the primary costs (walks, pickups, tightens) and the
      resource constraint that usable spanners are consumed by tighten actions.
    - It is greedy and not admissible in general, but tends to give realistic
      estimates that reduce node expansions in best-first search by preferring
      spanner collections that avoid repetitive travel.
    - The heuristic is efficiently computable: distances are precomputed once and
      each heuristic invocation performs a small simulation that is polynomial in
      the number of nuts and spanners.
    """

    def __init__(self, task):
        # Store goal facts and static facts
        self.goals = task.goals
        self.static = task.static

        # Build directed adjacency graph of locations from static link facts.
        self.adj = defaultdict(list)
        self.locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.adj[src].append(dst)
                self.locations.add(src)
                self.locations.add(dst)

        # Precompute shortest directed distances between all locations using BFS.
        # distances[src][dst] = number of walk actions from src to dst, INF if unreachable.
        self.distances = {}
        for loc in self.locations:
            dist = {loc: 0}
            q = deque([loc])
            while q:
                u = q.popleft()
                for v in self.adj.get(u, []):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            # For convenience, make unreachable distances INF
            self.distances[loc] = {v: dist.get(v, INF) for v in self.locations}

    def _dist(self, a, b):
        """Return walk distance from location a to b (INF if unknown or unreachable)."""
        if a is None or b is None:
            return INF
        if a not in self.distances:
            return INF
        return self.distances[a].get(b, INF)

    def __call__(self, node):
        state = node.state

        # 1) If goals are already reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse state facts
        # Identify nuts (those that appear in 'loose' or 'tightened' predicates)
        nuts = set()
        loose_nuts = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] in ("loose", "tightened") and len(parts) >= 2:
                nuts.add(parts[1])
            if parts[0] == "loose" and len(parts) >= 2:
                loose_nuts.add(parts[1])

        # Identify usable spanners (explicit usable facts)
        usable_spanners = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "usable" and len(parts) >= 2:
                usable_spanners.add(parts[1])

        # Map spanner -> whether it is currently carried and if so by which man
        spanner_carried_by = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "carrying" and len(parts) >= 3:
                man_nm, sp_nm = parts[1], parts[2]
                spanner_carried_by[sp_nm] = man_nm

        # Map object locations for 'at' facts (subject -> location)
        obj_location = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) >= 3:
                subj, loc = parts[1], parts[2]
                obj_location[subj] = loc

        # Determine the man's name and location.
        # Heuristic assumes one man. The man is an object that appears in an (at X L)
        # fact where X is not a location and not a spanner nor a nut.
        man_name = None
        man_loc = None
        # Candidate man names: any subject of an at fact that is not a location token.
        for subj, loc in obj_location.items():
            if loc in self.locations and subj not in self.locations and subj not in nuts and subj not in usable_spanners:
                # If subj also appears as a carrier in carrying facts, it's a strong candidate.
                if subj in (spanner_carried_by.get(s) for s in spanner_carried_by):
                    man_name = subj
                    man_loc = loc
                    break
                # otherwise keep as candidate but prefer one that is carrier if available
                if man_name is None:
                    man_name = subj
                    man_loc = loc

        # If we couldn't find man via the heuristic above but there is a carrying fact,
        # extract the man from the carrying facts.
        if man_name is None and spanner_carried_by:
            # pick any carrying entry
            any_sp, any_man = next(iter(spanner_carried_by.items()))
            man_name = any_man
            man_loc = obj_location.get(man_name, None)

        # If still unknown, try to find any 'at' fact whose subject is not a location token.
        if man_name is None:
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "at" and len(parts) >= 3:
                    subj, loc = parts[1], parts[2]
                    if loc in self.locations:
                        man_name = subj
                        man_loc = loc
                        break

        # If man location is still unknown, we cannot make distance computations -> return INF.
        if man_loc is None:
            return INF

        # Build a mapping spanner -> current location for usable spanners.
        # If a usable spanner is carried by the man, its effective location is the man's location.
        spanner_loc = {}
        for s in usable_spanners:
            if s in spanner_carried_by:
                # carried by someone; assume same man (or treat as carried)
                carrier = spanner_carried_by[s]
                # use carrier location if known, otherwise unknown
                loc = obj_location.get(carrier, man_loc)
                spanner_loc[s] = loc
            else:
                # look for an at fact for the spanner
                loc = obj_location.get(s, None)
                if loc is not None:
                    spanner_loc[s] = loc
                else:
                    # spanner usable but no location info -> treat as unreachable
                    spanner_loc[s] = None

        # Build mapping location -> list of spanners present (usable ones only)
        spanners_at_location = defaultdict(list)
        carried_usable = []  # list of spanners currently carried by the man and usable
        for s, loc in spanner_loc.items():
            if loc is None:
                continue
            if s in spanner_carried_by:
                # If carried by our man, add to carried; if carried by someone else treat as not available
                if spanner_carried_by[s] == man_name:
                    carried_usable.append(s)
                else:
                    # Carried by other agent: effectively unreachable
                    continue
            else:
                spanners_at_location[loc].append(s)

        # Build map nut -> location for loose nuts
        loose_nut_locations = {}
        for n in loose_nuts:
            loc = obj_location.get(n, None)
            loose_nut_locations[n] = loc

        # If there are no usable spanners anywhere and none carried, impossible to tighten more nuts.
        if not carried_usable and not spanners_at_location:
            # If there are no loose nuts, 0 handled earlier. Otherwise return large finite number.
            return INF

        # Greedy simulation: repeatedly pick the next nut to tighten with minimum incremental cost.
        remaining_nuts = set(loose_nuts)
        cost = 0
        # convert spanners_at_location values to counts and sets we will mutate
        spanners_at_location = {loc: list(sp_list) for loc, sp_list in spanners_at_location.items()}
        carried_count = len(carried_usable)

        # Heuristic will loop at most number of loose nuts times.
        while remaining_nuts:
            best_option = None
            best_cost = INF
            best_choice = None  # tuple describing how to execute: ("use_carried", nut) or ("visit_loc", loc, nut)

            # For each nut compute cost if we use a carried spanner (if available)
            for nut in remaining_nuts:
                nut_loc = loose_nut_locations.get(nut, None)
                if nut_loc is None:
                    # unknown nut location -> unreachable
                    continue
                # Option: use carried spanner
                if carried_count > 0:
                    d = self._dist(man_loc, nut_loc)
                    this_cost = d + 1  # walk + tighten
                    if this_cost < best_cost:
                        best_cost = this_cost
                        best_choice = ("use_carried", None, nut)

                # Option: go to some location L with usable spanners
                for L, sp_list in spanners_at_location.items():
                    if not sp_list:
                        continue
                    d1 = self._dist(man_loc, L)
                    d2 = self._dist(L, nut_loc)
                    if d1 >= INF or d2 >= INF:
                        continue
                    pickups_at_L = len(sp_list)
                    # We choose to pick up all usable spanners at L when we visit it (reasonable optimistic model).
                    this_cost = d1 + pickups_at_L + d2 + 1  # move to L, pickups, move to nut, tighten
                    if this_cost < best_cost:
                        best_cost = this_cost
                        best_choice = ("visit_loc", L, nut)

            # If no option found (e.g., unreachable), return INF
            if best_choice is None or best_cost >= INF:
                return INF

            # Apply the chosen option to our simulated state
            action_type, loc_chosen, chosen_nut = best_choice
            cost += best_cost
            # After tightening, the man ends up at the nut location
            man_loc = loose_nut_locations[chosen_nut]
            # consume one usable spanner
            if action_type == "use_carried":
                carried_count -= 1
            elif action_type == "visit_loc":
                L = loc_chosen
                # we picked up all spanners at L
                pickups = spanners_at_location.get(L, [])
                num_picked = len(pickups)
                carried_count += num_picked
                # clear that location's spanner list (they were picked up)
                spanners_at_location[L] = []
                # then one of them is used immediately
                if carried_count > 0:
                    carried_count -= 1
                else:
                    # this should not happen, but guard just in case
                    return INF
            else:
                return INF

            # Mark chosen nut as tightened in our simulation
            remaining_nuts.remove(chosen_nut)

            # Note: we do not need to explicitly track which spanner became unusable by name,
            # since we only model counts of usable spanners (we removed one per tightened nut).

            # If at some point we run out of usable spanners and no remaining spanners_at_location,
            # and still have remaining nuts, fail with INF
            any_spanners_left = carried_count > 0 or any(spanners_at_location.get(L, []) for L in spanners_at_location)
            if not any_spanners_left and remaining_nuts:
                return INF

        # The accumulated cost is our heuristic estimate
        # Ensure finite integer return
        return int(cost)