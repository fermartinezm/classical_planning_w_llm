from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact by removing outer parentheses and splitting on spaces.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a pattern allowing wildcards '*'.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Summary
    -------
    This heuristic estimates the number of actions required to tighten all nuts in the goal 
    starting from the current state. It focuses on the minimum number of walks, spanner pickups, 
    and tighten operations needed, assuming the man can only carry one spanner at a time and 
    spanners become unusable after tightening a nut.

    Assumptions
    -----------
    - Exactly one man exists (e.g., 'bob').
    - Multiple spanners and nuts can exist.
    - Each nut must be tightened exactly once.
    - To tighten a nut, the man must:
      (a) be at the nut's location,
      (b) carry a usable spanner,
      (c) nut must be loose.
    - Walking costs one action per link traversal.
    - Picking up a spanner costs one action.
    - Tightening a nut costs one action.
    - Spanners can only be used once before becoming unusable; after that, a new usable spanner must be picked up.

    Heuristic Initialization
    ------------------------
    - Extract:
      * The goal nuts that must be tightened.
      * Static 'link' facts as an undirected graph for shortest path calculation.
      * Current facts on:
        - Location of the man.
        - Location and usability of spanners.
        - Location and looseness of nuts.
    - Precompute shortest paths between all locations using BFS.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all nuts in the goal that are not yet tightened (i.e., currently loose).
    2. Identify:
       - The man's current location.
       - Locations of all currently usable spanners.
    3. For each loose nut:
       - Compute shortest path distances:
         a) From man's location (or last nut location) to a reachable usable spanner.
         b) From spanner location to nut location.
       - If the man currently carries a usable spanner, skip the spanner pickup step for the first nut.
    4. Assign a cost estimate for tightening each nut as:
       - Travel steps to pickup spanner (if needed) + 1 pickup action.
       - Travel steps to nut location + 1 tighten action.
    5. Since spanners become unusable after use, one spanner pickup is required per nut (except if already carrying one usable spanner).
    6. Sum these costs over all nuts, greedily assuming best reachable spanner per nut.
    7. Incorporate walking moves between nuts to estimate realistic travel cost.
    8. Return the sum as heuristic; 0 if all nuts tightened.

    The heuristic is not admissible but expected to effectively guide the search by approximating
    the main costs: walking, picking spanners, and tightening nuts.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static_links = [get_parts(fact) for fact in task.static if match(fact, "link", "*", "*")]
        self.locations = set()
        # Build undirected graph of locations from links.
        self.locations_graph = {}
        for l1, l2 in self.static_links:
            self.locations.add(l1)
            self.locations.add(l2)
        for loc in self.locations:
            self.locations_graph[loc] = set()
        for l1, l2 in self.static_links:
            self.locations_graph[l1].add(l2)
            self.locations_graph[l2].add(l1)

        # Extract nuts that must be tightened from goals: facts like (tightened nutX)
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened" and len(parts) == 2:
                self.goal_nuts.add(parts[1])

    def _bfs_distance(self, start):
        """
        Compute shortest distance (number of steps) from start location to all other locations
        using BFS on the undirected locations graph.
        Returns a dict: location -> distance (int)
        """
        from collections import deque

        dist = {loc: None for loc in self.locations}
        dist[start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.locations_graph[current]:
                if dist[neighbor] is None:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        return dist

    def __call__(self, node):
        state = node.state

        # Quick return if already at goal.
        if self.goals <= state:
            return 0

        # Helper to find all facts matching pattern
        def find_facts(pred, *args):
            return {fact for fact in state if match(fact, pred, *args)}

        # Get man's name by checking who appears in (at ?m ?loc), with type man.
        # We assume exactly one man. We find the fact (at ?m ?loc) where ?m is man: assume any man in state.
        man_facts = [get_parts(f) for f in state if match(f, "at", "*", "*")]
        man = None
        man_location = None
        # Identify man by intersection with objects of type man from domain? We only have one man: so find "at" with ?m being man
        # Since no explicit typing here, we identify man by checking carrying facts or names from the given problem instances:
        # We'll identify the man as the object appearing in facts 'carrying ?m SPANNER' or in '(at ?m LOC)' where ?m matches any man from domain or objects.
        # Safer: find any object that appears as "carrying ?m spanner" or that appears as "at ?m LOC" and that also appears in carrying facts.
        # For simplicity, find first object in carrying facts or in at facts that is in all carrying facts.

        # First, gather all objects with type man from state by intersection between (at X L) and (carrying X S).
        carrying_facts = [get_parts(f) for f in state if match(f, "carrying", "*", "*")]
        carrying_man_candidates = {parts[1] for parts in carrying_facts}
        if carrying_man_candidates:
            # Choose one candidate as man
            man = next(iter(carrying_man_candidates))
            # Find location of man
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "at" and parts[1] == man:
                    man_location = parts[2]
                    break
        else:
            # Man carries nothing, so find who is "at" man
            # Find any man object from all at facts that appears in man objects of the domain is unknown;
            # just pick the one from problem instances: e.g. 'bob' is the only man
            # Try to see if goal or init fact mentions man:
            # Fallback: find any object called "bob"
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "at" and parts[1] == "bob":
                    man = "bob"
                    man_location = parts[2]
                    break
            # If still none found, guess first at fact with man-like name
            if man is None:
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == "at":
                        man = parts[1]
                        man_location = parts[2]
                        break

        if man is None or man_location is None:
            # Cannot find the man in the current state, pessimistic heuristic
            return 1000  # arbitrary large heuristic

        # Find all loose nuts that need tightening (goal nuts that are still loose, NOT tightened)
        # So check for each goal nut:
        # - not (tightened nut)
        # - (loose nut)
        loose_nuts = set()
        tightened_nuts = {get_parts(f)[1] for f in state if match(f, "tightened", "*")}
        for nut in self.goal_nuts:
            if nut not in tightened_nuts:
                # Verify nut is loose now in state
                if any(match(f, "loose", nut) for f in state):
                    loose_nuts.add(nut)
                else:
                    # If not loose, but nut not tightened -> pessimistic or ignore?
                    # Ignore here, pessimistic return if nut neither loose nor tightened
                    return 1000

        if not loose_nuts:
            # All nuts are tightened => heuristic 0
            return 0

        # Find location of all loose nuts
        nut_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] in loose_nuts:
                nut_locations[parts[1]] = parts[2]
        # If some nut location missing, pessimistic
        if len(nut_locations) < len(loose_nuts):
            return 1000

        # Find all usable spanners and their locations
        usable_spanners = set()
        spanner_locations = {}
        # Spanners can be: at location OR carried by man
        # A spanner is usable if (usable ?s) is true in the state.
        # Find all usable spanners:
        usable_spanner_objs = {get_parts(f)[1] for f in state if match(f, "usable", "*")}
        # Find their locations either (at ?s ?loc) or (carrying ?m ?s)
        for spanner in usable_spanner_objs:
            # Check if carried by man:
            if any(match(f, "carrying", man, spanner) for f in state):
                spanner_locations[spanner] = "carried"
                usable_spanners.add(spanner)
            else:
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == "at" and parts[1] == spanner:
                        spanner_locations[spanner] = parts[2]
                        usable_spanners.add(spanner)
                        break

        # Determine if man is currently carrying a usable spanner:
        carrying_spanner = None
        for spanner in usable_spanners:
            if spanner_locations[spanner] == "carried":
                carrying_spanner = spanner
                break

        # Precompute BFS distances from all relevant locations to all locations needed:
        # We'll compute for:
        # - Man's current location
        # - All spanner locations that are not carried
        # - All nut locations
        all_relevant_locs = {man_location}
        all_relevant_locs.update(loc for loc in spanner_locations.values() if loc != "carried")
        all_relevant_locs.update(nut_locations.values())

        # Compute BFS distances only once from every location in all_relevant_locs for fast lookup
        distances_map = {}
        for loc in all_relevant_locs:
            distances_map[loc] = self._bfs_distance(loc)

        # Strategy:
        # For each loose nut, estimate (a) cost to get usable spanner + pick it up (if not carrying spanner),
        # then (b) cost to walk to nut location and tighten.
        # Since spanners become unusable after one use, one pickup per nut is needed except if already carrying spanner.
        #
        # To reduce heuristic, we'll greedily assign closest usable spanner for each nut in isolation.
        # This underestimates combined travel but is efficient.
        #
        # After first nut, assume man needs to pick a new usable spanner from some location (possibly different).

        total_cost = 0
        man_pos_for_next = man_location
        carried_spanner_for_next = carrying_spanner is not None

        # To avoid multiple counting pickup of the same spanner, we'll not mark spanners used or remove them,
        # but assume spanners scattered in locations (or carried).

        for nut in loose_nuts:
            nut_loc = nut_locations[nut]

            # 1. Cost to get usable spanner (skip if man already carries one)
            if carried_spanner_for_next:
                spanner_pick_cost = 0
                walk_to_spanner = 0
            else:
                # Find minimum distance to any usable spanner location from current man position
                min_dist_to_spanner = None
                pick_spanner_loc = None
                for sp in usable_spanners:
                    sp_loc = spanner_locations[sp]
                    if sp_loc == "carried":
                        # Can't pick usable spanner from carrying for next nut (used up after first nut)
                        continue
                    dist = distances_map[man_pos_for_next].get(sp_loc)
                    if dist is not None and (min_dist_to_spanner is None or dist < min_dist_to_spanner):
                        min_dist_to_spanner = dist
                        pick_spanner_loc = sp_loc
                if min_dist_to_spanner is None:
                    # No reachable usable spanner -> high cost
                    return 1000
                walk_to_spanner = min_dist_to_spanner
                spanner_pick_cost = 1  # pickup_spanner action

            # 2. Cost to get to nut location from spanner location (or from current man location if carrying)
            start_pos_for_tighten = man_pos_for_next if carried_spanner_for_next else pick_spanner_loc
            dist_to_nut = distances_map[start_pos_for_tighten].get(nut_loc)
            if dist_to_nut is None:
                # Nut unreachable -> high cost
                return 1000
            walk_to_nut = dist_to_nut
            tighten_cost = 1  # tighten_nut action

            # Sum costs for this nut
            nut_cost = walk_to_spanner + spanner_pick_cost + walk_to_nut + tighten_cost

            total_cost += nut_cost

            # Update man position after tightening nut = nut location
            man_pos_for_next = nut_loc
            # After tightening, carried spanner is used up
            carried_spanner_for_next = False

        return total_cost
