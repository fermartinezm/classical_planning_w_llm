from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the number of actions required to tighten all loose nuts
    using available usable spanners and one man (the mechanic). It accounts for the man
    moving between locations, picking up spanners when needed, and tightening the nuts.

    Assumptions:
    - The man can carry only one spanner at a time. (Derived from the domain and pickup.)
      However, the domain predicates allow carrying one spanner (carrying ?m ?s), and pickup_spanner effects
      make spanner not present at location and carrying true; no action to drop, so 
      from given domain, man can carry multiple spanners only if multiple pickup_spanner are performed,
      but pickup_spanner precondition requires spanner at location, 
      the domain doesn't have unload or putdown action, so carrying multiple spanners simultaneously is possible only 
      if man picks them all at the same location? The example state shows carrying bob spanner1 and bob spanner2.
      Thus, man can carry multiple spanners simultaneously.
    - The spanner becomes unusable after tightening a nut, so a spanner must be replaced before tightening another nut.
    - The man can move only along linked locations.
    - Tightening a nut requires the man to be at the nut's location, carrying a usable spanner, and the nut must be loose.
    - Goal is that all nuts are tightened.
    - The heuristic is allowed to be non-admissible and should be efficiently computable.

    Heuristic Initialization:
    - Extract the set of nuts from goals (those to be tightened).
    - Extract the links between locations and create a graph for shortest path estimation.
    - Extract static facts on links to build connectivity graph.
    - Extract the locations of spanners and which are usable.
    - Store initial task goals.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all loose nuts needing to be tightened.
    2. Identify nuts already tightened and exclude them from cost.
    3. Obtain current location of the man.
    4. Compute locations of all usable spanners.
    5. For each loose nut:
       - Estimate the minimal travel cost for the man to go from his current or last nut location to a usable spanner location,
         then from that spanner location to the nut location.
       - Account for the cost of pickup_spanner action (1 action) and tighten_nut action (1 action).
       - After tightening, the spanner is no longer usable; anticipate the need to pick up another usable spanner for the next nut.
    6. Assume man can carry multiple spanners simultaneously to reduce travel, or otherwise
       estimate one spanner pickup per nut.
    7. A reasonable approximation: 
       - Sum over all loose nuts:
         - 1 action for tighten_nut
         - 1 action for pickup_spanner (unless the man is currently carrying a usable spanner at start)
         - travel cost:
           - distance from current man's location to closest usable spanner (only once, if carrying at start)
           - then from that spanner to nut location
       - For subsequent nuts, assume:
         - travel from previous nut location to new spanner location + spanner pickup + nut location + tighten_nut
    8. Use shortest path distances over link graph to estimate travel costs.
    9. To reduce complexity, estimate sum by a simple greedy strategy:
       - Start man location as current location.
       - Maintain the set of usable spanners.
       - For each nut in arbitrary order (or nearest-first heuristic), add:
         a) travel from current_location to nearest usable spanner
         b) pickup_spanner action (1)
         c) travel from spanner location to nut location
         d) tighten_nut action (1)
         e) remove that spanner from usable (since it's consumed)
         f) update current_location to nut location
    10. If no usable spanners remain but nuts still loose, assume worst case and add pickup from any spanner location.
    11. Return the total cost estimate as heuristic value.
    12. Return heuristic 0 if state is goal.

    Note:
    - This heuristic does NOT guarantee admissibility, but aims to minimize expanded nodes during greedy best first search.
    - Ignores some complexities like carrying multiple spanners simultaneously or reusing spanners.
    """

    def __init__(self, task):
        """Initialize static structures and extract necessary info from the task."""
        self.goals = task.goals  # Goal facts (tightened nuts)
        self.static = task.static  # Static facts (links)

        # Extract links and create undirected graph for shortest path calculation.
        self.graph = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1)

        # Extract nuts that must be tightened from the goal facts and all nuts existing in the problem.
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened" and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # Precompute shortest paths between all pairs of locations using BFS on-demand cache
        self._dist_cache = {}

    def shortest_path_length(self, start, goal):
        """Compute shortest path length (number of hops) between start and goal in link graph."""
        if start == goal:
            return 0
        if (start, goal) in self._dist_cache:
            return self._dist_cache[(start, goal)]
        # BFS
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.graph.get(current, []):
                if neighbor == goal:
                    self._dist_cache[(start, goal)] = dist + 1
                    self._dist_cache[(goal, start)] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # If no path found (should not happen in well-formed instances)
        # Treat as infinite distance by returning a large number
        self._dist_cache[(start, goal)] = 1000
        self._dist_cache[(goal, start)] = 1000
        return 1000

    def __call__(self, node):
        """
        Estimate the number of actions required to tighten all loose nuts.

        Returns 0 if the goal is reached.
        Otherwise, computes a greedy approximation of actions consisting of:
         - travel (walk) cost
         - pickup_spanner cost (1 action)
         - tighten_nut cost (1 action)
        """
        state = node.state

        # If goal reached, heuristic = 0
        if self.goals <= state:
            return 0

        # Extract man name (there is only one man in domain)
        # Identify unique man by predicate at man location or carrying
        man_name = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                pred, m, loc = get_parts(fact)
                if pred == "at":
                    # Check if m is a man by looking at types? The domain is typed but we rely on given man in problem usually one man.
                    # The domain defines man as type, and we have one man named bob in examples
                    # So if we see a fact (at bob locationX), bob is our man.
                    # Let's just detect man names by the fact they appear in carrying predicates or at predicates involving man.
                    # Since no type info in this representation, we treat first found man as example.
                    # Safer to look for carrying predicates with man.
                    pass
            if match(fact, "carrying", "*", "*"):
                man_candidate = get_parts(fact)[1]
                man_name = man_candidate
                break
        if man_name is None:
            # Try fallback: scan for at with possible man
            for fact in state:
                if match(fact, "at", "*", "*"):
                    obj = get_parts(fact)[1]
                    # Heuristic assumes exactly one man; this will be first found entity appearing as man.
                    man_name = obj
                    break
        # If still None, fail safe return high cost
        if man_name is None:
            return 1000

        # Current location of the man
        man_loc = None
        for fact in state:
            if match(fact, "at", man_name, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
            # If man current location unknown, return high cost
            return 1000

        # Identify all loose nuts remaining to be tightened
        loose_nuts = []
        for nut in self.goal_nuts:
            # A nut is loose if (loose nut) in state and (tightened nut) not in state
            if ("(loose {})".format(nut) in state or 
                # the domain example uses only (loose nut) to indicate nut loose,
                # so presence of loose matters
                # Some nuts may be neither loose nor tightened, but we only consider nuts in goal.
                True) and ("(tightened {})".format(nut) not in state):
                # double check loose
                if ("(loose {})".format(nut) in state):
                    loose_nuts.append(nut)
                else:
                    # If no loose predicate, consider tightened or not
                    pass  # skip nut if no loose fact

        # Current usable spanners available (usable ?s)
        usable_spanners = set()
        spanner_locations = {}
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                usable_spanners.add(spanner)
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                spanner_locations[obj] = loc

        # Spanners currently carried by man (carrying ?m ?s)
        carried_spanners = set()
        for fact in state:
            if match(fact, "carrying", man_name, "*"):
                sp = get_parts(fact)[2]
                carried_spanners.add(sp)

        # Filter carried spanners to those usable
        carried_usable = usable_spanners & carried_spanners

        # The heuristic action costs for each action type
        WALK_COST = 1           # walk action between two linked locations
        PICKUP_SPANNER_COST = 1 # pickup_spanner action
        TIGHTEN_NUT_COST = 1    # tighten_nut action

        # For each nut to tighten, estimate cost
        # Heuristic approach:
        #  - If carrying a usable spanner, no pickup needed for that nut, else need pickup_spanner action
        #  - Travel from current man location to spanner location (if pickup needed)
        #  - Travel from spanner location to nut location
        #  - Tighten nut action
        #  - After each nut, man location updates to that nut's location
        #  - Spanner used up after tightening that nut, so remove it from usable spanners and carried spanners if relevant

        man_current_loc = man_loc
        available_usable_spanners = usable_spanners.copy()

        # Track carried spanners that are usable - we treat as usable without pickup cost initially
        carried_usable_spanners = carried_usable.copy()
        # Remove from available_usable_spanners those carried (since already carried)
        available_usable_spanners.difference_update(carried_usable_spanners)

        total_cost = 0

        # Sort nuts by minimal tentative cost to tighten, to simulate greedy nearest nut first
        # For efficiency just use the list as is (order of nuts in set) to avoid overhead

        for nut in loose_nuts:
            # Location of the nut
            nut_loc = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_loc = get_parts(fact)[2]
                    break
            if nut_loc is None:
                # Nut not located, assume high cost and continue
                total_cost += 1000
                continue

            need_pickup = True
            # If man already carrying usable spanner, use it for this nut
            if carried_usable_spanners:
                # Use one carried usable spanner
                spanner_used = carried_usable_spanners.pop()
                need_pickup = False
                spanner_loc = man_current_loc  # Already carried, we consider man at spanner
            else:
                # Need to pick up a usable spanner from some location
                # Find closest usable spanner location to man_current_loc
                min_dist = 10000
                chosen_spanner = None
                chosen_spanner_loc = None
                for sp in available_usable_spanners:
                    loc = spanner_locations.get(sp, None)
                    if loc is None:
                        continue
                    dist = self.shortest_path_length(man_current_loc, loc)
                    if dist < min_dist:
                        min_dist = dist
                        chosen_spanner = sp
                        chosen_spanner_loc = loc
                if chosen_spanner is None:
                    # No usable spanners available (should not happen if problem solvable)
                    # Assign high cost and break
                    total_cost += 1000
                    break
                spanner_used = chosen_spanner
                spanner_loc = chosen_spanner_loc
                available_usable_spanners.remove(spanner_used)

            # Add cost for traveling to spanner location if needed
            if need_pickup:
                dist_to_spanner = self.shortest_path_length(man_current_loc, spanner_loc)
                total_cost += dist_to_spanner * WALK_COST
                # Pickup spanner cost
                total_cost += PICKUP_SPANNER_COST
            else:
                # Spanner already carried, no pickup cost or travel to spanner location
                pass

            # Add cost for traveling from spanner location to nut location
            dist_spanner_to_nut = self.shortest_path_length(spanner_loc, nut_loc)
            total_cost += dist_spanner_to_nut * WALK_COST

            # Add tighten nut cost
            total_cost += TIGHTEN_NUT_COST

            # Update man current location to nut location
            man_current_loc = nut_loc

        return total_cost
