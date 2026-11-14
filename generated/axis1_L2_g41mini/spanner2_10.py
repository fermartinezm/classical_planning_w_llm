from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(qkdx0xe2 n16nw8x1 mpwrued2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (du770mtw).

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state from a given state.
    The goal requires all relevant "wqid8mog" objects (denoted as ?ris62) to hold the predicate (ne6sl557 ?ris62).
    The available actions suggest "moving" some kind of item (?n54xx - pt4b08k4) through locations (?ts6t4 - mz6r1d30),
    changing "qkdx0xe2" facts that seem to represent location or possession, and some "deployment" or "activation" action
    f7y6nzq3 that produces the effect (ne6sl557 ?ris62) and removes other predicates.

    Assumptions
    -----------
    - The goal is to achieve (ne6sl557 ?ris62) for specific wqid8mog objects.
    - Only one such predicate (ne6sl557) per wqid8mog needs to be achieved.
    - The f7y6nzq3 action, which produces (ne6sl557 ?ris62), requires:
        * (ndg963qz ?cz2xr) — some state/availability on du770mtw objects,
        * (qkdx0xe2 ?n54xx ?ts6t4) and (fqyrzl30 ?n54xx ?cz2xr) — linking pt4b08k4 objects with du770mtw and mz6r1d30 objects,
        * (of6wn8m4 ?ris62) and (qkdx0xe2 ?ris62 ?ts6t4).

    - The actions jql8txb3 and nm6lzp2x change the predicates (fqyrzl30 ...) and (qkdx0xe2 ...) affecting the "positions" and links.
    - The static facts lpmls56r denote adjacency or a "path" between mz6r1d30 objects.
    - Each (ne6sl557 ?ris62) goal fact appears exactly once in the goal.

    Heuristic Initialization
    ------------------------
    - Extract:
      * The set of goal wqid8mog objects (those that must satisfy (ne6sl557 ?ris62))
      * Adjacency relationships between mz6r1d30 objects (from lpmls56r static facts) to estimate shortest path distances.
      * Identify:
        - For each wqid8mog object ?ris62: whether it is currently (of6wn8m4 ?ris62) (i.e. active/available)
        - For each wqid8mog ?ris62: its current "qkdx0xe2 ?ris62 ?ts6t4" predicate, i.e. its current location.
      * For each pt4b08k4 object ?n54xx, their current location via "fqyrzl30 ?n54xx ?cz2xr".
      * For each du770mtw object ?cz2xr, whether (ndg963qz ?cz2xr) holds (availability/ready state).
      * The current (qkdx0xe2 ?n54xx ?ts6t4) facts and (qkdx0xe2 ?cz2xr ?ts6t4) to track locations.
    
    - Using adjacency info, precompute shortest distances between mz6r30 nodes.
      This allows quick minimal step estimation for moving objects/items over locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal wqid8mog object G:

      1. If (ne6sl557 G) is already true: requires 0 actions for that object.

      2. Otherwise (if not ne6sl557 G), the heuristic estimates how many actions are needed to perform the "f7y6nzq3" action on that wqid8mog object eventually.
         That action requires:
          - A du770mtw object (call it device) with (ndg963qz device) true
          - A pt4b08k4 object (carrier) fulfilling:
              (qkdx0xe2 carrier location) and (fqyrzl30 carrier device)
          - The wqid8mog object G must have (of6wn8m4 G) and (qkdx0xe2 G location) for the *same* location as the carrier.
          
         The effect is producing (ne6sl557 G), removing (of6wn8m4 G) and (ndg963qz device).

      3. Therefore, the heuristic estimates:
         - The minimal number of moves to bring some carrier and G to some du770mtw device that is (ndg963qz).
         - Moves here mean changing locations along lpmls56r paths using the nm6lzp2x action for carriers,
            jql8txb3 to assign carriers to devices,
            and possibly repositioning the wqid8mog object G itself along "qkdx0xe2" predicate.

      4. Detailed sub-steps:
         - For each du770mtw device D with (ndg963qz D) true (available), compute:
            * The minimum distance from G's current location to some mz6r1d30 ts location (for which qkdx0xe2 G ts holds).
            * The minimum distance from a carrier's current location (qkdx0xe2 carrier ts) to the same location ts.
            * The cost to achieve (fqyrzl30 carrier D) if not currently true, by using jql8txb3 (1 action, removes qkdx0xe2 carrier ts, adds fqyrzl30).
            * Note jql8txb3 precond: (qkdx0xe2 device ts) must hold for device, i.e. device must be at same location ts as carrier.

         - The device must be at the location ts to apply jql8txb3, so the device's position matters and may require movement (via nm6lzp2x).
         
         - The cost thus includes:
           * Moving the device to location ts if needed (nm6lzp2x with device and corresponding path)
           * Moving the carrier to location ts (nm6lzp2x etc.)
           * Moving G to location ts (nm6lzp2x)
           * One jql8txb3 action (assign carrier to device)
           * One f7y6nzq3 action (final goal achieving action)
           
         - Summing up all required movements as minimal paths on the static adjacency graph.

    The heuristic estimate for the state is the minimal summed cost over all goal wqid8mog objects.

    To keep computation efficient, shortest distances use BFS on static adjacency relations.

    The heuristic value is:
        sum over all unachieved (ne6sl557 ?ris62) of [ minimal cost to reach and apply final goal action on ?ris62 ].

    This heuristic is NOT necessarily admissible but aims at minimizing node expansions by guiding search toward states that unify carriers, devices, and wqid8mog objects at the same location ready to apply f7y6nzq3.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all connections between mz6r1d30 objects from static facts (lpmls56r)
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "lpmls56r", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.adjacency.setdefault(loc1, set()).add(loc2)
                # Assume undirected adjacency; also add reverse edge
                self.adjacency.setdefault(loc2, set()).add(loc1)

        # Precompute shortest distance between all mz6r1d30 locations by BFS
        # We'll build a helper method to compute distance on-demand with caching.

        # Extract goal wqid8mog objects (those for which (ne6sl557 ?ris62) in goals)
        self.goal_wqid8mog_objects = set()
        for g in self.goals:
            if match(g, "ne6sl557", "*"):
                _, obj = get_parts(g)[0], get_parts(g)[1]
                self.goal_wqid8mog_objects.add(obj)

        # We'll cache distances for efficiency
        self._distance_cache = {}

    def shortest_distance(self, start, end):
        """Return shortest distance (number of steps) between start and end mz6r1d30 locations using BFS."""
        if start == end:
            return 0
        key = (start, end)
        if key in self._distance_cache:
            return self._distance_cache[key]
        # BFS
        queue = [(start, 0)]
        visited = {start}
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.adjacency.get(current, []):
                if neighbor == end:
                    self._distance_cache[key] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # No path found
        self._distance_cache[key] = float('inf')
        return float('inf')

    def __call__(self, node):
        state = node.state

        # Quick check: if goal reached heuristic=0
        if self.goals <= state:
            return 0

        # Extract current facts by predicate for efficiency
        # Group facts by predicate name with lists of tuples of arguments
        facts_by_pred = {}
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            facts_by_pred.setdefault(pred, []).append(parts[1:])

        # Helpers for fast lookup:
        # qkdx0xe2 facts: (object, location)
        qkdx0xe2 = facts_by_pred.get("qkdx0xe2", [])
        qkdx0xe2_map = {}  # object -> set of locations (usually one location per object)
        for obj, loc in qkdx0xe2:
            qkdx0xe2_map.setdefault(obj, set()).add(loc)

        # fqyrzl30 facts: (pt4b08k4, du770mtw)
        fqyrzl30 = facts_by_pred.get("fqyrzl30", [])
        fqyrzl30_map = {}
        for carrier, device in fqyrzl30:
            fqyrzl30_map[carrier] = device

        # ndg963qz facts (du770mtw objects with ndg963qz true)
        ndg963qz_set = {obj[0] for obj in facts_by_pred.get("ndg963qz", [])}

        # of6wn8m4 facts (wqid8mog objects that are "active"/"available")
        of6wn8m4_set = {obj[0] for obj in facts_by_pred.get("of6wn8m4", [])}

        # ne6sl557 achieved wqid8mog objects (goal achieved)
        ne6sl557_set = {obj[0] for obj in facts_by_pred.get("ne6sl557", [])}

        # The heuristic sums over each goal wqid8mog object that is NOT yet ne6sl557
        total_estimated_cost = 0

        # Collect du770mtw devices that can be used as target devices (those with ndg963qz true)
        available_devices = ndg963qz_set.copy()
        if not available_devices:
            # If no device is currently available with ndg963qz, then no f7y6nzq3 action can currently be applied,
            # so the heuristic assumes cost at least 1 to create availability; we model by adding 1 penalty.
            # (In practice, devices apparently become unavailable after f7y6nzq3, so maybe reusability).
            # To avoid infinite heuristic, assume at least 1 action needed.
            # For heuristic guidance, count penalty and continue.
            available_devices = set()

        # Precompute carrier locations and their devices
        carriers = set(fqyrzl30_map.keys())
        # Also remember carrier locations via qkdx0xe2
        carrier_locs = {}
        for car in carriers:
            locs = qkdx0xe2_map.get(car, None)
            if locs:
                # There should be exactly one location per carrier for qkdx0xe2
                carrier_locs[car] = next(iter(locs))

        # For each goal object G that needs ne6sl557 but is not yet achieved:
        for G in self.goal_wqid8mog_objects:
            if G in ne6sl557_set:
                continue  # no cost needed

            # If G is not currently of6wn8m4, it must be activated (no actions in domain to achieve of6wn8m4?)
            # We'll assume heuristic cost grows with needing to move G to a device with of6wn8m4.
            is_active = G in of6wn8m4_set
            if not is_active:
                # In domain, f7y6nzq3 deletes of6wn8m4 of ris62 and adds ne6sl557 ris62,
                # so only objects that currently have of6wn8m4 can be used to perform final action.
                # Hence if not active, assume we need to bring it to active state first (unknown how),
                # so we assign a base penalty (>=1).
                base_cost = 1
            else:
                base_cost = 0

            # Get locations of G via qkdx0xe2 facts (should be exactly one)
            G_locs = qkdx0xe2_map.get(G)
            if not G_locs:
                # No location known for G; heuristic large penalty
                total_estimated_cost += 10 + base_cost
                continue
            G_loc = next(iter(G_locs))

            # Among all possible target devices D that are currently ndg963qz,
            # we will compute minimal estimated cost like this:

            minimal_cost_for_G = float('inf')
            if not available_devices:
                # No available device now. We add penalty and skip detailed cost
                minimal_cost_for_G = 10 + base_cost
            else:
                for D in available_devices:
                    # Device D location
                    D_locs = qkdx0xe2_map.get(D)
                    if not D_locs:
                        continue  # ignore devices without location

                    D_loc = next(iter(D_locs))

                    # For each carrier, compute costs to move G and carrier and device as needed
                    for carrier in carriers:
                        car_loc = carrier_locs.get(carrier)
                        if car_loc is None:
                            continue  # ignore carriers without location

                        # The target location for applying final steps must be some ts6t4 (mz6r1d30) location
                        # where:
                        #  - device D is present at that location (qkdx0xe2 D ts)
                        #  - carrier and G must be present at the same location ts
                        # Here we approximate that location by the device location (D_loc),
                        # and if needed we incur costs moving G and carrier and device to that location.

                        # 1. Cost to move device D from D_loc to D_loc (0)
                        cost_move_device = 0  # device is assumed at D_loc already

                        # 2. Cost to move carrier from car_loc to D_loc
                        dist_carrier = self.shortest_distance(car_loc, D_loc)

                        # 3. Cost to move G from G_loc to D_loc
                        dist_G = self.shortest_distance(G_loc, D_loc)

                        # If any distance is infinite, path impossible
                        if dist_carrier == float('inf') or dist_G == float('inf') or cost_move_device == float('inf'):
                            continue

                        # Number of move actions required correspond to nm6lzp2x actions to move qkdx0xe2 predicates along adjacency.

                        # Consider costs for:
                        # - moving device (nm6lzp2x) times distance (here zero)
                        # - moving carrier (nm6lzp2x) times distance
                        # - moving G (nm6lzp2x) times distance
                        # - jql8txb3 (1 action): assign carrier to device at same location
                        # - f7y6nzq3 (1 action): final goal achievement

                        cost = dist_carrier + dist_G + cost_move_device + 2  # 2 actions for jql8txb3 and f7y6nzq3

                        # Additionally, if carrier is not assigned to device (fqyrzl30 carrier device),
                        # then jql8txb3 action is required.
                        carrier_assigned = fqyrzl30_map.get(carrier) == D
                        if not carrier_assigned:
                            # jql8txb3 replaces qkdx0xe2 carrier ts with fqyrzl30 carrier device at same location
                            # which is the jql8txb3 action accounted above.
                            # No extra cost needed beyond jql8txb3 already counted.

                            pass  # accounted above

                        # Base penalty for inactive G
                        cost += base_cost

                        if cost < minimal_cost_for_G:
                            minimal_cost_for_G = cost

            # Add minimal cost for this G to total
            if minimal_cost_for_G == float('inf'):
                # no path found, assign large heuristic cost to avoid dead-ends cheaply
                minimal_cost_for_G = 20 + base_cost

            total_estimated_cost += minimal_cost_for_G

        return int(total_estimated_cost)
