from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(tightened nut1 locatable13)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (spanner) domain.
    
    Summary:
    This heuristic estimates the number of actions required to reach the goal in the spanner domain,
    where the main task is to move tightened nuts from their current locatables to goal locations,
    represented by predicates of the form (tightened-2 locatable location).
    It factors in the nuts' fastening state (loose or tightened), their current locatables, the
    links status, and location facts, and how nuts/spanners need to be transported and tightened
    at the goal location.
    
    Assumptions:
    - The goal requires certain locatables to be tightened at specified locations (tightened-2 predicates).
    - Each nut is associated with one tightened locatable (spanner) that needs to be transported.
    - Moving the spanner (locatable with "link" status) closer to the goal location is key.
    - Actions revolve around moving links, loosening/tightening nuts, and carrying locatables.
    - The heuristic does not have to be admissible, but should be informative and efficient.
    
    Heuristic Initialization:
    - Extract all goal predicates of the form (tightened-2 locatable location), representing target tightened spanner-locations.
    - Extract static facts defining relations such as carries, links, loose/tightened connections, places where nuts and locatables currently reside.
    - Precompute adjacency relations from static facts for efficient reasoning about moving tightened nuts/spanners.
    
    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the goal tightened-2 predicates: which locatables must be tightened at which locations.
    
    2. For each such goal, check the current state:
       - If the locatable is already tightened-2 in the required location, cost for this goal is 0.
       - Else, estimate cost based on how far the locatable (spanner) with tightened nuts is from the goal location.
    
    3. Use static facts (loose, carrying, link, usable, at) to estimate distances / required actions:
       - If the required (tightened nut locatable) connection is missing or in the wrong location,
         estimate additional cost to move the link or nut/spanner towards the target.
       - If the nut is loose-2 in some location other than where it should be, account for tightening moves.
    
    4. Sum up per-goal estimates: 
       - Count how many moves are required to bring the locatable to the goal location (roughly by counting 'loose' or 'at' relations).
       - Include actions for tightening/loosen nuts as needed.
       - In the absence of explicit distances, use counts and differences in location to estimate.
    
    5. Return the total heuristic estimate, with heuristic 0 only when all goal tightened-2 predicates are satisfied.
    """

    def __init__(self, task):
        # Goals: tuples (locatable, location) for predicate tightened-2
        self.goal_tightened2 = set()
        for g in task.goals:
            if match(g, "tightened-2", "*", "*"):
                parts = get_parts(g)
                self.goal_tightened2.add((parts[1], parts[2]))

        # Static info for adjacency and relations
        self.loose = set()
        self.loose2 = set()  # (nut, location) pairs
        self.tightened = set()  # (nut, locatable)
        self.at = set()  # (locatable, locatable)
        self.at2 = set()  # locations at-2
        self.link = set()
        self.carrying = set()
        self.carrying_rev = dict()  # locatable -> what it is carrying
        self.usable = set()  # (locatable, locatable)

        for fact in task.static:
            if match(fact, "loose", "*", "*"):
                parts = get_parts(fact)
                self.loose.add((parts[1], parts[2]))
            elif match(fact, "loose-2", "*", "*"):
                parts = get_parts(fact)
                self.loose2.add((parts[1], parts[2]))
            elif match(fact, "tightened", "*", "*"):
                parts = get_parts(fact)
                self.tightened.add((parts[1], parts[2]))
            elif match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                self.at.add((parts[1], parts[2]))
            elif match(fact, "at-2", "*"):
                parts = get_parts(fact)
                self.at2.add(parts[1])
            elif match(fact, "link", "*"):
                parts = get_parts(fact)
                self.link.add(parts[1])
            elif match(fact, "carrying", "*", "*"):
                parts = get_parts(fact)
                self.carrying.add((parts[1], parts[2]))
                self.carrying_rev[parts[1]] = parts[2]
            elif match(fact, "usable", "*", "*"):
                parts = get_parts(fact)
                self.usable.add((parts[1], parts[2]))

        # Build adjacency for loose and at relations to estimate moves
        # Using a graph-like approach with locatables as nodes and edges for 'loose' and 'at'
        self.adjacency = {}
        for (n1, n2) in self.loose:
            self.adjacency.setdefault(n1, set()).add(n2)
            self.adjacency.setdefault(n2, set()).add(n1)
        for (l1, l2) in self.at:
            self.adjacency.setdefault(l1, set()).add(l2)
            self.adjacency.setdefault(l2, set()).add(l1)

        # Also include usable links as edges, they often allow movement/tightening actions
        for (u1, u2) in self.usable:
            self.adjacency.setdefault(u1, set()).add(u2)
            self.adjacency.setdefault(u2, set()).add(u1)

    def __call__(self, node):
        state = node.state

        # Convert state facts into useful sets for fast querying
        tightened2_state = set()
        loose2_state = set()
        tightened_state = set()
        link_state = set()
        carrying_state = set()
        carrying_rev_state = dict()
        loose_state = set()
        at_state = set()
        at2_state = set()

        for fact in state:
            if match(fact, "tightened-2", "*", "*"):
                parts = get_parts(fact)
                tightened2_state.add((parts[1], parts[2]))
            elif match(fact, "loose-2", "*", "*"):
                parts = get_parts(fact)
                loose2_state.add((parts[1], parts[2]))
            elif match(fact, "tightened", "*", "*"):
                parts = get_parts(fact)
                tightened_state.add((parts[1], parts[2]))
            elif match(fact, "link", "*"):
                parts = get_parts(fact)
                link_state.add(parts[1])
            elif match(fact, "carrying", "*", "*"):
                parts = get_parts(fact)
                carrying_state.add((parts[1], parts[2]))
                carrying_rev_state[parts[1]] = parts[2]
            elif match(fact, "loose", "*", "*"):
                parts = get_parts(fact)
                loose_state.add((parts[1], parts[2]))
            elif match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                at_state.add((parts[1], parts[2]))
            elif match(fact, "at-2", "*"):
                parts = get_parts(fact)
                at2_state.add(parts[1])

        # If current state satisfies goals, heuristic is 0
        if self.goal_tightened2 <= tightened2_state:
            return 0

        # Helper: BFS search to estimate minimum moves (distance) between locatables or location nodes
        def bfs_distance(start_nodes, targets, allowed_nodes=None, max_dist=50):
            """
            BFS to find shortest distance from any of start_nodes to any target in targets
            within allowed_nodes (if given). Returns min distance or max_dist if unreachable.

            Nodes are strings (locatable/location names).
            """
            if not start_nodes or not targets:
                return max_dist

            visited = set()
            frontier = [(node, 0) for node in start_nodes]
            for node, _ in frontier:
                visited.add(node)
            targets = set(targets)
            while frontier:
                current, dist = frontier.pop(0)
                if current in targets:
                    return dist
                if dist >= max_dist:
                    continue
                neighbors = self.adjacency.get(current, set())
                if allowed_nodes is not None:
                    neighbors = neighbors & allowed_nodes
                for nbr in neighbors:
                    if nbr not in visited:
                        visited.add(nbr)
                        frontier.append((nbr, dist + 1))
            return max_dist

        heuristic_total = 0

        # For each goal (locatable, location), compute estimated cost
        # Cost components:
        # - Distance to bring the spanner (locatable) to the goal location (tightened-2 requires locatable at location)
        # - If nut loose-2 in wrong location, account for tightening moves
        # - Moves needed to position nut, spanner, and link correctly
        
        # Unpack sets for quick membership tests:
        link_nodes = link_state
        tightened_nuts = tightened_state
        loose_2_facts = loose2_state
        at_2_facts = at2_state

        # We assume that moving locatables between locations is modeled by adjacency,
        # thus approximating the number of walk or pickup_spanner actions.

        for (goal_locatable, goal_location) in self.goal_tightened2:
            # Check if goal already achieved for this locatable
            if (goal_locatable, goal_location) in tightened2_state:
                continue  # no cost for this goal

            # Find current location(s) of the locatable
            # If tightened ?nut ?locatable holds for some nut and this locatable
            nuts_for_locatable = {nut for (nut, loc) in tightened_nuts if loc == goal_locatable}
            if not nuts_for_locatable:
                # No tightened nut on the locatable; we expect a nut tightened on it
                # Use nuts that are tightened somewhere else? Choose nearest nut
                # Get nuts loose-2 somewhere + tightened on some locatable + try estimate cost
                nut_candidates = {n for (n, _) in loose_2_facts} | {n for (n, _) in tightened_nuts}
                # For heuristic, we pick arbitrary nut to associate (likely nut1, nut2, nut3)
                nut = next(iter(nut_candidates)) if nut_candidates else None
                # If no nut at all, cost at least 3 to place and tighten
                base_cost = 3  
            else:
                nut = next(iter(nuts_for_locatable))
                base_cost = 0

            # Current location of the nut via loose-2 or tightened?
            curr_nut_loc = None
            # Prefer loose-2 location for nut, else assign goal location for loosened nuts
            for (n, loc) in loose_2_facts:
                if n == nut:
                    curr_nut_loc = loc
                    break
            if curr_nut_loc is None and nut is not None:
                # Try from tighter
                for (n, locatable) in tightened_nuts:
                    if n == nut:
                        # We don't know the nut location exactly, but assume near locatable
                        curr_nut_loc = None

            # Current location(s) of the locatable: check if it is "tightened" on some locatable
            # Locatable position can be estimated either as the locatable itself, or linked location
            curr_locatable_links = []
            # A locatable is "link" if in link_state
            locatable_is_link = goal_locatable in link_nodes

            # Check if locatable is linked to other locatables (via loose or carrying or at)
            # We approximate locatable position by link nodes if possible
            # Assume locatable = node, so position = locatable itself

            # Distance component: estimated by BFS from locatable to goal location
            # BFS over allowed nodes: only locatable nodes and locations (string names)
            allowed_nodes = None  # no restriction to avoid misses
            dist_locatable_to_goal = bfs_distance([goal_locatable], [goal_location], allowed_nodes)

            # Check if (loose-2 nut goal_location) holds: this means nut in place but loose
            nut_loose_at_goal = (nut, goal_location) in loose_2_facts if nut else False
            if nut_loose_at_goal:
                # Nut needs tightening: estimated cost 1 action to tighten_nut
                cost_tighten = 1
            else:
                cost_tighten = 2  # might require moving nut and tightening

            # Count how many spanners at wrong location that need moving
            spanner_at_goal = (goal_locatable, goal_location) in tightened2_state
            cost_move_spanner = dist_locatable_to_goal

            # Additional cost approximations:
            # If nut is at wrong loose-2 location, cost to move it closer: BFS between loose-2 nut location and goal_location
            if curr_nut_loc is not None and curr_nut_loc != goal_location:
                dist_nut_to_goal = bfs_distance([curr_nut_loc], [goal_location])
            else:
                dist_nut_to_goal = 0

            # Estimate total cost as sum of moves and tighten steps
            cost = cost_move_spanner + cost_tighten + dist_nut_to_goal

            # Sanity: minimal cost at least 1 if not in goal already
            if cost < 1:
                cost = 1

            heuristic_total += cost

        return heuristic_total
