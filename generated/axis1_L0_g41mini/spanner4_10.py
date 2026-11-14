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
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    This heuristic estimates the number of required actions for a man to tighten all loose nuts.
    It considers the locations of the man, spanners, and nuts, the usability of spanners,
    and connectivity between locations. The heuristic computes a minimal estimate of moving
    between locations to pickup spanners, carry them to nuts, and tighten all loose nuts.

    Assumptions:
    - There is a single man performing all actions.
    - Only one spanner can be carried at a time (as per domain definition carrying predicate).
    - Each nut must be tightened exactly once.
    - Usable spanners become unusable after one tighten action.
    - The man can walk only along linked locations.
    - Traveling between locations costs 1 action per edge.
    - Pickup, tighten, and walk actions count as one action each.
    - No admissibility guarantee; heuristic aims to be informative and efficiently computable.

    Heuristic Initialization:
    - Extract all locations and their adjacency links into a graph structure.
    - Extract all spanners and their usability status.
    - Extract all nuts and whether they are loose or tightened.
    - Precompute shortest path distances between all locations using BFS for efficient distance queries.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all loose nuts in the current state.
       If none, heuristic is zero (goal reached).

    2. Identify all usable spanners and their locations.
       Identify which spanners the man is currently carrying (at most one).

    3. Identify the man's current location.

    4. For each loose nut, find the minimal distance from the man or from usable spanners if pickup needed.

    5. Consider that the man must:
       - Walk to a spanner location (or use one he's carrying if usable),
       - Pickup the spanner if not already carrying one,
       - Walk to the nut's location,
       - Perform a tighten action.

    6. Since each spanner becomes unusable after tightening one nut,
       multiple tightening steps may require multiple spanners.
       The heuristic assumes one nut per usable spanner.

    7. For multiple nuts, sum minimal costs for each nut greedily:
       - When man is carrying a usable spanner, he can directly walk to the nut and tighten.
       - Otherwise, he must walk to a spanner to pickup first.

    8. This heuristic accounts for travel and actions but ignores spanner reusability assignments for speed.

    9. Returns the sum of estimated action costs (walking + pickups + tightenings).

    """

    def __init__(self, task):
        # Save goal facts (nuts tightened)
        self.goals = task.goals

        # Store static facts (links) in adjacency dictionary for BFS
        self.adj = {}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.adj.setdefault(l1, set()).add(l2)
                self.adj.setdefault(l2, set()).add(l1)

        # Precompute all pairs shortest paths efficiently on demand:
        # We'll cache BFS results from queried locations.
        self.dist_cache = {}

    def bfs_dist(self, start):
        """Compute single-source shortest distances from start to all reachable locations in the graph."""
        if start in self.dist_cache:
            return self.dist_cache[start]

        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            for neighbor in self.adj.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        self.dist_cache[start] = distances
        return distances

    def __call__(self, node):
        state = node.state

        # Check if goal reached: all nuts tightened
        if self.goals <= state:
            return 0

        # Extract man (we assume single man as per domain)
        man = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                # Find man object using type man - from domain : only 'man' type is man
                # We do not have explicit type info here, but domain has only one man instance.
                # So we pick the one appearing in 'at' with type man.
                obj = get_parts(fact)[1]
                # Check if carrying predicate references this obj as man somewhere to confirm
                # or assume first 'at' fact with man object is sufficient as per domain definition.
                # We simply assume there's a single 'man' named in problem and that the only man will be used.
                # From examples: 'bob' is the man.
                # We'll find the man name by looking for presence in carrying predicate.
                if man is None:
                    # We detect man by the fact that he may carry spanners or is man type by name from examples.
                    # Conservative: pick the first object which occurs as 'carrying ?man ?spanner' agent
                    man_candidate = obj
                    for f in state:
                        if match(f, "carrying", man_candidate, "*"):
                            man = man_candidate
                            break
                    if man is None:
                        # If no carrying, fallback where object is man in problem (e.g. bob)
                        # Assume man found in fact like (at bob gate)
                        man = man_candidate
                if man is not None:
                    break

        if man is None:
            # If man not found in at facts, try to find from carrying facts
            for fact in state:
                if match(fact, "carrying", "*", "*"):
                    man = get_parts(fact)[1]
                    break
        if man is None:
            # If still none, return a high heuristic
            return 1000

        # Man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", man, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
            return 1000  # man location unknown

        # Usable spanners present and their locations
        usable_spanners = {}
        carrying_spanner = None  # which spanner man is carrying (only one can be carried per domain)

        for fact in state:
            if match(fact, "usable", "*"):
                sp = get_parts(fact)[1]
                usable_spanners[sp] = None  # to fill location

        for fact in state:
            if match(fact, "at", "*", "*"):
                obj = get_parts(fact)[1]
                loc = get_parts(fact)[2]
                if obj in usable_spanners:
                    usable_spanners[obj] = loc

        for fact in state:
            if match(fact, "carrying", man, "*"):
                carrying_spanner = get_parts(fact)[2]

        # Loose nuts and their locations
        loose_nuts = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                loose_nuts[nut] = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj = get_parts(fact)[1]
                loc = get_parts(fact)[2]
                if obj in loose_nuts:
                    loose_nuts[obj] = loc

        if len(loose_nuts) == 0:
            return 0  # no loose nuts, goal reached

        # Precompute shortest distances from man location and spanner locations to all locations
        dist_from_man = self.bfs_dist(man_loc)

        dist_from_spanner = {}
        for sp, loc in usable_spanners.items():
            dist_from_spanner[sp] = self.bfs_dist(loc)

        # Calculate heuristic sum of minimal costs to tighten all loose nuts
        # Strategy:
        # For each loose nut, find minimal cost to:
        #   - If carrying usable spanner: walk from man_loc to nut_loc + 1 (tighten)
        #   - Else:
        #       * walk from man_loc to closest usable spanner location, +1 pickup
        #       * walk from spanner location to nut location + 1 tighten
        # Minimize over usable spanners.
        # Sum these minimal costs.

        # To approximate, assume:
        # - The man must sequentially handle each nut independently (ignore multitasking)
        # - The man drops used spanner after one use, must pickup another for next nut

        # Start: if carrying a usable spanner, can tighten a nut directly after walking
        # Otherwise, must walk to a spanner, pickup, walk to nut, tighten.

        # For simplicity, we assume the man starts fresh (no spanner carried) except when carrying usable spanner.

        # But domain allows only carrying one spanner, and usable spanners become unusable after first tighten,
        # so for each nut assume a pickup+tighten cycle.

        total_cost = 0
        # Initially assume man is carrying usable spanner if carrying_spanner and it is usable
        carrying_usable = carrying_spanner in usable_spanners if carrying_spanner else False
        current_man_loc = man_loc
        spanners_left = set(usable_spanners.keys())
        if carrying_usable:
            # Assume carried spanner is usable and available
            spanners_left.discard(carrying_spanner)

        for nut, nut_loc in loose_nuts.items():
            # If carrying usable spanner:
            if carrying_usable:
                dist_to_nut = dist_from_man.get(nut_loc, 1000)
                # Cost: walk + tighten
                cost = dist_to_nut + 1
                total_cost += cost
                # After use, spanner becomes unusable, no longer carried
                carrying_usable = False
                current_man_loc = nut_loc
                # Update dist_from_man for new man location
                dist_from_man = self.bfs_dist(current_man_loc)
            else:
                # Need to pickup a spanner first
                # Find closest usable spanner location from current_man_loc
                best_spanner = None
                best_cost = 1000
                for sp in spanners_left:
                    sp_loc = usable_spanners[sp]
                    dist_to_sp = dist_from_man.get(sp_loc, 1000)
                    dist_sp_to_nut = dist_from_spanner[sp].get(nut_loc, 1000)
                    cost = dist_to_sp + 1 + dist_sp_to_nut + 1  # walk+pickup + walk + tighten
                    if cost < best_cost:
                        best_cost = cost
                        best_spanner = sp
                if best_spanner is None:
                    # No spanner reachable, large cost assumed
                    best_cost = 1000
                total_cost += best_cost
                if best_spanner is not None:
                    spanners_left.discard(best_spanner)
                    # Update man location to nut_loc for next iteration
                    current_man_loc = nut_loc
                    dist_from_man = self.bfs_dist(current_man_loc)
                # After picking spanner and tightening nut, not carrying usable spanner anymore
                carrying_usable = False

        return total_cost
