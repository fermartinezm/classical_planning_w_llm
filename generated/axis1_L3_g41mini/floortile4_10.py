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

    Summary
    -------
    Estimates the number of required actions to achieve all goal `(tightened-2 locatable location)` facts.
    The domain involves relocating tightened nuts on locatable objects to specific locations.
    Nuts have to be loosened in one location and tightened in another, and locatables must be properly linked and carried.

    Assumptions
    -----------
    - Each goal requires one locatable to be tightened at a specific location (`tightened-2` predicate).
    - Transfers of tightened nuts between locatables/locations require multiple "walk" or "pickup_spanner" actions.
    - The "link" predicate indicates which locatables/spanners are currently usable for action moves.
    - Movement and tightening actions follow the patterns in the domain actions.
    - Static facts (`loose`, `at`, `carrying`, `usable`) describe connectivity and possible transitions.
    - The heuristic is not required to be admissible but aims at guiding search by estimating the minimal number of operations roughly.

    Heuristic Initialization
    -------------------------
    From the static facts, we extract:
    - The directed graph of locatables connected by `(loose ?loc1 ?loc2)` representing possible moves/transfers paths.
    - The possible usable transitions `(usable ?loc1 ?loc2)` to approximate how tightening can be propagated.
    - Mapping from nuts and locatables to current tightness states (used in heuristic calls).
    From the goals, extract the set of `(tightened-2 locatable location)` goals.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all goals of the form `(tightened-2 locatable location)` that must be achieved.

    2. For each goal `(tightened-2 L loc)`, check if it is already satisfied in the current state;
       if yes, no cost for that goal.

    3. For unsatisfied goals:
       - Estimate the minimum number of actions to get the nut tightened on the target locatable at the desired location.
       - This involves:
         a. Finding the current locatable that the nut is tightened on.
         b. Finding a shortest path of locatables via `(loose)` connections to move the nut and tighten it on the target locatable.
         c. Estimating the cost to move the nut across that path by swapping tightened nuts between locatables (using `tighten_nut-2`, `walk-2`, `pickup_spanner-2`, etc.), plus the final tightening action.
         d. Accounting for moving the nut from its current loose location to the final location (`loose-2` locations and `tightened-2` may be considered as states of nuts and locatables).
    
    4. To estimate path lengths, use the graph defined by `(loose ?loc1 ?loc2)` as undirected edges between locatables.

    5. The kernel of the cost is approximated by:
       - Number of hops to propagate the tightened nut from current locatable to target locatable.
       - Additional cost to relocate the loose nut between locations if needed.
       - One action per tightening switch along the path.
       - One action per walking/spanner pickup movement (+ additional actions if tightening must move across locations).
    
    6. Sum estimated costs over all goal tightens not yet achieved.

    7. Return 0 if all goals are met.

    This heuristic is designed to be efficiently computable and to guide the search by estimating how difficult it is to move tightened nuts to the right locatables and locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static structures and goals.

        - Parse static facts to build the graph of locatables connected by `loose` relations.
        - Extract goal `(tightened-2 locatable location)` atoms.
        """
        self.goals = task.goals

        # Build adjacency graph of locatables from static facts (undirected graph of loose edges)
        self.loose_adj = {}
        for fact in task.static:
            if match(fact, "loose", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.loose_adj.setdefault(loc1, set()).add(loc2)
                self.loose_adj.setdefault(loc2, set()).add(loc1)

        # Map to quickly check which locatables exist: derive from loose adjacency keys
        self.locatables = set(self.loose_adj.keys())
        # Some locatables might appear only as keys in predicates like 'link' or 'carrying' - optionally add them:
        for fact in task.static:
            if match(fact, "link", "*"):
                _, loc = get_parts(fact)
                self.locatables.add(loc)

        # Extract goal tightens as a dict: locatable -> location
        self.goal_tightened_2 = {}
        for goal in self.goals:
            if match(goal, "tightened-2", "*", "*"):
                _, locatable, loc = get_parts(goal)
                self.goal_tightened_2[locatable] = loc

    def __call__(self, node):
        """
        Compute heuristic estimate based on how many actions are roughly needed to fulfill all `(tightened-2 locatable location)` goals.

        The heuristic tries to estimate:
        - How far is each locatable (with its current tightened nut) from its goal location.
        - How many move/tighten actions are needed to propagate tightened nuts along locatables graph.
        - How many loose-to-tighten location moves must be done.

        Returns 0 if all `(tightened-2 locatable location)` goals are satisfied.
        """
        state = node.state

        # Goal check quick return
        if self.goals <= state:
            return 0

        # Extract current tightened nuts on locatables: map nut->locatable
        # Also track locations of loose nuts: nut -> loose-2 location
        # We interpret "tightened ?nut ?locatable" as nut tightened on locatable
        tightened_on = {}
        loose2_location_of_nut = {}

        # Extract which locatables have link predicate (free spanners)
        link_locatables = set()
        for fact in state:
            if match(fact, "link", "*"):
                _, loc = get_parts(fact)
                link_locatables.add(loc)

        for fact in state:
            if match(fact, "tightened", "*", "*"):
                _, nut, locatable = get_parts(fact)
                tightened_on[nut] = locatable
            elif match(fact, "loose-2", "*", "*"):
                _, nut, loc = get_parts(fact)
                loose2_location_of_nut[nut] = loc

        # Extract current tightened-2 facts: locatable -> location
        tightened_2_on = {}
        for fact in state:
            if match(fact, "tightened-2", "*", "*"):
                _, locatable, loc = get_parts(fact)
                tightened_2_on[locatable] = loc

        # Extract all loose-2 locations for nuts (for potential movement estimation)
        # Extract all at-2 locations (where the agent/spanner may be)
        at_2_locations = set()
        for fact in state:
            if match(fact, "at-2", "*"):
                _, loc = get_parts(fact)
                at_2_locations.add(loc)

        # Function to find shortest path length (number of edges) between locatables using loose adjacency.
        # Use BFS, undirected graph.
        def shortest_path_length(start, goal):
            if start == goal:
                return 0
            visited = {start}
            queue = [(start, 0)]
            while queue:
                current, dist = queue.pop(0)
                for neighbor in self.loose_adj.get(current, []):
                    if neighbor == goal:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            # If no path found, return a large cost to avoid ignoring unreachable states.
            return 1000

        # Heuristic cost accumulator
        total_cost = 0

        # For each goal tightened-2 locatable and location:
        for locatable, goal_loc in self.goal_tightened_2.items():
            # If already satisfied, no cost for this goal.
            if locatable in tightened_2_on and tightened_2_on[locatable] == goal_loc:
                continue

            # We want to estimate cost to get tightened-2 locatable at goal_loc.

            # Step 1: Identify the nut that should be tightened on this locatable.
            # Since domain ties nuts to locatables via tightened predicate, find nut tightened on locatable:
            # But possibly the nut is tightened on a different locatable now.

            # Find nut associated with this locatable in goal:
            # We try to find the nut currently tightened associated with this locatable in goal (if multiple nuts, we estimate for each).

            # Nuts tightened anywhere may be spread, so we try to find nut that belongs here:
            nuts_for_this_locatable = [
                nut for nut, loc in tightened_on.items()
                if loc == locatable
            ]
            # If no nut currently tightened on the locatable, use goal nut candidates: any nut loose at some location?
            # Since goal predicates lock locatables not nuts, we conservatively try all nuts.

            # But heuristic can assume one nut corresponds to one locatable (observed from domain).
            # So we try to find nut tightened anywhere and assume this corresponds to this locatable's nut.
            # To be safe, consider nuts that appear in goals as loose-2 too.

            # We'll guess nut by matching loosened nut located at same location or tightened somewhere

            # Find nut associated loosely or tightened:
            # Look first for loose-2 nuts at goal location: meaning nut must be moved there.
            candidate_nuts = []
            for fact in state:
                if match(fact, "loose-2", "*", goal_loc):
                    _, nut, _ = get_parts(fact)
                    candidate_nuts.append(nut)
            if not candidate_nuts:
                # fallback: all nuts in tightened_on
                candidate_nuts = list(tightened_on.keys())
            if not candidate_nuts:
                # no nuts at all, potentially goal cannot be fulfilled; return large
                total_cost += 20
                continue

            # Select best nut candidate: the one minimizing move cost
            best_cost = 10000
            for nut in candidate_nuts:
                # Current tightened locatable of nut:
                nut_tightened_loc = tightened_on.get(nut)
                nut_loose_loc = loose2_location_of_nut.get(nut)

                ### Rough cost computation ###

                # Needed: Move nut from nut_tightened_loc (or loose location) to goal locatable at goal_loc

                # Cost1: If nut tightened on some locatable different from locatable target:
                # Move tightened nut across locatables (via loose edges)

                cost_move_tightened = 0
                if nut_tightened_loc and nut_tightened_loc != locatable:
                    cost_move_tightened = shortest_path_length(nut_tightened_loc, locatable)
                elif not nut_tightened_loc:
                    # If nut not tightened anywhere, must first tighten on some locatable
                    cost_move_tightened = 1  # One tightening action assumed minimal

                # Cost2: Move loose nut from its loose-2 location to goal location (loose-2 loc)
                # If loose location differs from goal location for tightened-2 predicate => must move the loose nut.
                cost_move_loose = 0
                if nut_loose_loc and nut_loose_loc != goal_loc:
                    # Assume 1 action per location move (tighten_nut) for each location difference.
                    # We don't have full location connectivity graph but locations are either location1 or location2 typically.
                    cost_move_loose = 1

                # Cost3: Final tightening of nut on locatable at goal location
                cost_final_tighten = 1

                # Cost4: Moving spanners/link transfers (approximate by counting link changes needed)
                # This is domain-specific; as a rough estimate add cost proportional to cost_move_tightened
                cost_link_moves = cost_move_tightened

                # Sum approximate cost for this nut:
                rough_cost = cost_move_tightened + cost_move_loose + cost_final_tighten + cost_link_moves

                if rough_cost < best_cost:
                    best_cost = rough_cost

            total_cost += best_cost

        return total_cost
