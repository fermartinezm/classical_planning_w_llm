from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(u4srx1nb bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (e2zu8y93) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state from
    a given state in the secret_domain. The domain involves moving 'ilfxqhcc' objects (like
    spanners) through a chain of locations ('i6azntdy' type), assembling or using 'rupjajag'
    objects (nuts), with the ultimate goal of activating certain 'rupjajag' objects
    signaled by the (g59iwyxf ?nut) predicate.

    Assumptions
    -----------
    - The goal facts are always of the form (g59iwyxf ?nut), meaning each required nut
      must be in the 'activated' state.
    - The main actions to achieve this are:
      1. Moving an 'ilfxqhcc' object (e.g. spanner) along locations (operators n9sbfmrs).
      2. Using an 'ilfxqhcc' at a location where a 'rupjajag' (nut) is present to create a
         yud7t3vb relation (moi6wnfu), marking some intermediate state.
      3. Finally, applying kjsqfni8 action to activate the nuts (g59iwyxf) by consuming
         the (g0brvc06) spanner, the yud7t3vb relation for the ilfxqhcc and e2zu8y93, and
         the nut being in an appropriate state.
    - The domain enforces nut activation only after proper conditions are met.
    - The locations form a connectivity graph given by kxokb5hx facts.
    - The heuristic does not attempt to be admissible but tries to be informative and
      efficiently computable.

    Heuristic Initialization
    ------------------------
    - Extract all connectivity relations between locations (kxokb5hx) as an undirected graph.
    - Extract the set of all nuts that must be activated (from the goal).
    - Extract which nuts are already activated in the current state.
    - Extract which spanners (e2zu8y93) currently have the (g0brvc06) predicate set (signalling they are ready to be used).
    - Extract the current positions of:
      * each 'ilfxqhcc' (e.g., bob) — their exact location (i6azntdy).
      * each spanner (e2zu8y93) — their exact location (i6azntdy).
      * each nut (rupjajag) — their exact location (i6azntdy).
    - Extract which (yud7t3vb ?ilfxqhcc ?e2zu8y93) relations currently hold.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. For each nut required by the goal but not yet activated:
       a) Identify the nut's current location and whether it still has the (bu1om0m1) flag,
          indicating it needs processing.
       b) Check which spanners are ready (g0brvc06) and their positions.
       c) For each required spanner involved, estimate the minimal cost to:
          - Move it to the nut's location.
          - Move the required ilfxqhcc (bob) to the spanner's location or vice versa to enable
            the (moi6wnfu) action that creates the (yud7t3vb) predicate.
          - Apply the final activation action (kjsqfni8).
    2. Estimate distances between locations using shortest paths on the location graph.
       These distances reflect how many move actions (n9sbfmrs) are needed to transport
       spanners and ilfxqhcc objects.
    3. Sum the estimated costs for all nuts to be activated.
    4. Adjust the heuristic cost based on missing intermediate predicates like (yud7t3vb).

    Overall, the heuristic's main driver is the transport cost of spanners between locations,
    moving ilfxqhcc to spanners, and applying the activation action. It approximates the number
    of moves plus additional action costs (uses of moi6wnfu and kjsqfni8) in a compact manner.

    The heuristic returns 0 iff all nut goals (g59iwyxf) are achieved.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract graph connectivity: build adjacency list for locations (i6azntdy)
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "kxokb5hx", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:3]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Extract goal nuts: nuts that must be activated (have g59iwyxf)
        self.goal_nuts = set()
        for fact in self.goals:
            if match(fact, "g59iwyxf", "*"):
                nut = get_parts(fact)[1]
                self.goal_nuts.add(nut)

    def shortest_path_length(self, start, goal):
        """
        Compute shortest path length between two locations using BFS on the location graph.
        Returns a positive integer number of edges, or a large number if unreachable.
        """
        if start == goal:
            return 0
        visited = {start}
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            for neighbor in self.location_graph.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, dist + 1))
        # If no path found, return a large number
        return 1000

    def __call__(self, node):
        state = node.state

        # Quick check: if all goals achieved, heuristic=0
        goal_achieved = all(goal in state for goal in self.goals)
        if goal_achieved:
            return 0

        # Extract currently activated nuts (g59iwyxf)
        activated_nuts = {
            get_parts(fact)[1]
            for fact in state if match(fact, "g59iwyxf", "*")
        }

        # Nuts needing activation = goal nuts - activated nuts
        nuts_to_activate = self.goal_nuts - activated_nuts
        if not nuts_to_activate:
            return 0

        # Extract which nuts still need processing (have bu1om0m1 flag)
        nuts_need_processing = {
            get_parts(fact)[1] for fact in state if match(fact, "bu1om0m1", "*")
        }

        # Extract positions of nuts (u4srx1nb ?nut ?location)
        nut_positions = {}
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                obj, loc = get_parts(fact)[1:3]
                # Identify if obj is a nut by checking type name pattern
                # From domain: nuts are of type rupjajag
                # We'll rely on goal nuts to identify nut objects here
                if obj in self.goal_nuts or obj in nuts_need_processing:
                    nut_positions[obj] = loc

        # Extract spanners that are currently ready (g0brvc06)
        spanners_ready = {
            get_parts(fact)[1] for fact in state if match(fact, "g0brvc06", "*")
        }

        # Extract positions of spanners (u4srx1nb ?spanner ?location)
        spanner_positions = {}
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                obj, loc = get_parts(fact)[1:3]
                if obj in spanners_ready or obj in spanner_positions:
                    spanner_positions[obj] = loc

        # Extract ilfxqhcc object(s) (like bob) and their positions
        ilfxqhcc_positions = {}
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                obj, loc = get_parts(fact)[1:3]
                # Identify ilfxqhcc objects: from domain "bob" is a common ilfxqhcc
                # We'll heuristically assume only one ilfxqhcc (bob) for simplification
                # because in examples only 'bob' is ilfxqhcc typed. If more appear, extend here.
                if obj == "bob":
                    ilfxqhcc_positions[obj] = loc

        # Extract existing yud7t3vb relations (yud7t3vb ?ilfxqhcc ?spanner)
        yud7t3vb_pairs = set()
        for fact in state:
            if match(fact, "yud7t3vb", "*", "*"):
                ilfx, sp = get_parts(fact)[1:3]
                yud7t3vb_pairs.add((ilfx, sp))

        # COST MODEL:
        # Actions required per nut to activate:
        # 1. Move a ready spanner (with g0brvc06) to nut location:
        #    cost proportional to shortest_path_length(spanner_pos, nut_pos)
        # 2. Move ilfxqhcc (bob) to spanner location if no yud7t3vb yet, then apply moi6wnfu:
        #    cost = shortest_path_length(bob_pos, spanner_pos) + 1 (moi6wnfu action)
        # 3. Apply kjsqfni8 action to activate the nut: cost = 1
        #
        # We'll pick the minimal over all spanners for each nut.

        # If bob position unknown, assign large cost
        if "bob" not in ilfxqhcc_positions:
            bob_pos = None
        else:
            bob_pos = ilfxqhcc_positions["bob"]

        total_cost = 0

        for nut in nuts_to_activate:
            # Skip if nut position unknown: assign high penalty
            if nut not in nut_positions:
                total_cost += 20  # penalty for unknown nut location
                continue

            nut_pos = nut_positions[nut]

            # Check if nut needs processing (bu1om0m1 present)
            need_processing = nut in nuts_need_processing

            # We consider only spanners that are ready to be used (g0brvc06)
            if not spanners_ready:
                # No available spanners means high cost
                total_cost += 20
                continue

            # For each spanner ready, estimate:
            best_cost_for_nut = 1000

            for spanner in spanners_ready:
                # spanner location
                if spanner not in spanner_positions:
                    continue
                spanner_pos = spanner_positions[spanner]

                # cost to move spanner to nut location
                move_spanner_cost = self.shortest_path_length(spanner_pos, nut_pos)

                # cost to move bob (ilfxqhcc) to spanner location if yud7t3vb missing
                has_yud7t3vb = ("bob", spanner) in yud7t3vb_pairs
                if bob_pos is None:
                    move_bob_cost = 10  # large penalty if bob position unknown
                elif has_yud7t3vb:
                    move_bob_cost = 0  # already has the relation
                else:
                    move_bob_cost = self.shortest_path_length(bob_pos, spanner_pos) + 1  # +1 for moi6wnfu action

                # final activation requires kjsqfni8 action (cost 1)
                activation_cost = 1

                # If nut needs processing (bu1om0m1) apply kjsqfni8 after moi6wnfu and moving
                # else assume nut already processed (probably should never happen if in nuts_to_activate)
                total_nut_cost = move_spanner_cost + move_bob_cost + activation_cost

                if total_nut_cost < best_cost_for_nut:
                    best_cost_for_nut = total_nut_cost

            # Accumulate minimal cost for this nut
            total_cost += best_cost_for_nut

        # Return the heuristic estimate as an integer
        return int(total_cost) if total_cost > 0 else 0
