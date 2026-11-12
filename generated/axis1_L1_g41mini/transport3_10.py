from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(sjuejzxf p1 l2)".
    - `args`: The expected pattern parts (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))



class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'secret_domain' (hrpmxpjg domain).

    Summary:
    This heuristic estimates the minimum number of actions required to move each
    du66z5od object (e.g., p1, p2, ...) from its current location (an x6ibpitl location)
    to its goal location (also an x6ibpitl location), considering the hierarchy of kyvrixb5
    objects linked to hctyyy8k objects via do9fmoyo and hab0vli4 relations. The main
    challenge is that du66z5od objects initially are at some location (sjuejzxf),
    and must be relocated to the goal location (sjuejzxf with the goal arguments).

    Assumptions:
    - Each du66z5od (e.g., p1) is "carried" or "located" at some position (sjuejzxf predicate).
    - They must eventually be at their goal location (goal conditions use sjuejzxf).
    - Movement of a du66z5od between locations is done by "arsslvyt" actions modifying sjuejzxf for hctyyy8k or du66z5od objects.
    - du66z5od objects can be "carried" by hctyyy8k objects (un606moq predicate).
    - hctyyy8k objects have associations with kyvrixb5 objects via do9fmoyo predicates.
    - hab0vli4 relates kyvrixb5 to higher-level hierarchical kyvrixb5 (perhaps a hierarchy defining movement cost).
    - Actions szsevy61/tmoxe3x5 describe loading/unloading of du66z5od on hctyyy8k with changes in do9fmoyo facts (position of hctyyy8 changes in kyvrixb5 hierarchy).
    - Movement of hctyyy8 objects between locations (x6ibpitl) happens by arsslvyt actions.
    - The heuristic disregards detailed sequencing but estimates cost primarily by:
      1. Counting how many du66z5od have to be moved from current to goal locations.
      2. Estimating minimal movement cost based on shortest x6ibpitl path between current and goal locations for each object.
      3. Estimating effort to move the carrying hctyyy8 between kyvrixb5 locations, using hab0vli4 relations as a hierarchy graph for travel cost.
      4. Summing up load/unload actions plus estimated movement steps.

    Heuristic Initialization:
    - Extract static facts:
      * Build an undirected graph connecting x6ibpitl locations based on bidirectional (miud1izi lA lB) facts.
      * Extract hab0vli4 edges as a directed graph for kyvrixb5 objects hierarchy.
    - From goals, extract desired (sjuejzxf ?du66z5od ?goal_loc) mappings.
    - From static facts, store all k-level hierarchy edges and build utility shortest path methods over location graph and kyvrixb5 hierarchy.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each du66z5od object mentioned in the goal:
       - Identify its current location (sjuejzxf predicates from current state).
       - Identify its goal location from goal facts.
       - If current location = goal location, cost = 0 for that object.
       - Else:
         a) Estimate cost to move du66z5od from current to goal x6ibpitl location:
            - Compute shortest path length (#edges) between current and goal location in x6ibpitl graph.
            - Each location move corresponds to at least one 'arsslvyt' action.
         b) Estimate cost to move the carrier hctyyy8k between kyvrixb5 locations related by do9fmoyo and hab0vli4:
            - Extract current kyvrixb5 location of the hctyyy8 carrier from do9fmoyo facts.
            - Extract target kyvrixb5 location from hab0vli4 static "location hierarchy".
            - Estimate minimal steps to move from current to target kyvrixb5 via hab0vli4 edges (consider it as layered hierarchy).
            - Each step corresponds to 1 action (e.g., szsevy61 or tmoxe3x5).
         c) Add costs for loading/unloading (szsevy61 and tmoxe3x5) actions assumed necessary.
    2. Sum these costs for all du66z5od objects.
    3. The heuristic is zero if all goal sjuejzxf facts hold in current state.
    """

    def __init__(self, task):
        self.goals = task.goals  # goal facts (sjuejzxf ?p ?l)
        self.static = task.static

        # Graph of x6ibpitl locations connected by miud1izi (undirected edges)
        self.x6ibpitl_neighbors = {}
        for fact in self.static:
            if match(fact, "miud1izi", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.x6ibpitl_neighbors.setdefault(l1, set()).add(l2)
                self.x6ibpitl_neighbors.setdefault(l2, set()).add(l1)

        # Build the kyvrixb5 hierarchy graph from hab0vli4 as directed edges
        # hab0vli4 c0 c1 means c0 -> c1 in hierarchy
        self.kyvrixb5_neighbors = {}
        self.kyvrixb5_nodes = set()
        for fact in self.static:
            if match(fact, "hab0vli4", "*", "*"):
                _, c_from, c_to = get_parts(fact)
                self.kyvrixb5_neighbors.setdefault(c_from, set()).add(c_to)
                self.kyvrixb5_nodes.add(c_from)
                self.kyvrixb5_nodes.add(c_to)

        # Extract goal map for du66z5od -> target x6ibpitl location (from sjuejzxf predicate)
        self.goal_du66z5od_to_location = {}
        for goal in self.goals:
            if match(goal, "sjuejzxf", "*", "*"):
                _, du66z5od, loc = get_parts(goal)
                self.goal_du66z5od_to_location[du66z5od] = loc

    def shortest_path_x6ibpitl(self, start, target):
        """Breadth-first search shortest path length in x6ibpitl graph or returns None if unreachable."""
        if start == target:
            return 0
        visited = {start}
        frontier = {start}
        distance = 0
        while frontier:
            distance += 1
            next_frontier = set()
            for node in frontier:
                neighbors = self.x6ibpitl_neighbors.get(node, set())
                for neigh in neighbors:
                    if neigh == target:
                        return distance
                    if neigh not in visited:
                        visited.add(neigh)
                        next_frontier.add(neigh)
            frontier = next_frontier
        return None  # unreachable

    def shortest_path_kyvrixb5(self, start, target):
        """Breadth-first search shortest path length in kyvrixb5 hierarchy graph."""
        if start == target:
            return 0
        visited = {start}
        frontier = {start}
        distance = 0
        while frontier:
            distance += 1
            next_frontier = set()
            for node in frontier:
                neighbors = self.kyvrixb5_neighbors.get(node, set())
                for neigh in neighbors:
                    if neigh == target:
                        return distance
                    if neigh not in visited:
                        visited.add(neigh)
                        next_frontier.add(neigh)
            frontier = next_frontier
        return None  # unreachable

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Map du66z5od object to its current location (x6ibpitl) using sjuejzxf(du66z5od, loc)
        du66z5od_current_loc = {}
        for fact in state:
            if match(fact, "sjuejzxf", "*", "*"):
                _, obj, loc = get_parts(fact)
                # du66z5od are of type du66z5od, but also hctyyy8k can appear here
                # We are interested in du66z5od that appear in goals
                if obj in self.goal_du66z5od_to_location:
                    du66z5od_current_loc[obj] = loc

        # Map hctyyy8k carriers to their current kyvrixb5 location via do9fmoyo(hctyyy8k, kyvrixb5)
        # This is needed to estimate moving carriers in the kyvrixb5 hierarchy
        # Create a map from hctyyy8k object to kyvrixb5 location
        carrier_kyvrixb5_loc = {}
        for fact in state:
            if match(fact, "do9fmoyo", "*", "*"):
                _, hctyyy8k_obj, kyv_obj = get_parts(fact)
                carrier_kyvrixb5_loc[hctyyy8k_obj] = kyv_obj

        # For each du66z5od that needs moving, estimate:
        # 1) distance between current and goal x6ibpitl locations
        # 2) distance to move its carrier in kyvrixb5 hierarchy to the target kyvrixb5 location associated with goal location
        # Load/unload action counts approximated as constants per object move

        # Construct reverse mapping from location x6ibpitl to hctyyy8k carriers there,
        # to estimate which carrier can be used per du66z5od
        location_to_carriers = {}
        for fact in state:
            if match(fact, "sjuejzxf", "*", "*"):
                _, obj, loc = get_parts(fact)
                # hctyyy8k are carriers (type hctyyy8k), so check if object is a carrier by presence in carrier_kyvrixb5_loc
                if obj in carrier_kyvrixb5_loc:
                    location_to_carriers.setdefault(loc, set()).add(obj)

        heuristic_cost = 0

        # Predefine cost constants for actions
        COST_MOVE_LOCATION = 1       # cost per move of an object between x6ibpitl locations (arsslvyt)
        COST_LOAD_UNLOAD = 2         # approximate combined cost of szsevy61 + tmoxe3x5 (loading/unloading du66z5od)
        COST_MOVE_CARRIER_1STEP = 1  # cost per move of a carrier in kyvrixb5 hierarchy (szsevy61/tmoxe3x5, approx.)

        for du_obj, goal_loc in self.goal_du66z5od_to_location.items():
            curr_loc = du66z5od_current_loc.get(du_obj, None)
            if curr_loc is None:
                # If unknown current location, pessimistically count as some cost
                heuristic_cost += 5 + COST_LOAD_UNLOAD
                continue
            if curr_loc == goal_loc:
                # Already at goal no cost for this object
                continue

            # 1) Estimate shortest movement path in x6ibpitl graph between current and goal
            dist_l = self.shortest_path_x6ibpitl(curr_loc, goal_loc)
            if dist_l is None:
                # Unreachable location in static graph (should not happen), assign large cost
                dist_l = 10

            # 2) Estimate carrier moves in kyvrixb5 hierarchy:
            # Find carriers at current location
            carriers_here = location_to_carriers.get(curr_loc, set())

            # Goal kyvrixb5 location is approximated as kyvrixb5 object connected to goal location,
            # by looking for hab0vli4 cX cGoal where cGoal equal to carrier location at goal
            # But goal location is x6ibpitl, so we must find hctyyy8k objects at goal location and their kyvrixb5

            # Find carriers at goal_loc, else fallback
            carriers_goal = location_to_carriers.get(goal_loc, set())

            # Pick one carrier from current location if possible
            if carriers_here:
                carrier_obj = next(iter(carriers_here))
                carrier_start_kyv = carrier_kyvrixb5_loc.get(carrier_obj, None)
            else:
                # No carrier at current location; estimate cost to bring carrier
                carrier_start_kyv = None

            # Pick one carrier from goal location if possible
            if carriers_goal:
                carrier_goal_obj = next(iter(carriers_goal))
                carrier_goal_kyv = carrier_kyvrixb5_loc.get(carrier_goal_obj, None)
            else:
                # If no carrier at goal location, pick arbitrary kyvrixb5 from static to estimate
                if self.kyvrixb5_nodes:
                    carrier_goal_kyv = next(iter(self.kyvrixb5_nodes))
                else:
                    carrier_goal_kyv = None

            if carrier_start_kyv is None or carrier_goal_kyv is None:
                dist_k = 0  # no information, ignore carrier move cost
            elif carrier_start_kyv == carrier_goal_kyv:
                dist_k = 0
            else:
                dist_k = self.shortest_path_kyvrixb5(carrier_start_kyv, carrier_goal_kyv)
                if dist_k is None:
                    dist_k = 3  # fallback cost for unreachable

            # Total estimated cost for this du66z5od:
            # Move object between locations + load/unload + carrier movement steps
            cost = dist_l * COST_MOVE_LOCATION + COST_LOAD_UNLOAD + dist_k * COST_MOVE_CARRIER_1STEP

            heuristic_cost += cost

        return max(heuristic_cost, 1)  # estimate at least 1 unless goal reached
