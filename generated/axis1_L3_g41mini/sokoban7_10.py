from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(boarded passenger1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (miconic) domain.

    Summary:
    This heuristic estimates the number of required actions to reach the goal where the lift's origin (floor or passenger) must be set correctly. 
    The domain is about passengers boarding and departing lifts located at different passengers or floors.
    The goal typically states a specific (origin <passenger-2> <passenger>) fact.
    To reach the goal, the lift must "carry" passengers (boarded) and travel among them (changing lift-at facts), basically walking through a chain of served relations.

    Assumptions:
    - The goal always involves one or more origin facts to be true.
    - A passenger can be boarded or not.
    - The lift can be at different passengers.
    - The "served" fact represents connectivity between passengers and floors (like edges in a graph).
    - Each "up" or "depart" action changes boarding or lift location along the served edges.
    - The cost estimate is based on how far the current lift position and boarded passengers are from achieving the goal origin fact(s).
    - The heuristic is not required to be admissible, but should guide the search efficiently.

    Heuristic Initialization:
    - Extract the goal origin facts, i.e. pairs (passenger-2, passenger) that must hold.
    - Extract the static 'served' relations seen as edges of a graph: from passenger to passenger or passenger to floor.
      These define valid "moves" for the lift (depart/up actions).
    - Build adjacency mappings from passenger to reachable passengers/floors according to served facts to facilitate distance computations.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the goal origin predicate (origin <passenger-2> <passenger>) to find goal target passenger pairs.
    2. Identify current boarded passenger(s). If none is boarded, the lift can only move among lift-at positions.
    3. Identify current lift positions (via lift-at facts). Note that multiple lift-at facts may exist simultaneously.
    4. Our main challenge is to connect the current situation to the goal origin facts using a minimal sequence of actions:
       - Depart removes the boarded passenger and moves lift-at.
       - Up moves the lift and changes boarded passengers along a served chain.
    5. Because served facts represent adjacency in a graph, we use BFS to estimate minimum edge/hop distances.
       - From current boarded passenger(s) to the goal passenger(s).
       - From current lift-at passengers to the goal passenger(s).
    6. The heuristic estimates:
       - If the goal origin already holds in the state, heuristic = 0.
       - Else, estimate minimum number of actions to get from current state to goal:
         - If a passenger is boarded but does not match goal origin passenger: 
           estimate how many depart/up steps to reach the goal passenger (distance in the served graph).
         - If no passenger boarded, estimate steps to get lift and boarded passenger(s) to the goal position.
    7. The cost contribution:
       - Boarding/unboarding typically costs one or two actions (depart/up).
       - Each graph edge traversal estimates one action.
    8. Combine the above information into a numeric heuristic value.

    This heuristic focuses on the shortest graph hop distance to the target passenger(s) involved in the goal origin and the boarding state, giving a good domain-dependent estimate.

    """

    def __init__(self, task):
        # Extract goal origin pairs (passenger-2, passenger) from goal facts.
        self.goal_origins = set()
        for fact in task.goals:
            parts = get_parts(fact)
            if parts[0] == "origin" and len(parts) == 3:
                self.goal_origins.add((parts[1], parts[2]))

        # Extract static "served" relations defining the connectivity graph.
        # We build adjacency maps passenger->passenger and also consider floors as intermediate nodes.
        # Because "served" has signature (served ?v3 - passenger ?v4 - passenger ?v5 - floor)
        # We construct adjacency in both directions passenger<->passenger.
        self.adjacency = {}  # Maps passenger -> set of connected passengers
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) == 4:
                p_from = parts[1]  # passenger
                p_to = parts[2]    # passenger
                # floors are parts[3], unused here except maybe for extra info (but not needed for adjacency)

                if p_from not in self.adjacency:
                    self.adjacency[p_from] = set()
                if p_to not in self.adjacency:
                    self.adjacency[p_to] = set()
                # Add bi-directional edges
                self.adjacency[p_from].add(p_to)
                self.adjacency[p_to].add(p_from)

        # It's possible some passengers have no adjacency, ensure they exist with empty set
        # for stable lookups.
        # Also add passengers from goal origins that may not appear in served? Just in case.
        for p2, p1 in self.goal_origins:
            if p1 not in self.adjacency:
                self.adjacency[p1] = set()
            if p2 not in self.adjacency:
                self.adjacency[p2] = set()

    def _bfs_distance(self, starts, targets):
        """
        BFS helper to find shortest distance in adjacency graph from any node in starts to any node in targets.
        Returns the minimal number of edges (hops).
        If no path, returns None.
        """
        if not starts:
            return None
        targets = set(targets)
        visited = set()
        frontier = list(starts)
        distance = 0

        # If any start node already in targets, distance = 0
        if any(node in targets for node in frontier):
            return 0

        while frontier:
            next_frontier = []
            for node in frontier:
                visited.add(node)
                neighbors = self.adjacency.get(node, set())
                for nbr in neighbors:
                    if nbr in targets:
                        return distance + 1
                    if nbr not in visited:
                        next_frontier.append(nbr)
            frontier = next_frontier
            distance += 1

        return None

    def __call__(self, node):
        state = node.state

        # 1) Check if goal is already reached:
        # Goal requires all origin facts to hold.
        if all(f"(origin {p2} {p1})" in state for (p2, p1) in self.goal_origins):
            return 0

        # 2) Get current boarded passengers (may be zero or one).
        boarded = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        # 3) Get current lift-at passengers (positions).
        lift_at = {get_parts(fact)[1] for fact in state if match(fact, "lift-at", "*")}

        # 4) Goal targets: the passenger(s) in goal origins as destination passenger (second element)
        goal_passengers = {p1 for _, p1 in self.goal_origins}

        # If no lift-at positions or no passengers in graph, return a default non-zero heuristic.
        # Defensive fallback.
        if not lift_at:
            # No known lift position, assume minimal cost 1 (cannot be zero here)
            return 1

        # 5) Estimate distances:
        # If there is a boarded passenger:
        #  - Likely needs to be moved (depart/up) towards goal passenger.
        # If no boarded passengers:
        #  - Lift must move to a passenger and board to reach the origin goal.

        # Distance from boarded passenger(s) to the goal passenger(s).
        dist_boarded_to_goal = None
        if boarded:
            dist_boarded_to_goal = self._bfs_distance(boarded, goal_passengers)

        # Distance from lift position(s) to goal passenger(s).
        dist_lift_to_goal = self._bfs_distance(lift_at, goal_passengers)

        # Distance from lift position(s) to boarded passenger(s).
        # This is relevant when boarded passenger is missing or to estimate how to pick up boarded passenger again.
        dist_lift_to_boarded = None
        if boarded:
            dist_lift_to_boarded = self._bfs_distance(lift_at, boarded)

        # 6) Heuristic value composition:

        # If we have boarded passenger(s) but distance to goal is unknown (no path), assign a higher penalty.
        if boarded and dist_boarded_to_goal is None:
            dist_boarded_to_goal = 10_000  # large penalty (disconnected)

        # If no boarded passenger, we will need at least one boarding action plus moving to goal.
        if not boarded:
            # Estimated cost: 
            # - move lift from current position to a passenger position (assume lift_at are passenger nodes)
            # - board that passenger (cost 1 action)
            # - move to goal passenger (distance estimate)
            # - finally set origin

            # dist_lift_to_goal hopefully > 0, if unknown then fallback to penalty
            if dist_lift_to_goal is None:
                dist_lift_to_goal = 10_000
            
            # +1 for boarding +1 for setting origin (depart or up)
            heuristic = dist_lift_to_goal + 2
            return heuristic

        # If boarded passenger(s) exist:
        # We can estimate cost as distance boarded->goal passenger + 1 to depart/set origin.
        # Also consider distance lift->boarded passenger to adjust if lift needs to relocate.
        # The elevator might need to move from lift-at to boarded before departing.

        if dist_lift_to_boarded is None:
            dist_lift_to_boarded = 0  # if unknown, consider zero (maybe lift is already there)

        if dist_boarded_to_goal is None:
            dist_boarded_to_goal = 10_000  # disconnected penalty

        # Compose heuristic with some weight:
        # Distance from lift to boarded + distance from boarded to goal + 1 action to depart/set origin
        heuristic = dist_lift_to_boarded + dist_boarded_to_goal + 1

        return heuristic
