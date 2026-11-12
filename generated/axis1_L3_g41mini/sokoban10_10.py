from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(boarded passenger28)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (miconic).

    Summary:
    This heuristic estimates the number of actions required to achieve the goal
    of repositioning the lift to a specific passenger (the origin fact's passenger)
    while picking up and offloading passengers appropriately.

    Assumptions:
    - The goal is always expressed as (origin ?passenger-2 ?passenger).
      The heuristic estimates the cost to achieve this exact origin condition.
    - Passengers "boarded" are currently on the lift.
    - The lift can move between passengers through "served" relations representing floors.
    - The heuristic aims to estimate the number of actions (depart, up) needed, 
      focusing on how many passengers must be moved (boarded and/or shifted) to achieve the goal origin passenger and lift location.
    - The heuristic does NOT guarantee admissibility but aims to be informative and efficiently computed.

    Heuristic Initialization:
    - Extract the single origin goal passenger-2 and passenger from the goal facts.
    - Parse all "served" static facts to build a connectivity graph between passengers and floors.
    - Build a mapping from passenger to served-connected passengers and floors to represent adjacency (based on "served" predicates).
    - This graph structure helps estimate how many "up" moves are needed to move the lift with passengers to the target passenger.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the goal origin pair (passenger-2 and passenger) from the goal.
    2. Calculate the current lift positions, boarded passengers, and current origin pairs.
    3. The heuristic components:
       a) If the goal origin is already true, heuristic = 0.
       b) Otherwise, estimate the minimal number of actions as follows:
          - Count how many passengers currently boarded must be departed before the correct boarding occurs.
          - Estimate the number of "up" actions needed to move the lift from its current location(s) to the goal location (passenger).
            Use BFS on the passenger graph derived from "served" relations.
          - Possibly count 1 action to board the goal origin passenger if not boarded yet.
          - The heuristic sums approximate minimal required "depart", "up", and "board" steps.
    4. The heuristic is capped at least by the number of boarded passengers that need departure + moves to goal passenger.
    5. Returns a finite integer heuristic; never zero except in goal states.

    The heuristic is designed to guide greedy best-first search effectively by 
    favoring states closer in terms of passenger lift boarding/departing and lift position to the goal.
    """

    def __init__(self, task):
        # Extract goal origin passenger-2 and passenger.
        self.goal_origin_passenger2 = None
        self.goal_origin_passenger = None
        for goal_fact in task.goals:
            if match(goal_fact, "origin", "*", "*"):
                parts = get_parts(goal_fact)
                # parts = ["origin", passenger-2, passenger]
                self.goal_origin_passenger2 = parts[1]
                self.goal_origin_passenger = parts[2]
                break

        # Parse all served relations for adjacency between passengers and floors.
        # Build a graph of passenger <-> passenger, passenger <-> floor connections.
        self.adjacency = dict()  # key: node (passenger or floor), value: set of neighbors

        def add_edge(a, b):
            if a not in self.adjacency:
                self.adjacency[a] = set()
            if b not in self.adjacency:
                self.adjacency[b] = set()
            self.adjacency[a].add(b)
            self.adjacency[b].add(a)

        for fact in task.static:
            if match(fact, "served", "*", "*", "*"):
                _, p1, p2, floor = get_parts(fact)
                # Add edges: p1 <-> p2, p1 <-> floor, p2 <-> floor
                add_edge(p1, p2)
                add_edge(p1, floor)
                add_edge(p2, floor)

    def _bfs_shortest_distance(self, sources, target):
        """
        Compute shortest undirected distance in the adjacency graph
        from any node in sources set to the target node.
        Returns integer distance or a large number if unreachable.
        """
        if target is None or target == "":
            return 0
        if target in sources:
            return 0

        visited = set(sources)
        fringe = list(sources)
        dist = 0

        while fringe:
            next_fringe = []
            dist += 1
            for node in fringe:
                neighbors = self.adjacency.get(node, set())
                for nbr in neighbors:
                    if nbr == target:
                        return dist
                    if nbr not in visited:
                        visited.add(nbr)
                        next_fringe.append(nbr)
            fringe = next_fringe

        # Target unreachable
        return 1000  # Large heuristic penalty for unreachable states

    def __call__(self, node):
        state = node.state

        # 0) If goal reached, heuristic = 0.
        goal_origin_fact = f"(origin {self.goal_origin_passenger2} {self.goal_origin_passenger})"
        if goal_origin_fact in state:
            return 0

        # 1) Identify facts in current state:
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        lift_positions = {get_parts(fact)[1] for fact in state if match(fact, "lift-at", "*")}
        origin_facts = [fact for fact in state if match(fact, "origin", "*", "*")]

        # 2) Find current origin passenger, if any
        current_origin = None
        for fact in origin_facts:
            parts = get_parts(fact)
            # parts = ["origin", passenger-2, passenger]
            if parts[1] == self.goal_origin_passenger2:
                current_origin = parts[2]
                break

        # 3) Heuristic components:

        # a) Number of boarded passengers different from the goal origin passenger boarded:
        # Generally, we must depart all boarded passengers that are not the goal passenger
        # because only the goal passenger should be boarded at the end for the goal origin to hold.
        boarded_except_goal = boarded_passengers - {self.goal_origin_passenger}

        # departing each wrong boarded passenger requires at least one 'depart' action
        depart_cost = len(boarded_except_goal)

        # b) If the goal origin passenger is not boarded, we will need at least one 'up' or 'board' action 
        # to get that passenger on the lift.
        board_cost = 0 if self.goal_origin_passenger in boarded_passengers else 1

        # c) We need the lift at the goal origin passenger location.
        # Lift may currently be at multiple passengers (multiple 'lift-at').
        # Compute shortest distance from any current lift position to goal passenger in adjacency graph.
        if not lift_positions:
            # No lift position? Heuristic penalty
            move_cost = 10
        else:
            move_cost = self._bfs_shortest_distance(lift_positions, self.goal_origin_passenger)

        # d) Additionally, if the goal origin passenger-2 is not explicitly connected to the goal passenger's lift,
        # this indicates extra moves, but the domain goal only requires origin fact.

        # e) Consider we may need one final 'depart' action to complete goal if origin passenger is currently boarded.
        # Actually, the goal origin itself does NOT require that passenger to be boarded, so no extra depart counted here.

        # Combine heuristic components:
        # - depart_cost: number of passengers that must be departed to fix boarding.
        # - move_cost: moves (ups) to get lift to goal passenger.
        # - board_cost: possibly board goal passenger if not on lift yet.
        heuristic_value = depart_cost + move_cost + board_cost

        # Return at least 1 if heuristic otherwise zero (except at goal) to avoid zero heuristic in non-goal states.
        return max(heuristic_value, 1)
