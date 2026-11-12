from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(boarded passenger39)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (the miconic variant) domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state, where
    a particular origin relation must hold. The domain describes an elevator system represented
    by passengers, floors, and a "lift-at" predicate that marks the current location of lifts.
    The goal is that a specific "origin" predicate holds, indicating that a designated passenger-2
    object (floor proxy) and a passenger are linked.

    The heuristic is designed to guide a greedy best-first search by estimating how far the
    current state is from reaching the goal, based on how many passengers must be moved
    through the network of served floor relations and lifted to the origin passenger.

    Assumptions:
    - The goal requires a single (origin ?passenger-2 ?passenger) fact to be true.
    - The "served" predicate encodes a directed graph between passengers and floors,
      indicating valid transitions possibly restricting movement.
    - The lift can be at various passengers (standing proxy for floors).
    - The "boarded" predicate denotes the passenger currently in the lift.
    - Actions include lifting passengers "up" (boarding/unboarding along floors), and "depart"
      which changes the onboard passenger.
    - The static facts entail a connectivity graph to be used for navigating passengers to the goal.
    - States are solvable only if there is a path from passenger-2 in origin to the passenger in origin,
      following "served" relations.

    Heuristic Initialization:
    - Parse the static "served" facts to build:
      - A directed graph representing connections between passengers/floors.
    - Extract the goal origin relation:
      - Identify the target origin passenger-2 and passenger for the goal.
    - Precompute reverse lookup to quickly find neighbors for heuristic graph search.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current origin passenger-2 and passenger from the state's "origin" facts.
       These represent the current positions relevant for reaching the goal origin fact.

    2. If the current origin fact already matches the goal, return 0 (goal state).

    3. Use the static "served" graph edges to estimate the shortest "distance" from the current
       passenger-2 and passenger to the goal passenger-2 and passenger.

       - Interpret the "served" predicate as edges (served ?p1 ?p2 ?floor) meaning ?p1 leads to ?p2.
       - This lets us consider movement among passengers/floors through served relations.
    
    4. Since "lift-at" and "boarded" predicates define who holds the lift and who is in/on the lift,
       estimate the minimum number of "up" and "depart" actions needed:
       - Roughly, each transition in the "served" graph corresponds to an "up" action moving a passenger.
       - Depart actions correspond to swapping boarded passengers.

    5. The heuristic counts:
       - The graph distance from the current origin passenger to the goal origin passenger.
       - Plus an estimated cost if the lift is not at the right passenger or the passenger not boarded.

    6. Return the sum as the heuristic value.

    This heuristic is not guaranteed admissible but aims to reduce expanded nodes efficiently
    in greedy best-first search by giving a close estimate of transitions/actions needed.

    """

    def __init__(self, task):
        """
        Initialize by building a directed graph of served edges and extracting goal origin.

        :param task: The planning task instance.
        """
        self.goals = task.goals
        self.static = task.static

        # Build directed graph from served relations: key = passenger/floor, value = set of neighbors
        self.served_graph = dict()  # adjacency list: passenger/floor -> set of passengers/floors reachable
        for fact in self.static:
            if fact.startswith("(served "):
                parts = get_parts(fact)
                # served ?v3 ?v4 ?v5  => parts = ['served', v3, v4, v5]
                from_node = parts[1]
                to_node = parts[2]
                self.served_graph.setdefault(from_node, set()).add(to_node)

        # Extract the goal origin: (origin ?passenger-2 ?passenger)
        self.goal_origin = None
        for goal in self.goals:
            if goal.startswith("(origin "):
                parts = get_parts(goal)
                # (origin passenger50 passenger11)
                # parts = ['origin', passenger-2, passenger]
                self.goal_origin = (parts[1], parts[2])
                break

    def __call__(self, node):
        """
        Estimate steps (actions) left to reach goal state from the given node state.

        :param node: Search node containing the current state.
        :return: Estimated cost as an integer.
        """
        state = node.state

        # Extract current origin predicate
        current_origin = None
        for fact in state:
            if fact.startswith("(origin "):
                parts = get_parts(fact)
                current_origin = (parts[1], parts[2])
                break

        # Goal check: if current origin == goal origin => heuristic = 0
        if current_origin == self.goal_origin:
            return 0

        # Extract current 'boarded' passenger(s), typically at most one
        boarded_passengers = {get_parts(fact)[1] for fact in state if fact.startswith("(boarded ")}

        # Extract passengers where the lift is currently at
        lift_positions = {get_parts(fact)[1] for fact in state if fact.startswith("(lift-at ")}

        # For heuristic, we want to estimate shortest path (in terms of served edges) from current passenger-2 and passenger
        # to goal passenger-2 and passenger.
        #
        # The "origin" fact is (origin passenger-2 passenger), so we consider a pair (floor proxy, passenger),
        # and the goal is a particular such pair.
        #
        # By domain semantics, we want to find a path from current origin to goal origin on served edges,
        # where passengers are nodes and edges are served relations between them.
        #
        # We can do a BFS on pairs, from current_origin to goal_origin:
        # Each state is a pair (floor_proxy, passenger).
        #
        # Transitions correspond to applying served relations from floor_proxy or passenger to next reachable passengers.
        #
        # However, to keep heuristic efficient, we approximate by considering shortest distance from current passenger to goal passenger
        # in the served graph ignoring floor proxy changes.
        #
        # Additionally, consider lift position and boarded passenger to adjust heuristic.

        # To proceed:
        # 1) Compute shortest distance between current_origin passenger and goal_origin passenger with BFS in served graph
        # 2) Add costs for lift and boarded differences

        from collections import deque

        def bfs_distance(start, goal):
            if start == goal:
                return 0
            visited = set()
            queue = deque([(start, 0)])
            while queue:
                node, dist = queue.popleft()
                if node == goal:
                    return dist
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in self.served_graph.get(node, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
            # No path found, return a large cost (domain hopefully no dead ends)
            return 1000

        # Compute passenger distance in served graph from current to goal
        passenger_distance = bfs_distance(current_origin[1], self.goal_origin[1])

        # Compute floor proxy distance (between passenger-2 in origin)
        floor_distance = bfs_distance(current_origin[0], self.goal_origin[0])

        # Heuristic base cost: sum graph distances
        # Intuition: need at least these many "up" or "depart" actions to traverse passengers/floors.

        base_cost = floor_distance + passenger_distance

        # Additional costs due to lift and boarding status:

        # Cost if lift not at current_origin passenger (need 1 move to get lift there)
        lift_at_goal = self.goal_origin[1] in lift_positions

        # Cost if boarded passenger not current_origin passenger (need depart to swap)
        boarded_correct = current_origin[1] in boarded_passengers

        move_lift_cost = 0 if current_origin[1] in lift_positions else 1
        depart_cost = 0 if boarded_correct else 1

        # Total estimated cost
        estimated_cost = base_cost + move_lift_cost + depart_cost

        # Ensure heuristic is at least 1 if state not goal, to maintain progress signal
        if estimated_cost == 0:
            estimated_cost = 1

        return estimated_cost
