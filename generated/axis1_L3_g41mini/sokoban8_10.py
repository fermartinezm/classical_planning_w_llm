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
    A domain-dependent heuristic for the secret_domain (miconic) domain.

    Summary
    -------
    The heuristic estimates the number of actions required to achieve the goal where the lift's
    origin passenger must be a specific passenger. The domain models a "lift" system with passengers,
    floors, and a notion of boarding, origins, served relations, and lift positions.

    Assumptions
    -----------
    - The goal is always expressed as an (origin passenger-2 passenger) fact.
    - Achieving the goal essentially means moving the origin relation from its current passenger
      to the target passenger in the goal.
    - Actions `depart` and `up` modify the state by transitioning the lift and passengers through
      the served floors and updating boarded and lift-at facts.
    - The heuristic focuses on the minimal number of "up" and "depart" actions needed to propagate
      the origin from the current passenger holding it to the goal passenger.
    - All passengers and floors, and the served relations between them, form a graph that can be
      traversed to estimate distance.
    - Boarding/unboarding and lift moves cost at least 1 action, so the heuristic counts the minimal
      number edges to traverse along "served" edges to reach the goal passenger.
    - Boarding status impacts possible moves — boarded passengers are carried by the lift.

    Heuristic Initialization
    ------------------------
    Upon initialization, the heuristic extracts and builds:
    - A mapping of served edges: from passenger to neighbors via floors (bidirectional),
      as given by `(served p1 p2 floorX)` facts from static info.
    - The set of floors and passengers are known but the heuristic mainly requires the
      served graph for shortest path computations.
    - The goal passenger of the origin relation.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the goal origin fact: (origin passenger-2 goal_passenger).
    2. From the current state, find which passenger currently holds the origin relation.
       There should be exactly one fact `(origin passenger-2 current_passenger)`.
       If that passenger is already the goal passenger, heuristic = 0.
    3. Build a graph where nodes are passengers, edges exist if `(served passengerX passengerY floorZ)` is true.
       Since the served relation is static and bidirectional (usually mirrored with floor directions),
       build an undirected graph of passengers connected by served edges.
    4. Compute the shortest passenger-path in this graph from the current origin passenger to the goal passenger,
       using BFS for efficiency.
    5. The heuristic value is the length of this path, representing the minimal number of "up" actions needed to move 
       the origin from current passenger to goal passenger, plus an estimate of steps for boarding and departing.
       Each move along served edges corresponds roughly to one up action.
    6. Additionally, if the origin passenger is not currently boarded, a boarding action will be needed first (+1).
    7. Finally, a depart action may be needed when reaching the goal passenger (+1).
    8. Sum these values for the heuristic estimate.
    9. This heuristic is zero if the state is a goal (origin relation on goal passenger).
    10. This heuristic is always finite if the goal is reachable, and is efficient as it precomputes connectivity and uses BFS.

    Note
    ----
    This heuristic ignores the floors and lift-at positions explicitly since those are implicitly accounted through the
    served graph edges between passengers, and actions allowed only when boarding/lift-at conditions hold.
    """

    def __init__(self, task):
        """Initializes the heuristic by processing static served relations and the goal."""
        self.goals = task.goals
        self.static = task.static

        # Extract the goal passenger from the origin goal fact: (origin passenger-2 goal_passenger)
        self.goal_passenger = None
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "origin" and len(parts) == 3:
                # parts = [origin, passenger-2, passenger]
                self.goal_passenger = parts[2]
                break

        # Build served graph: map passenger -> set of neighbor passengers connected by served relations
        self.served_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) == 4:
                p1, p2, floor = parts[1], parts[2], parts[3]
                self.served_graph.setdefault(p1, set()).add(p2)
                self.served_graph.setdefault(p2, set()).add(p1)  # assuming served edges are bidirectional

    def __call__(self, node):
        """
        Estimates the minimal number of required actions to move the origin relation
        from the current passenger to the goal passenger.

        The heuristic counts:
        - The BFS shortest passenger-path length along served edges from current origin passenger to goal.
        - Plus 1 if the origin passenger is not boarded (boarding needed).
        - Plus 1 for the depart action at the goal.
        """
        state = node.state

        # Find current origin fact: (origin passenger-2 current_passenger)
        current_origin_passenger = None
        for fact in state:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                # parts = ["origin", passenger-2, current_passenger]
                current_origin_passenger = parts[2]
                break

        # If no origin fact found or goal passenger is unknown, return a default large cost
        if current_origin_passenger is None or self.goal_passenger is None:
            return 1000  # large cost indicating uncertainty or unsolvable

        # If already at goal, no cost
        if current_origin_passenger == self.goal_passenger:
            return 0

        # BFS over served_graph from current_origin_passenger to goal_passenger
        from collections import deque

        visited = set()
        queue = deque([(current_origin_passenger, 0)])

        shortest_distance = None

        while queue:
            current, dist = queue.popleft()
            if current == self.goal_passenger:
                shortest_distance = dist
                break
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.served_graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

        # If no path found, return a large cost
        if shortest_distance is None:
            return 1000

        # Check if the current origin passenger is boarded
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        boarding_cost = 0 if current_origin_passenger in boarded_passengers else 1

        # Add 1 for the depart action at the goal
        depart_cost = 1

        # Total heuristic value
        return shortest_distance + boarding_cost + depart_cost
