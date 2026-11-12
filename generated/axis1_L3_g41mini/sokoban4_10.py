from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(boarded passenger5)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (miconic) elevator domain.

    Summary
    -------
    This heuristic estimates the number of remaining actions required to transform the 
    current state into a goal state by focusing on passengers' boarding status, origin 
    passengers, lift positions, and served floors. The domain involves passengers moving 
    in a lift through floors in a served graph, with complex requirements for boarding, 
    lifting, and serving passengers.

    Assumptions
    -----------
    - The goal is to reach a specific origin relation `(origin p50 pX)` which uniquely 
      identifies the passenger where the lift must end.
    - Passengers can be boarded or not; boarded passengers must be departed eventually.
    - The "served" predicate forms a connectivity graph which models possible moves between 
      passengers and floors.
    - The lift moves between passengers, carrying exactly one boarded passenger at a time.
    - Static facts about "served" allow building a passenger adjacency graph representing 
      accessible moves for the lift.
    - Multiple lifts can be at different passengers simultaneously; however, the goal only 
      demands a particular origin relation changed.
    - The heuristic does not need to be admissible but should better guide the search than a trivial heuristic.

    Heuristic Initialization
    ------------------------
    - Extract from static facts the "served" relations to build a bi-directional graph that 
      connects passengers and floors, enabling shortest path computations.
    - Extract the goal passenger for origin from the goal facts.
    - Store passengers known from static served facts to avoid costly extraction at runtime.
    
    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all boarded passengers in the current state: these must be departed eventually.
    2. Identify the passenger(s) at which the lift is currently located.
    3. Identify the current origin passenger, and the target origin passenger required by the goal.
    4. Calculate minimal "lift moves" between passengers based on traversing the served graph.
    5. Actions needed:
       - Each boarded passenger must be departed at least once (count boarded passengers).
       - If the lift is not at the passenger who should be the origin passenger in the goal, estimate the minimal moves needed to bring the lift there (shortest path on served graph).
       - If origin is not yet correct, estimate at least one boarding/depart/reassign action to reach the origin goal state.
       - Consider that passengers move through a chain of served floors, so moving the lift involves traversing served relations.
    6. The heuristic value is the sum of:
       - Number of boarded passengers (each requires at least 1 action: depart).
       - Minimal number of transitions to get lift(s) to the goal origin passenger.
       - Additional minimal steps to adjust the origin predicate if needed.
    7. When the state is a goal state, heuristic returns 0.

    This heuristic balances accuracy and efficiency by:
    - Using shortest paths precomputed over the static served graph.
    - Counting actions related to boarded passengers explicitly.
    - Encoding minimum necessary lift movements and reassignments.
    """

    def __init__(self, task):
        """Initialize with served adjacency graph extracted from static facts and goal origin passenger."""
        self.goals = task.goals
        self.static = task.static

        # Extract "served" relations from static facts to build adjacency graph for shortest path search.
        # served facts are of the form: (served pX pY floorZ) indicating connectivity.
        self.adjacency = {}  # key: passenger or floor, value: set of connected passengers/floors

        # Build adjacency from served facts as undirected edges between two nodes (passengers/floors).
        # served facts: (served ?v3 - passenger ?v4 - passenger ?v5 - floor)
        for fact in self.static:
            if match(fact, "served", "*", "*", "*"):
                parts = get_parts(fact)
                _, p1, p2, floor = parts
                # Connect p1 <-> p2
                self.adjacency.setdefault(p1, set()).add(p2)
                self.adjacency.setdefault(p2, set()).add(p1)
                # Also consider the floor as connected to both passengers, to allow lift traversals via floors
                self.adjacency.setdefault(p1, set()).add(floor)
                self.adjacency.setdefault(p2, set()).add(floor)
                self.adjacency.setdefault(floor, set()).add(p1)
                self.adjacency.setdefault(floor, set()).add(p2)

        # Identify the passenger that must be the origin passenger at the goal.
        # Goal is like: (origin passenger50 passenger11)
        # We extract the second argument of the origin predicate in the goal.
        self.goal_origin = None
        for goal in self.goals:
            if match(goal, "origin", "*", "*"):
                parts = get_parts(goal)
                # parts = ['origin', originPassenger, passenger_lift_at]
                # We want the "passenger_lift_at" as goal for lift's position
                # In fact, the goal is (origin ?passenger-2 ?passenger)
                # Origin predicate is (origin passenger50 passenger11) means lift is at passenger11
                self.goal_origin = parts[2]  # The passenger where the lift should be at goal
                break

    def shortest_path_moves(self, start, goal):
        """
        Compute shortest path distance length between start and goal nodes in adjacency graph.
        Each edge traversal counts as 1 move.

        Returns a non-negative integer or a large number if no path exists.
        """
        if start == goal:
            return 0
        if start not in self.adjacency or goal not in self.adjacency:
            return 1000  # large cost when no connectivity

        from collections import deque

        visited = set()
        queue = deque([(start, 0)])
        visited.add(start)

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 1000  # large cost if no path found

    def __call__(self, node):
        """Compute heuristic estimate of actions needed to reach goal from current state."""

        state = node.state

        # If goal reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract boarded passengers in current state
        boarded_passengers = {
            get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")
        }

        # Count boarded passengers (each needs at least one depart action)
        num_boarded = len(boarded_passengers)

        # Extract lift-at passengers (positions of the lift in current state)
        lifts_positions = {
            get_parts(fact)[1] for fact in state if match(fact, "lift-at", "*")
        }

        # Extract current origin relation (should be unique)
        current_origin_passenger = None
        for fact in state:
            if match(fact, "origin", "*", "*"):
                # origin (passenger-2 passenger)
                parts = get_parts(fact)
                current_origin_passenger = parts[2]
                break

        # Compute minimal moves needed for at least one lift's position to reach goal_origin
        # If multiple lifts present, take minimal distance among them
        if lifts_positions:
            moves_to_goal_origin = min(
                self.shortest_path_moves(pos, self.goal_origin) for pos in lifts_positions
            )
        else:
            # No lift positions known, pessimistically large moves
            moves_to_goal_origin = 1000

        # If current origin passenger is already goal origin, zero cost adjusting origin, else 1 for adjusting departure/origin
        origin_adj_cost = 0 if current_origin_passenger == self.goal_origin else 1

        # Final heuristic estimate:
        # - Depart actions needed for boarded passengers
        # - Moves for lift to get to goal origin (to satisfy goal origin predicate)
        # - Origin adjustment cost if needed (boarding/depart/reassign steps)
        heuristic_value = num_boarded + moves_to_goal_origin + origin_adj_cost

        # In difficult or dead-end states, value won't be infinite and will guide search reasonably towards:
        # - unboard all boarded passengers
        # - move lift to correct passenger floor
        # - achieve correct origin predicate
        return heuristic_value
