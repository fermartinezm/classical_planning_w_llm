from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(origin p1 f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    by considering:
    - The current position of the elevator
    - Which passengers still need to be boarded
    - Which boarded passengers still need to be served
    - The floor relationships (above) for movement costs

    # Assumptions:
    - The elevator can only move between floors connected by 'above' relations
    - Each passenger must be boarded from their origin floor before being served at their destination
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract destination floors for each passenger from static facts
    - Build a graph representation of floor connectivity using 'above' relations
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
        a. If not boarded:
            - Add cost to move elevator from current position to passenger's origin floor
            - Add 1 for boarding action
            - Add cost to move from origin to destination floor
        b. If already boarded:
            - Add cost to move elevator from current position to passenger's destination floor
        c. Add 1 for depart action
    2. For efficiency, we:
        - Process boarded passengers first (since they're already in the elevator)
        - Then process unboarded passengers in an order that minimizes elevator movement
    3. The movement cost between floors is estimated as the difference in their levels
       (since floors are linearly ordered by 'above' relations)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations
        self.destinations = {}
        # Extract floor hierarchy
        self.above_relations = set()
        self.all_floors = set()

        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor
            elif match(fact, "above", "*", "*"):
                _, floor1, floor2 = get_parts(fact)
                self.above_relations.add((floor1, floor2))
                self.all_floors.update([floor1, floor2])

        # Build floor ordering for distance calculation
        self.floor_order = self._compute_floor_order()

    def _compute_floor_order(self):
        """Compute a linear ordering of floors based on 'above' relations."""
        # The 'above' relations form a total order, so we can sort floors
        from collections import defaultdict

        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        for upper, lower in self.above_relations:
            graph[upper].add(lower)
            in_degree[lower] += 1
        
        # Find the top floor (with no incoming edges)
        top_floors = [f for f in self.all_floors if in_degree[f] == 0]
        assert len(top_floors) == 1, "Floor hierarchy should have a single top floor"
        
        # Perform topological sort
        order = []
        queue = [top_floors[0]]
        
        while queue:
            current = queue.pop()
            order.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return order

    def _floor_distance(self, floor1, floor2):
        """Estimate distance between two floors based on their positions in the order."""
        try:
            idx1 = self.floor_order.index(floor1)
            idx2 = self.floor_order.index(floor2)
            return abs(idx1 - idx2)
        except ValueError:
            # Fallback if floors aren't in order (shouldn't happen in valid problems)
            return 1

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        current_floor = None
        boarded_passengers = set()
        unserved_passengers = set()
        served_passengers = set()

        # Extract current state information
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, floor = get_parts(fact)
                current_floor = floor
            elif match(fact, "boarded", "*"):
                _, passenger = get_parts(fact)
                boarded_passengers.add(passenger)
            elif match(fact, "served", "*"):
                _, passenger = get_parts(fact)
                served_passengers.add(passenger)

        # Find passengers that still need to be served
        remaining_passengers = set(self.destinations.keys()) - served_passengers
        if not remaining_passengers:
            return 0  # Goal reached

        # Separate boarded and unboarded passengers
        boarded_remaining = remaining_passengers & boarded_passengers
        unboarded_remaining = remaining_passengers - boarded_passengers

        # Initialize heuristic cost
        total_cost = 0
        current_pos = current_floor

        # First process already boarded passengers (they're in the elevator)
        for passenger in boarded_remaining:
            dest_floor = self.destinations[passenger]
            # Move to destination floor
            total_cost += self._floor_distance(current_pos, dest_floor)
            current_pos = dest_floor
            # Depart action
            total_cost += 1

        # Then process unboarded passengers
        for passenger in unboarded_remaining:
            # Find origin floor from state (since origin is deleted when boarded)
            origin_floor = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    _, _, floor = get_parts(fact)
                    origin_floor = floor
                    break
            
            if origin_floor is None:
                continue  # Shouldn't happen for valid states

            # Move to origin floor
            total_cost += self._floor_distance(current_pos, origin_floor)
            current_pos = origin_floor
            # Board action
            total_cost += 1
            # Move to destination floor
            dest_floor = self.destinations[passenger]
            total_cost += self._floor_distance(current_pos, dest_floor)
            current_pos = dest_floor
            # Depart action
            total_cost += 1

        return total_cost
