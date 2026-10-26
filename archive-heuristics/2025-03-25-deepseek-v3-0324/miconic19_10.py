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
    1. The current position of the elevator
    2. The passengers that still need to be boarded
    3. The passengers that are boarded but not yet served
    4. The floor relationships (above) for movement costs

    # Assumptions:
    - The elevator can only move between floors connected by 'above' relations
    - Each passenger must be boarded from their origin floor before being served at their destination
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract destination floors for each passenger from static facts
    - Build a graph of floor connections from 'above' relations
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger not yet served:
        a) If not boarded:
            - Add cost to move elevator to passenger's origin floor
            - Add boarding action cost
            - Add cost to move from origin to destination floor
        b) If already boarded:
            - Add cost to move elevator to passenger's destination floor
        c) Add departing action cost
    2. Optimize by considering passengers that can be served along the way
    3. Use floor distances (minimum steps between floors) for movement cost estimation
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract destination floors for each passenger
        self.destinations = {}
        # Extract origin floors (may change as passengers are boarded)
        self.origins = {}
        # Build floor connectivity graph
        self.above_graph = {}

        for fact in self.static:
            parts = get_parts(fact)
            if match(fact, "destin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor
            elif match(fact, "above", "*", "*"):
                upper, lower = parts[1], parts[2]
                if upper not in self.above_graph:
                    self.above_graph[upper] = []
                self.above_graph[upper].append(lower)

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        total_cost = 0

        # Get current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        if not current_floor:
            return float('inf')  # Invalid state

        # Track passengers that still need service
        unserved_passengers = []
        boarded_passengers = set()
        remaining_origins = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "served", "*"):
                continue  # Already served
            elif match(fact, "origin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                remaining_origins[passenger] = floor
                unserved_passengers.append(passenger)
            elif match(fact, "boarded", "*"):
                passenger = parts[1]
                boarded_passengers.add(passenger)
                unserved_passengers.append(passenger)

        if not unserved_passengers:
            return 0  # Goal state

        # For each unserved passenger, calculate their individual cost
        for passenger in unserved_passengers:
            if passenger in boarded_passengers:
                # Passenger is boarded - need to go to destination
                dest = self.destinations[passenger]
                # Add movement cost from current position to destination
                total_cost += self._floor_distance(current_floor, dest)
                # Add depart action
                total_cost += 1
                # Update current floor to destination after movement
                current_floor = dest
            else:
                # Passenger needs to be boarded first
                origin = remaining_origins[passenger]
                dest = self.destinations[passenger]
                # Move to origin floor
                total_cost += self._floor_distance(current_floor, origin)
                # Board action
                total_cost += 1
                # Move to destination floor
                total_cost += self._floor_distance(origin, dest)
                # Depart action
                total_cost += 1
                # Update current floor to destination after movement
                current_floor = dest

        return total_cost

    def _floor_distance(self, floor1, floor2):
        """
        Estimate the minimum number of up/down actions needed to move between floors.
        Uses BFS on the 'above' graph to find shortest path.
        """
        if floor1 == floor2:
            return 0

        # The 'above' relations form a DAG where we can move both up and down
        # We'll implement a simple BFS to find the shortest path
        visited = set()
        queue = [(floor1, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == floor2:
                return dist
            
            if current in visited:
                continue
            visited.add(current)
            
            # Check floors above current
            if current in self.above_graph:
                for upper in self.above_graph[current]:
                    queue.append((upper, dist + 1))
            
            # Check floors below current (reverse lookup)
            for upper, lowers in self.above_graph.items():
                if current in lowers:
                    queue.append((upper, dist + 1))
        
        return float('inf')  # No path found (shouldn't happen in valid problems)
