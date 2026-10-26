from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the required moves for the lift to reach each passenger's origin and destination.

    # Assumptions:
    - The lift can move up or down between floors.
    - Each passenger must be boarded and then served at their destination.
    - The heuristic assumes the optimal path for the lift to minimize moves.

    # Heuristic Initialization
    - Extracts static facts about floor hierarchies to determine distances between floors.
    - Maps each floor to its above floors for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current state of passengers and the lift.
    2. For each passenger not served:
       a. If not boarded, calculate moves to reach their origin.
       b. Calculate moves from origin to destination.
       c. Add boarding and serving actions.
    3. Sum all required actions to estimate the total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic with static floor information."""
        self.goals = task.goals
        static_facts = task.static

        # Build floor hierarchy from static facts
        self.floor_above = {}
        self.floor_below = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f1, f2 = get_parts(fact)[1], get_parts(fact)[2]
                if f1 not in self.floor_above:
                    self.floor_above[f1] = []
                self.floor_above[f1].append(f2)
                if f2 not in self.floor_below:
                    self.floor_below[f2] = []
                self.floor_below[f2].append(f1)

    def __call__(self, node):
        """Compute the estimated number of actions to serve all passengers."""
        state = node.state
        current_lift_floor = None
        passengers = {}

        # Extract current state information
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
            elif match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1], get_parts(fact)[2]
                passengers[p] = {'origin': f, 'destin': None, 'boarded': False, 'served': False}
            elif match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1], get_parts(fact)[2]
                passengers[p]['destin'] = f
            elif match(fact, "boarded", "*"):
                p = get_parts(fact)[1]
                passengers[p]['boarded'] = True
            elif match(fact, "served", "*"):
                p = get_parts(fact)[1]
                passengers[p]['served'] = True

        total_actions = 0

        # For each passenger, calculate required actions
        for p, data in passengers.items():
            if data['served']:
                continue

            # Determine if boarded
            if not data['boarded']:
                # Find path from current lift floor to origin
                origin = data['origin']
                path = self.find_path(current_lift_floor, origin)
                if not path:
                    return float('inf')  # No path, state unsolvable
                total_actions += len(path)  # Moves to reach origin
                total_actions += 1  # Board action

            # Calculate moves from origin to destination
            destin = data['destin']
            path = self.find_path(data['origin'], destin)
            if not path:
                return float('inf')
            total_actions += len(path)  # Moves to reach destination
            total_actions += 1  # Serve action

        return total_actions

    def find_path(self, start, end):
        """
        Find the shortest path from start to end floor using the floor hierarchy.
        Returns the list of floors in the path or empty list if no path.
        """
        visited = set()
        queue = [start]
        while queue:
            current = queue.pop(0)
            if current == end:
                return []
            if current in visited:
                continue
            visited.add(current)
            if current in self.floor_above:
                for floor in self.floor_above[current]:
                    if floor not in visited:
                        new_path = [current] + self.find_path(floor, end)
                        if new_path:
                            return new_path
            if current in self.floor_below:
                for floor in self.floor_below[current]:
                    if floor not in visited:
                        new_path = [current] + self.find_path(floor, end)
                        if new_path:
                            return new_path
        return []

    @staticmethod
    def get_parts(fact):
        """Extract components of a PDDL fact by removing parentheses and splitting."""
        return fact[1:-1].split()

    @staticmethod
    def match(fact, *args):
        """
        Check if a PDDL fact matches a given pattern.
        Returns True if the fact matches the pattern, else False.
        """
        parts = miconicHeuristic.get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))
