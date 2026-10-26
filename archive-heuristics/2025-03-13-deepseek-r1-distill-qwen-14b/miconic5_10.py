from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by:
    1. Boarding each passenger at their origin floor.
    2. Moving the lift to the destination floor.
    3. Departing to serve the passenger.

    # Assumptions:
    - Each passenger requires exactly 3 actions: board, move, depart.
    - The lift starts at a specific floor and must move to each passenger's origin floor if not already there.
    - Movement between floors is based on the hierarchy defined in static facts.

    # Heuristic Initialization
    - Extracts static facts to build a floor hierarchy.
    - Maps each floor to its parent floors for efficient distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current state of the lift and passengers.
    2. For each passenger:
       a. If already served, skip.
       b. If boarded but not served, calculate the moves needed to reach their destination.
       c. If not boarded, calculate the moves needed to reach their origin floor, then to destination, and finally to serve them.
    3. Sum the required actions for all unserved passengers.
    4. Adjust the total based on the lift's current position to minimize unnecessary movements.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Build floor hierarchy from static facts
        self.floor_above = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                floor1, floor2 = get_parts(fact)[1], get_parts(fact)[2]
                if floor1 not in self.floor_above:
                    self.floor_above[floor1] = []
                self.floor_above[floor1].append(floor2)

        # Precompute parent floors for each floor
        self.parent_floor = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                floor1, floor2 = get_parts(fact)[1], get_parts(fact)[2]
                self.parent_floor[floor2] = floor1

    def __call__(self, node):
        """Estimate the minimum number of actions to serve all passengers."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Extract current lift position
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        # Extract information about passengers
        passengers = {}
        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1], get_parts(fact)[2]
                passengers[p] = {'origin': f}
            if match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1], get_parts(fact)[2]
                passengers[p]['destin'] = f
            if match(fact, "boarded", "*"):
                p = get_parts(fact)[1]
                passengers[p]['boarded'] = True
            if match(fact, "served", "*"):
                p = get_parts(fact)[1]
                passengers[p]['served'] = True

        total_actions = 0

        for p, data in passengers.items():
            if 'served' not in data or not data['served']:
                if 'destin' not in data:
                    continue  # No destination specified, skip
                if 'origin' not in data:
                    continue  # No origin specified, skip

                # Passenger needs to be served
                origin = data['origin']
                destin = data['destin']
                boarded = data.get('boarded', False)
                served = data.get('served', False)

                if not served:
                    if not boarded:
                        # Need to board the passenger
                        total_actions += 1

                        # Calculate moves to reach origin floor
                        current_floor = lift_floor
                        target_floor = origin
                        if current_floor != target_floor:
                            # Move down or up as needed
                            total_actions += self.get_floor_distance(current_floor, target_floor)

                    # Move to destination floor
                    current_floor = origin if boarded else lift_floor
                    target_floor = destin
                    if current_floor != target_floor:
                        total_actions += self.get_floor_distance(current_floor, target_floor)

                    # Depart to serve
                    total_actions += 1

        return total_actions

    def get_floor_distance(self, from_floor, to_floor):
        """
        Calculate the number of moves required to go from one floor to another.
        """
        distance = 0
        current = to_floor
        while current != from_floor:
            if current in self.parent_floor:
                current = self.parent_floor[current]
                distance += 1
            else:
                # If no path exists (shouldn't happen in valid instances)
                return float('inf')
        return distance

    def __eq__(self, other):
        return isinstance(other, MiconicHeuristic)

    def __hash__(self):
        return hash(MiconicHeuristic)
