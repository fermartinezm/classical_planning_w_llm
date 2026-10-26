from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(lift-at f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    by considering the necessary movements of the lift for each unserved passenger.
    It sums up the estimated cost for each passenger independently, which might overestimate
    the actual cost but provides a computationally efficient heuristic.

    # Assumptions:
    - The floor names are ordered in a way that allows distance calculation by comparing floor names.
    - Serving each passenger requires at least moving the lift to their origin floor, boarding them,
      moving to their destination floor, and departing them.
    - The heuristic does not consider optimizing lift movements for multiple passengers simultaneously.

    # Heuristic Initialization
    - Extracts static information about passenger origins and destinations.
    - Prepares data structures to quickly access origin and destination floors for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger that is not yet served:
    1. Check if the passenger is already boarded.
    2. If not boarded:
       - Determine the passenger's origin floor.
       - Determine the current floor of the lift.
       - Estimate the number of 'up' or 'down' actions needed to move the lift from its current floor to the origin floor.
       - Add 1 action for the 'board' action.
    3. If boarded:
       - Determine the passenger's destination floor.
       - Determine the current floor of the lift.
       - Estimate the number of 'up' or 'down' actions needed to move the lift from its current floor to the destination floor.
       - Add 1 action for the 'depart' action.
    4. Sum up the estimated actions for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic.
        Extracts passenger origins and destinations from the static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above_relations = set()

        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_origins[passenger] = floor
            elif match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_destinations[passenger] = floor
            elif match(fact, "above", "*", "*"):
                self.above_relations.add(fact)

        self.floor_order = self._determine_floor_order(static_facts)

    def _determine_floor_order(self, static_facts):
        """
        Infers the order of floors based on 'above' predicates.
        If floor order cannot be fully determined, it returns a list of unique floors in arbitrary order.
        """
        floors = set()
        above_map = {}
        below_map = {}

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f1, f2 = get_parts(fact)[1], get_parts(fact)[2]
                floors.add(f1)
                floors.add(f2)
                above_map.setdefault(f1, []).append(f2)
                below_map.setdefault(f2, []).append(f1)

        sorted_floors = []
        unprocessed_floors = list(floors)

        while unprocessed_floors:
            current_floor = unprocessed_floors.pop(0)
            if current_floor not in below_map: # Start from the lowest floor
                sorted_floors.append(current_floor)
                processed = {current_floor}
                queue = [current_floor]
                while queue:
                    floor = queue.pop(0)
                    if floor in above_map:
                        for next_floor in above_map[floor]:
                            if next_floor not in processed:
                                sorted_floors.append(next_floor)
                                processed.add(next_floor)
                                queue.append(next_floor)
                break # Assume there is a single chain of floors. For more complex cases, this needs refinement.
        if not sorted_floors: # Fallback if no order can be inferred, use arbitrary order.
            return list(floors)
        return sorted_floors


    def _get_floor_distance(self, floor1, floor2):
        """Calculates the distance between two floors based on their order."""
        try:
            index1 = self.floor_order.index(floor1)
            index2 = self.floor_order.index(floor2)
            return abs(index1 - index2)
        except ValueError: # Handle cases where floor order is not fully defined. Fallback to distance 1 if floors are different.
            return 1 if floor1 != floor2 else 0


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        Estimates the number of actions needed to reach the goal state from the current state.
        """
        state = node.state
        heuristic_value = 0

        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            return float('inf') # Should not happen in valid miconic problems

        unserved_passengers = []
        for passenger in self.passenger_origins: # Iterate through all passengers defined in the problem
            served_predicate = f'(served {passenger})'
            if served_predicate not in state:
                unserved_passengers.append(passenger)

        for passenger in unserved_passengers:
            boarded_predicate = f'(boarded {passenger})'
            if boarded_predicate not in state:
                origin_floor = self.passenger_origins[passenger]
                distance_to_origin = self._get_floor_distance(lift_floor, origin_floor)
                heuristic_value += distance_to_origin + 1 # Move to origin + board
            else:
                destination_floor = self.passenger_destinations[passenger]
                distance_to_destination = self._get_floor_distance(lift_floor, destination_floor)
                heuristic_value += distance_to_destination + 1 # Move to destination + depart

        return heuristic_value
