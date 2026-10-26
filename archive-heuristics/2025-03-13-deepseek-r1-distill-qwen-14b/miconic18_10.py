from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the required movements of the lift and the number of depart actions.

    # Assumptions:
    - The lift can move between floors in either direction.
    - Each passenger requires exactly one depart action after being boarded.
    - The goal is to serve all passengers, meaning they must be transported to their destination floors.

    # Heuristic Initialization
    - Extract the 'above' relationships from static facts to determine floor hierarchy.
    - Store the floor above each floor for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current position of the lift.
    2. For each passenger, determine if they have been served. If not, note their destination floor.
    3. For each unserved passenger, calculate the minimal distance from the lift's current position to their destination.
    4. Track the maximum distance required in both upward and downward directions.
    5. The movement cost is the sum of these maximum distances.
    6. The number of depart actions is equal to the number of unserved passengers.
    7. The total heuristic value is the sum of movement cost and depart actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor hierarchy from static facts.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Build floor hierarchy: floor_above[floor] = the floor directly above it
        self.floor_above = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                below, above = get_parts(fact)[1], get_parts(fact)[2]
                self.floor_above[below] = above

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        current_lift = None
        served_passengers = set()
        passenger_destinations = {}

        # Extract current lift position
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift = get_parts(fact)[1]
                break

        # Identify served passengers
        for fact in state:
            if match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])

        # Identify destinations of unserved passengers
        for fact in state:
            if match(fact, "destin", "*", "*"):
                passenger = get_parts(fact)[1]
                if passenger not in served_passengers:
                    destination = get_parts(fact)[2]
                    passenger_destinations[passenger] = destination

        # If no unserved passengers, return 0
        if not passenger_destinations:
            return 0

        max_up_distance = 0
        max_down_distance = 0

        # Helper function to calculate distance between two floors
        def calculate_distance(from_floor, to_floor):
            distance = 0
            # Move up
            if from_floor < to_floor:
                current = from_floor
                while current != to_floor:
                    current = self.floor_above[current]
                    distance += 1
            # Move down
            elif from_floor > to_floor:
                current = from_floor
                while current != to_floor:
                    # Find the floor below current
                    for floor, above_floor in self.floor_above.items():
                        if above_floor == current:
                            current = floor
                            break
                    else:
                        current = None  # No floor below, can't move further
                        break
                    distance += 1
            return distance

        # Calculate max up and down distances
        for passenger, destination in passenger_destinations.items():
            distance = calculate_distance(current_lift, destination)
            if distance == 0:
                continue  # Already at destination
            if destination > current_lift:
                if distance > max_up_distance:
                    max_up_distance = distance
            else:
                if distance > max_down_distance:
                    max_down_distance = distance

        movement_cost = max_up_distance + max_down_distance
        depart_actions = len(passenger_destinations)
        return movement_cost + depart_actions
