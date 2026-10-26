# Helper function to extract parts from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to extract the numerical part from a floor name
import re
def get_floor_number(floor_name):
    """Extract the number from a floor name like 'f1', 'f10'."""
    match = re.match(r'f(\d+)', floor_name)
    if match:
        return int(match.group(1))
    # Return a large value for floor names that don't match the pattern,
    # ensuring they are sorted last. This handles unexpected formats gracefully.
    return float('inf')

# Import the base Heuristic class
from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions (board, depart) and
    the total floor distance the lift needs to travel to serve all unserved
    passengers, assuming each passenger is served on an independent trip.

    # Assumptions
    - Floors are named 'f1', 'f2', ..., 'fN' and are ordered numerically.
    - The 'above' static facts define the floor hierarchy consistently with
      this naming convention.
    - The lift has infinite capacity (this assumption is implicit in summing
      individual passenger costs).
    - Passengers are served individually for heuristic calculation purposes,
      ignoring potential optimizations from serving multiple passengers in one trip.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from static facts.
    - Maps floor names to numerical indices based on the 'above' static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift by finding the fact `(lift-at ?f)` in the state.
    2. Get the numerical index for the current lift floor using the pre-calculated floor mapping.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each passenger whose destination is known (which should be all passengers defined in the problem).
    5. For the current passenger:
       a. Check if the passenger is already 'served' by looking for the fact `(served ?p)` in the state. If the passenger is served, add 0 to the total cost for this passenger and proceed to the next passenger.
       b. If the passenger is not served, check if the passenger is 'boarded' by looking for the fact `(boarded ?p)` in the state.
          - If the passenger is 'boarded':
            - Add 1 to the cost for this passenger (representing the necessary 'depart' action).
            - Get the numerical index for the passenger's destination floor.
            - Add the absolute difference between the current lift floor index and the passenger's destination floor index to the cost (this estimates the lift movement needed to reach the destination).
       c. If the passenger is not served and not 'boarded' (they must be waiting at their 'origin' floor):
          - Find the passenger's origin floor by looking for the fact `(origin ?p ?f)` in the current state.
          - Add 2 to the cost for this passenger (representing the necessary 'board' and 'depart' actions).
          - Get the numerical index for the passenger's origin floor and destination floor.
          - Add the absolute difference between the current lift floor index and the passenger's origin floor index to the cost (estimates movement to origin).
          - Add the absolute difference between the passenger's origin floor index and the passenger's destination floor index to the cost (estimates movement from origin to destination).
    6. The total heuristic value is the sum of costs calculated for all unserved passengers.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and
        mapping floor names to indices.
        """
        # The set of facts that must hold in goal states. In Miconic, this is (served ?p) for all passengers.
        self.goals = task.goals
        # Static facts are facts that do not change during planning.
        static_facts = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor

        # Map floor names to numerical indices based on 'above' facts
        floor_names = set()
        # Collect all floor names mentioned in 'above' facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                # (above floor_higher floor_lower)
                floor_names.add(parts[1]) # floor_higher
                floor_names.add(parts[2]) # floor_lower

        # Sort floor names based on their numerical part (e.g., f1, f2, ..., f10)
        # This assumes floor names follow the pattern 'f' followed by a number.
        sorted_floor_names = sorted(list(floor_names), key=get_floor_number)

        # Create floor name to index mapping (1-based index)
        # The lowest floor (e.g., f1) gets index 1, the next gets 2, and so on.
        self.floor_to_index = {name: i + 1 for i, name in enumerate(sorted_floor_names)}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # Current world state, represented as a frozenset of fact strings.

        total_cost = 0 # Initialize the total heuristic cost.

        # Find the current floor of the lift
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        # If the lift location is not found, the state is likely invalid or
        # represents an unsolvable situation from this point.
        if current_lift_floor is None:
             # In a well-formed problem, lift-at should always be true for exactly one floor.
             # Returning infinity signals this state is likely problematic or far from goal.
             return float('inf')

        current_lift_floor_index = self.floor_to_index[current_lift_floor]

        # Iterate through all passengers defined in the problem (those with a destination)
        for passenger, destin_floor in self.destinations.items():
            # Check if the passenger is already served. Served passengers don't contribute to the heuristic cost.
            if f'(served {passenger})' in state:
                continue

            # Get the index of the passenger's destination floor
            destin_floor_index = self.floor_to_index[destin_floor]

            # Check if the passenger is currently boarded in the lift
            if f'(boarded {passenger})' in state:
                # This passenger is boarded and not yet served.
                # They need to be transported to their destination and depart.
                cost_for_passenger = 1 # Cost for the 'depart' action.
                # Add the estimated movement cost for the lift to reach the destination floor
                cost_for_passenger += abs(current_lift_floor_index - destin_floor_index)
                total_cost += cost_for_passenger
            else:
                # This passenger is not served and not boarded. They must be waiting at their origin floor.
                # Find the passenger's origin floor in the current state.
                origin_floor = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'origin' and parts[1] == passenger:
                        origin_floor = parts[2]
                        break

                # If the passenger is unserved, unboarded, and not at an origin,
                # the state is likely invalid.
                if origin_floor is None:
                    # In a well-formed problem, an unserved, unboarded passenger
                    # must have an 'origin' fact.
                    return float('inf') # Signal problematic state

                origin_floor_index = self.floor_to_index[origin_floor]

                # This passenger needs to be boarded at their origin and then depart at their destination.
                cost_for_passenger = 2 # Cost for 'board' + 'depart' actions.
                # Add the estimated movement cost for the lift to reach the origin floor
                cost_for_passenger += abs(current_lift_floor_index - origin_floor_index)
                # Add the estimated movement cost for the lift to travel from the origin floor to the destination floor
                cost_for_passenger += abs(origin_floor_index - destin_floor_index)
                total_cost += cost_for_passenger

        return total_cost
