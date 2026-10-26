# Need to import the base class Heuristic
# from heuristics.heuristic_base import Heuristic
# Assuming Heuristic base class is available in the environment

from fnmatch import fnmatch
import math # Import math for float('inf')

# Utility functions from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class inheriting from Heuristic
class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve
    all passengers. It calculates the cost for each unserved passenger
    independently and sums these costs.

    # Assumptions
    - The floor structure is a linear sequence defined by `(above f1 f2)` facts.
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic assumes the lift can serve each passenger optimally as if it
      were the only passenger, ignoring potential synergies (picking up/dropping
      off multiple passengers at the same floor).

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - Parses the `(above f1 f2)` static facts to determine the linear order
      of floors and creates a mapping from floor name to a numerical floor level.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the current state is a goal state (all passengers served). If yes, the heuristic is 0.
    2.  Identify the current floor of the lift.
    3.  For each passenger defined in the problem (those with a destination):
        a.  Check if the passenger is already served. If yes, the cost for this passenger is 0.
        b.  If the passenger is not served, determine their current status:
            i.  If the passenger is at their origin floor (fact `(origin p f_origin)` is true):
                - Calculate the number of moves needed for the lift to go from its current floor to the passenger's origin floor: `abs(current_lift_floor_number - origin_floor_number)`.
                - Add 1 for the `board` action.
                - Calculate the number of moves needed for the lift to go from the origin floor to the passenger's destination floor: `abs(origin_floor_number - destination_floor_number)`.
                - Add 1 for the `depart` action.
                - The total cost for this passenger is the sum of these moves and actions.
            ii. If the passenger is boarded in the lift (fact `(boarded p)` is true):
                - Calculate the number of moves needed for the lift to go from its current floor to the passenger's destination floor: `abs(current_lift_floor_floor_number - destination_floor_number)`.
                - Add 1 for the `depart` action.
                - The total cost for this passenger is the sum of these moves and actions.
            iii. (Implicit case: Passenger is served - handled in 3a).
    4.  The total heuristic value is the sum of the costs calculated for each unserved passenger.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions.
        - Static facts (`destin` relationships and `above` relationships).
        - Floor numbering based on `above` facts.
        """
        # Call the base class constructor to store goals and static facts
        super().__init__(task)

        # Store goal conditions for easy checking
        self.goals = task.goals

        # Extract passenger destinations from static facts
        self.passenger_destinations = {}
        for fact in self.static:
            predicate, *args = get_parts(fact)
            if predicate == "destin" and len(args) == 2:
                passenger, destination_floor = args
                self.passenger_destinations[passenger] = destination_floor

        # Determine floor order and create floor_to_number mapping
        self.floor_to_number = self._build_floor_numbering(self.static)

    def _build_floor_numbering(self, static_facts):
        """
        Parses (above f_above f_below) facts to determine floor order
        and assigns a numerical level to each floor.
        Assumes floors form a single linear sequence.
        """
        immediately_above = {}
        all_floors = set()

        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "above" and len(args) == 2:
                f_above, f_below = args
                immediately_above[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)

        if not all_floors:
             # Handle case with no floors or no above facts
             return {}

        # Find the lowest floor (a floor that is never the value in immediately_above)
        lowest_floor = None
        above_values = set(immediately_above.values())
        for floor in all_floors:
            if floor not in above_values:
                lowest_floor = floor
                break

        # Build the ordered list of floors from lowest to highest
        ordered_floors = []
        current_floor = lowest_floor
        # Check if lowest_floor was found before starting the loop
        if lowest_floor is not None:
            while current_floor is not None:
                ordered_floors.append(current_floor)
                current_floor = immediately_above.get(current_floor) # Get floor immediately above

        # Create the floor name to number mapping
        floor_to_number = {floor: i + 1 for i, floor in enumerate(ordered_floors)}

        return floor_to_number

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all passengers.
        """
        state = node.state  # Current world state (frozenset of facts)

        # If the goal is reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        total_cost = 0  # Initialize action cost counter.

        # Find the current location of the lift
        lift_current_floor = None
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "lift-at" and len(args) == 1:
                lift_current_floor = args[0]
                break

        # If lift location is not found or floor numbering failed, return a large value
        if lift_current_floor is None or lift_current_floor not in self.floor_to_number:
             # Cannot compute heuristic without lift location or valid floor numbering
             # This indicates an invalid state or problem definition.
             # Return a large value to discourage this state.
             return math.inf

        lift_current_floor_num = self.floor_to_number[lift_current_floor]

        # Iterate through all passengers we know the destination for
        for passenger, destin_floor in self.passenger_destinations.items():
            # Check if the passenger is already served
            if f"(served {passenger})" in state:
                continue # Passenger is served, no cost for this one

            # Ensure destination floor is in our numbering (should be if problem is valid)
            if destin_floor not in self.floor_to_number:
                 # This indicates an invalid problem definition.
                 # Cannot compute cost for this passenger.
                 return math.inf # Return large value for malformed problem

            destin_floor_num = self.floor_to_number[destin_floor]

            # Check the status of the unserved passenger
            origin_floor = None
            is_boarded = False
            found_status = False # Track if we found origin or boarded status

            # Iterate through state facts to find passenger status
            for fact in state:
                 predicate, *args = get_parts(fact)
                 if predicate == "origin" and len(args) == 2 and args[0] == passenger:
                      origin_floor = args[1]
                      found_status = True
                      break # Found origin status
                 if predicate == "boarded" and len(args) == 1 and args[0] == passenger:
                      is_boarded = True
                      found_status = True
                      break # Found boarded status

            # A valid state for an unserved passenger must have either origin or boarded status
            if not found_status:
                 # This indicates an invalid state (passenger exists but is nowhere)
                 return math.inf # Return large value for invalid state

            if origin_floor:
                # Passenger is waiting at origin
                # Ensure origin floor is in our numbering
                if origin_floor not in self.floor_to_number:
                     # This indicates an invalid problem definition.
                     return math.inf # Return large value for malformed problem

                origin_floor_num = self.floor_to_number[origin_floor]

                # Cost: Move lift to origin + Board + Move lift to destination + Depart
                # Moves: abs(current -> origin) + abs(origin -> destin)
                # Actions: Board + Depart = 1 + 1 = 2
                move_cost = abs(lift_current_floor_num - origin_floor_num) + abs(origin_floor_num - destin_floor_num)
                action_cost = 2 # 1 for board, 1 for depart
                total_cost += move_cost + action_cost

            elif is_boarded:
                # Passenger is boarded
                # Cost: Move lift to destination + Depart
                # Moves: abs(current -> destin)
                # Actions: Depart = 1
                move_cost = abs(lift_current_floor_num - destin_floor_num)
                action_cost = 1 # 1 for depart
                total_cost += move_cost + action_cost

            # If neither origin nor boarded was found, it's an invalid state,
            # handled by the `if not found_status:` check above.

        return total_cost
