# Assuming Heuristic base class is available in this path
# from heuristics.heuristic_base import Heuristic # This line might need adjustment based on actual path

# Define a dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        pass

    def __call__(self, node):
        raise NotImplementedError

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle potential empty string or non-string input
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Use fnmatch for pattern matching on each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to serve all unserved passengers.
    It sums the cost of necessary 'board' and 'depart' actions for each
    unserved passenger and adds an estimate of the minimum lift travel
    distance required to visit all floors where actions are needed.

    # Assumptions
    - Floors are ordered linearly by the 'above' predicate.
    - Each 'board', 'depart', 'up', and 'down' action costs 1.
    - The lift can carry multiple passengers.
    - The heuristic assumes the lift must visit the origin floor for waiting
      passengers and the destination floor for boarded passengers.

    # Heuristic Initialization
    - Parses 'above' facts to establish floor order and assign numerical levels.
    - Parses 'destin' facts (from static) to map passengers to their destination floors.
      Note: Passenger destinations are static in this domain.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift by finding the '(lift-at ?f)' fact in the state.
    2. Identify all passengers who are not yet 'served' by checking for '(served ?p)' facts in the state.
    3. Initialize counters for necessary 'board' and 'depart' actions to zero.
    4. Initialize a set to store the numerical levels of floors the lift *must* visit.
    5. Iterate through the state facts to find passengers who are not served:
       - If a fact is '(origin ?p ?f)' and ?p is not served:
         - Increment the 'board' action counter.
         - Add the numerical level of floor ?f to the set of required floors.
       - If a fact is '(boarded ?p)' and ?p is not served:
         - Increment the 'depart' action counter.
         - Look up the destination floor ?d for passenger ?p (stored during initialization).
         - Add the numerical level of floor ?d to the set of required floors.
    6. Calculate the minimum travel distance for the lift:
       - If the set of required floor numbers is empty (meaning all relevant passengers are served or there are no unserved passengers requiring action), the travel distance is 0.
       - Otherwise:
         - Get the numerical level of the current lift floor.
         - Find the minimum and maximum numerical levels among the required floors.
         - The minimum travel distance is estimated as the sum of:
           - The absolute difference between the current lift floor number and the number of the *nearest* required floor. This is the cost to reach the "action zone".
           - The difference between the maximum and minimum required floor numbers (the span). This is the minimum travel needed to traverse the entire range of required floors once the zone is reached.
    7. The total heuristic value is the sum of the total 'board' actions needed, the total 'depart' actions needed, and the estimated minimum travel distance.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        super().__init__(task)

        # Build floor order and number mapping from 'above' facts
        # above_map: {floor_above: floor_below}
        above_map = {}
        all_floors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                above_map[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)

        self.floor_to_number = {}
        if not all_floors:
             # No floors defined, handle gracefully (though unlikely in valid problems)
             pass
        elif len(all_floors) == 1:
             # Only one floor
             self.floor_to_number = {list(all_floors)[0]: 0}
        else:
            # Find the lowest floor: a floor that is a value in above_map but not a key
            potential_lowest = set(above_map.values()) - set(above_map.keys())
            if len(potential_lowest) != 1:
                 # This indicates a non-linear or disconnected floor structure, or no floors
                 # For miconic, we expect a single lowest floor. Fallback to sorting.
                 # print(f"Warning: Expected exactly one lowest floor, found {len(potential_lowest)}. Using sorted order.")
                 sorted_floors = sorted(list(all_floors)) # Sort alphabetically as a fallback
                 self.floor_to_number = {f: i for i, f in enumerate(sorted_floors)}
            else:
                lowest_floor = potential_lowest.pop()
                current_floor = lowest_floor
                number = 0
                self.floor_to_number[current_floor] = number

                # Build a reverse map: below_to_above_map = {floor_below: floor_above}
                below_to_above_map = {v: k for k, v in above_map.items()}

                # Traverse upwards to assign numbers
                while current_floor in below_to_above_map:
                    next_floor = below_to_above_map[current_floor]
                    number += 1
                    self.floor_to_number[next_floor] = number
                    current_floor = next_floor

        # Store goal destinations for each passenger from static facts
        self.passenger_to_destin_floor = {}
        for static_fact in self.static:
             parts = get_parts(static_fact)
             if parts and parts[0] == "destin":
                  passenger, destin_floor = parts[1], parts[2]
                  self.passenger_to_destin_floor[passenger] = destin_floor


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # If the goal is already reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        current_lift_floor = None
        served_passengers = set()
        passengers_at_origin = {} # {passenger: floor}
        boarded_passengers = set()

        # Extract relevant information from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == "lift-at":
                current_lift_floor = parts[1]
            elif predicate == "served":
                served_passengers.add(parts[1])
            elif predicate == "origin":
                passenger, floor = parts[1], parts[2]
                passengers_at_origin[passenger] = floor
            elif predicate == "boarded":
                boarded_passengers.add(parts[1])

        # Calculate the number of 'board' and 'depart' actions needed for unserved passengers
        num_board_needed = 0
        num_depart_needed = 0
        required_floor_numbers = set() # Floors the lift must visit to pick up or drop off

        # Consider passengers waiting at their origin floors
        for passenger, floor in passengers_at_origin.items():
             if passenger not in served_passengers:
                  num_board_needed += 1
                  # Add the origin floor to the set of required floors
                  if floor in self.floor_to_number:
                       required_floor_numbers.add(self.floor_to_number[floor])
                  # else: # Should not happen in valid problems if floor mapping is correct
                       # print(f"Warning: Origin floor {floor} for passenger {passenger} not found in floor map.")


        # Consider passengers currently boarded in the lift
        for passenger in boarded_passengers:
             if passenger not in served_passengers:
                  num_depart_needed += 1
                  # Add the destination floor to the set of required floors
                  if passenger in self.passenger_to_destin_floor:
                       destin_floor = self.passenger_to_destin_floor[passenger]
                       if destin_floor in self.floor_to_number:
                            required_floor_numbers.add(self.floor_to_number[destin_floor])
                       # else: # Should not happen if destin facts are in static and floor map is correct
                            # print(f"Warning: Destination floor {destin_floor} for boarded passenger {passenger} not found in floor map.")
                  # else: # Should not happen if destin facts are in static
                       # print(f"Warning: Destination not found for boarded passenger {passenger}.")


        # Calculate the minimum travel distance for the lift
        travel_distance = 0
        # Travel is only needed if there are floors to visit and the lift's current floor is mapped
        if required_floor_numbers and current_lift_floor in self.floor_to_number:
            n_current = self.floor_to_number[current_lift_floor]
            n_min = min(required_floor_numbers)
            n_max = max(required_floor_numbers)

            # Minimum distance from current floor to reach any required floor
            distance_to_nearest_required = min(abs(n_current - n) for n in required_floor_numbers)

            # Minimum distance to traverse the range of required floors
            span_distance = n_max - n_min

            # The total travel must cover the distance to reach the action zone
            # and then traverse the zone. This is a lower bound.
            travel_distance = distance_to_nearest_required + span_distance
        # else: # Debugging print if needed
             # print(f"Debug: No required floors ({len(required_floor_numbers)}), or current lift floor {current_lift_floor} not mapped.")


        # The total heuristic value is the sum of the estimated actions
        total_cost = num_board_needed + num_depart_needed + travel_distance

        return total_cost
