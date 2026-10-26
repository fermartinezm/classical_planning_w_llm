# Add necessary imports
# from fnmatch import fnmatch # Not strictly needed for this implementation
from heuristics.heuristic_base import Heuristic # Assuming this base class exists
import re # To extract numbers from floor names

# Helper function to parse PDDL facts
def parse_fact(fact_string):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Remove leading/trailing parentheses and split by spaces
    return fact_string[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort by summing the number of
    passengers who still need to be served and the vertical span of the floors
    involved in serving them (current lift floor, origin floors of waiting
    passengers, and destination floors of boarded passengers).

    # Assumptions
    - Floor names are structured as 'f' followed by an integer (e.g., 'f1', 'f10').
      The integer represents the floor level, with higher numbers being higher floors.
    - The goal is to have all passengers served.
    - The 'above' facts define the relative order of floors, consistent with the
      numerical suffix of floor names.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static 'destin' facts.
    - Builds a mapping from floor names (e.g., 'f1', 'f2') to integer floor levels
      based on the numerical suffix in their names. This assumes a linear floor structure.
    - Identifies all passengers present in the problem instance based on 'destin' facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current location of the lift.
    2. Identify all passengers who have not yet been served.
    3. For each unserved passenger:
       - If the passenger is waiting at their origin floor (predicate 'origin'), add their origin floor
         to the set of floors the lift needs to visit.
       - If the passenger is boarded in the lift (predicate 'boarded'), add their destination floor
         to the set of floors the lift needs to visit.
    4. Count the total number of unserved passengers. This count contributes directly
       to the heuristic value (representing the number of board/depart actions needed).
    5. Determine the set of all relevant floor levels: the level of the current
       lift floor, and the levels of all floors identified in step 3.
    6. Calculate the vertical span of these relevant floors: the difference between
       the maximum and minimum level in the set from step 5. This span represents
       a minimum vertical distance the lift must cover.
    7. The total heuristic value is the sum of the count of unserved passengers
       and the calculated vertical span.
    8. If all passengers are served, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor levels based on floor names.
        - Passenger destinations.
        - List of all passengers.
        """
        self.goals = task.goals # Goal conditions (used implicitly to find all passengers)
        static_facts = task.static

        self.floor_levels = {}
        self.passenger_destinations = {}
        self.all_passengers = set()

        # Collect all floor names from static facts and determine levels based on numerical suffix
        floor_names = set()
        for fact in static_facts:
            parts = parse_fact(fact)
            if len(parts) > 1: # Ensure fact has enough parts
                if parts[0] == 'above':
                    # Facts like (above f1 f2), (above f1 f3), etc.
                    # We assume floor names are 'f' followed by a number.
                    if len(parts) == 3:
                        floor_names.add(parts[1])
                        floor_names.add(parts[2])
                elif parts[0] == 'destin':
                    # Facts like (destin p1 f2)
                    if len(parts) == 3:
                        self.passenger_destinations[parts[1]] = parts[2]
                        self.all_passengers.add(parts[1])
                        floor_names.add(parts[2])

        # Extract level from floor name (e.g., 'f1' -> 1)
        for f_name in floor_names:
            # Use regex to find all digits in the floor name
            digit_match = re.search(r'\d+', f_name)
            if digit_match:
                try:
                    level = int(digit_match.group(0))
                    self.floor_levels[f_name] = level
                except ValueError:
                     # Should not happen if regex found digits, but for safety
                     print(f"Warning: Could not convert digit part to integer for floor: {f_name}")
            else:
                 # Handle floor names without numbers if necessary, though examples don't show this
                 print(f"Warning: Could not find digit part in floor name: {f_name}")


        # Ensure all passengers mentioned in goals are included (though destin facts should cover this)
        # This loop is mostly redundant if all goal passengers have destin facts, but adds robustness.
        for goal in self.goals:
             parts = parse_fact(goal)
             if parts and parts[0] == 'served' and len(parts) > 1:
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        current_lift_floor = None
        unserved_passengers_count = 0
        required_floors = set() # Floors the lift must visit (origins for waiting, destins for boarded)

        # Parse state facts into dictionaries/sets for easier lookup
        state_facts_dict = {}
        for fact in state:
            parts = parse_fact(fact)
            if parts: # Ensure fact is not empty after parsing
                predicate = parts[0]
                args = tuple(parts[1:])
                if predicate not in state_facts_dict:
                    state_facts_dict[predicate] = {}
                # Store facts allowing quick lookup, e.g., state_facts_dict['origin']['p1'] = 'f6'
                # For predicates like lift-at or served, just store presence or value
                if predicate == 'lift-at' and len(args) == 1:
                    current_lift_floor = args[0]
                elif predicate == 'served' and len(args) == 1:
                    state_facts_dict[predicate][args[0]] = True # Just mark passenger as served
                elif predicate == 'origin' and len(args) == 2:
                     state_facts_dict[predicate][args[0]] = args[1] # Store origin floor for passenger
                elif predicate == 'boarded' and len(args) == 1:
                     state_facts_dict[predicate][args[0]] = True # Just mark passenger as boarded
                # Add other predicates if needed for heuristic, but these seem sufficient


        # Check each passenger's status
        served_passengers = state_facts_dict.get('served', {})
        origin_passengers = state_facts_dict.get('origin', {})
        boarded_passengers = state_facts_dict.get('boarded', {})

        for passenger in self.all_passengers:
            if passenger not in served_passengers:
                unserved_passengers_count += 1

                if passenger in origin_passengers:
                    # Passenger is waiting at origin
                    origin_floor = origin_passengers[passenger]
                    required_floors.add(origin_floor)
                elif passenger in boarded_passengers:
                    # Passenger is boarded
                    destination_floor = self.passenger_destinations.get(passenger)
                    if destination_floor: # Should always exist for unserved boarded passenger
                         required_floors.add(destination_floor)
                    # else: # Defensive: if boarded but no destination? Invalid state?
                    #     pass # Assuming valid states

        # If all passengers are served, heuristic is 0
        if unserved_passengers_count == 0:
            return 0

        # Calculate move cost based on the span of required floors and current lift floor
        relevant_levels = set()

        # Add current lift floor level
        if current_lift_floor and current_lift_floor in self.floor_levels:
             relevant_levels.add(self.floor_levels[current_lift_floor])
        # else: # Defensive: current lift floor not in floor_levels? Invalid state?
        #     pass # Assuming valid states

        # Add required floor levels
        for floor in required_floors:
             if floor in self.floor_levels:
                  relevant_levels.add(self.floor_levels[floor])
             # else: # Defensive: required floor not in floor_levels? Invalid state?
             #     pass # Assuming valid states

        move_cost = 0
        # relevant_levels might be empty if, e.g., unserved passengers are in an unexpected state
        # (not origin, not boarded, not served). Assuming valid states, this won't be empty
        # if unserved_passengers_count > 0.
        if relevant_levels:
            min_level = min(relevant_levels)
            max_level = max(relevant_levels)
            move_cost = max_level - min_level
        # else: # Defensive: if relevant_levels is empty but unserved_count > 0
        #     # This state is likely invalid or represents a problem not fitting the heuristic's assumptions.
        #     # Returning unserved_passengers_count provides a lower bound.
        #     move_cost = 0 # Or some penalty? Let's stick to 0 for simplicity.


        # Total heuristic is the sum of unserved passengers (boarding/departing actions)
        # and the minimum vertical travel span required.
        total_heuristic = unserved_passengers_count + move_cost

        return total_heuristic
