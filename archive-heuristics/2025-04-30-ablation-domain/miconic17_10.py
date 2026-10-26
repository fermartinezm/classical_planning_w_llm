import re
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and empty facts
    if not fact or fact.strip() == '()':
        return []
    return fact.strip()[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the number of 'board' actions needed (for unboarded passengers),
    the number of 'depart' actions needed (for boarded passengers), and
    estimates the minimum floor travel required to visit all necessary floors
    (origins of unboarded passengers and destinations of boarded passengers).

    # Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < f3).
      The heuristic parses the number from floor names like 'f<number>'.
    - The lift can carry multiple passengers.
    - Actions have a cost of 1.

    # Heuristic Initialization
    - Extracts all floor names from static facts and initial state and maps them
      to integer levels based on their numerical suffix.
    - Stores the destination floor for each passenger by parsing static facts.
    - Identifies all passengers in the problem from destination facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Lift Location:** Find the current floor of the lift from the state
        and get its corresponding integer level using the pre-calculated mapping.
    2.  **Identify Passenger States:** Iterate through the state facts to determine
        which passengers are currently `served`, `boarded`, or waiting at their `origin`.
    3.  **Categorize Unserved Passengers:** Based on the total list of passengers
        (identified during initialization from destination facts) and the `served`
        facts in the state, find all unserved passengers. Further categorize unserved
        passengers into `unboarded` (those still at their origin) and `boarded`
        (those inside the lift).
    4.  **Count Required Actions (Passenger State):**
        -   Count the number of `unboarded` passengers. Each requires a `board` action.
        -   Count the number of `boarded` passengers. Each requires a `depart` action.
        These counts contribute directly to the heuristic value as they represent
        actions that must eventually happen for each passenger.
    5.  **Identify Required Stop Floors:** Determine the set of floors the lift
        *must* visit to make progress towards serving passengers:
        -   For every `unboarded` passenger, their origin floor must be visited.
        -   For every `boarded` passenger, their destination floor must be visited.
        Collect the integer levels for all these required stop floors.
    6.  **Estimate Floor Travel Cost:**
        -   If there are no required stop floors (meaning all unserved passengers
            are already at their destination floors, i.e., served), the travel cost is 0.
        -   Otherwise, find the minimum and maximum floor levels among the required stops.
        -   Calculate the estimated travel cost as the minimum distance from the
            current lift level to either the minimum or maximum required level,
            *plus* the total range of required levels (maximum level - minimum level).
            This estimates the travel needed to reach the "action zone" defined by
            the required stops and traverse it.
    7.  **Calculate Total Heuristic:** The total heuristic value is the sum of the
        counts from step 4 (number of unboarded + number of boarded) and the
        estimated floor travel cost from step 6. This combines the individual
        passenger handling actions with the necessary lift movement.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals  # Goals are used to check if a state is the goal state (h=0)
        self.static = task.static  # Static facts contain 'above' and 'destin'

        self.destinations = {}
        floor_names = set()
        all_passengers = set()

        # Extract destinations and collect all passenger and floor names from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts
            if parts[0] == "destin":
                # Fact format: (destin passenger floor)
                if len(parts) == 3:
                    passenger, floor = parts[1], parts[2]
                    self.destinations[passenger] = floor
                    all_passengers.add(passenger)
                    floor_names.add(floor)
            elif parts[0] == "above":
                 # Fact format: (above floor_above floor_below)
                 if len(parts) == 3:
                     floor_names.add(parts[1])
                     floor_names.add(parts[2])

        # Also collect passengers and floors from initial state if not already found
        # (e.g., from 'origin' or 'lift-at')
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue # Skip empty facts
             if parts[0] == "origin":
                 # Fact format: (origin passenger floor)
                 if len(parts) == 3:
                     passenger, floor = parts[1], parts[2]
                     all_passengers.add(passenger)
                     floor_names.add(floor)
             elif parts[0] == "lift-at":
                 # Fact format: (lift-at floor)
                 if len(parts) == 2:
                     floor_names.add(parts[1])
             elif parts[0] == "boarded":
                 # Fact format: (boarded passenger)
                 if len(parts) == 2:
                     all_passengers.add(parts[1])
             elif parts[0] == "served":
                 # Fact format: (served passenger)
                 if len(parts) == 2:
                     all_passengers.add(parts[1])

        self.all_passengers = frozenset(all_passengers)

        # Map floor names to levels (assuming f<number> format)
        # Sort floors based on the number in their name
        def get_floor_number(floor_name):
            match = re.search(r'\d+', floor_name)
            if match:
                return int(match.group())
            # If floor name doesn't match f<number>, this is unexpected based on examples.
            # Raising an error indicates a problem with the input format.
            raise ValueError(f"Floor name '{floor_name}' does not contain a number.")

        try:
            # Ensure floor_names is not empty before sorting
            if not floor_names:
                 self.floor_to_level = {}
            else:
                sorted_floors = sorted(list(floor_names), key=get_floor_number)
                self.floor_to_level = {floor: i + 1 for i, floor in enumerate(sorted_floors)}
        except ValueError as e:
             # Propagate the error if floor names cannot be parsed
             raise e


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached - this is the most efficient way to get h=0
        if self.goals <= state:
            return 0

        current_lift_floor = None
        served_passengers = set()
        boarded_passengers_in_state = set() # Passengers with (boarded p) fact
        origin_locations = {} # {passenger: floor} for passengers with (origin p f) fact

        # Parse the current state to find lift location and passenger states
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts
            if parts[0] == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
            elif parts[0] == "served" and len(parts) == 2:
                served_passengers.add(parts[1])
            elif parts[0] == "boarded" and len(parts) == 2:
                boarded_passengers_in_state.add(parts[1])
            elif parts[0] == "origin" and len(parts) == 3:
                origin_locations[parts[1]] = parts[2]

        # Identify unserved passengers
        unserved_passengers = self.all_passengers - served_passengers

        # Identify unboarded and boarded passengers among the unserved ones
        unboarded_passengers = {p for p in unserved_passengers if p in origin_locations}
        boarded_passengers = {p for p in unserved_passengers if p in boarded_passengers_in_state}

        num_unboarded = len(unboarded_passengers)
        num_boarded = len(boarded_passengers)

        # Identify required stop levels
        required_levels = set()
        # Add origin levels for unboarded passengers
        for p in unboarded_passengers:
            f_origin = origin_locations.get(p) # Use .get for safety, though logic implies it exists
            if f_origin and f_origin in self.floor_to_level:
                 required_levels.add(self.floor_to_level[f_origin])

        # Add destination levels for boarded passengers
        for p in boarded_passengers:
            f_destin = self.destinations.get(p) # Use .get for safety
            if f_destin and f_destin in self.floor_to_level:
                 required_levels.add(self.floor_to_level[f_destin])

        # Calculate estimated floor travel
        travel_cost = 0
        # Travel is only needed if there are floors to visit and the lift location is known and valid
        if required_levels and current_lift_floor in self.floor_to_level:
            current_level = self.floor_to_level[current_lift_floor]
            min_level = min(required_levels)
            max_level = max(required_levels)

            # Estimated travel = distance to closest required floor + range of required floors
            # This estimates the travel needed to reach the 'action zone' [min_level, max_level]
            # and then traverse it.
            dist_to_min = abs(current_level - min_level)
            dist_to_max = abs(current_level - max_level)
            travel_cost = min(dist_to_min, dist_to_max) + (max_level - min_level)
        # If required_levels is empty, travel_cost remains 0.
        # If current_lift_floor is not found in floor_to_level, travel_cost remains 0,
        # which might underestimate but prevents errors on malformed states.

        # Total heuristic is sum of actions needed per passenger state + estimated travel
        # Each unboarded passenger needs board (1) + travel + depart
        # Each boarded passenger needs travel + depart (1)
        # The travel cost is shared.
        # Heuristic = (Num unboarded passengers) + (Num boarded passengers) + Estimated travel
        # This counts the 'board' and 'depart' actions needed for unserved passengers
        # and adds the estimated travel to reach the necessary floors.

        heuristic_value = num_unboarded + num_boarded + travel_cost

        return heuristic_value
