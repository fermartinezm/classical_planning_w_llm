from fnmatch import fnmatch
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining cost by summing the estimated cost
    for each unserved passenger independently. The estimated cost for a passenger
    includes the actions needed to board, depart, and the lift movement required
    to travel between their origin and destination floors (or current floor and destination).

    # Assumptions
    - The lift has unlimited capacity.
    - The floor structure is linear and ordered by the 'above' predicate, defining
      a clear lowest and highest floor.
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.

    # Heuristic Initialization
    - Parses the 'above' facts from static information to determine the linear floor ordering
      and assign numerical levels to each floor, starting from 1 for the lowest floor.
    - Parses the 'destin' facts from static information to map each passenger
      to their destination floor.
    - Identifies the set of passengers that need to be served based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Identify all passengers who are not yet 'served' (i.e., '(served p)' is not in the state)
       among those listed in the goal.
    3. For each unserved passenger 'p':
       - Determine their current status: are they at their origin floor ('origin p f_origin') or are they 'boarded' ('boarded p')?
       - Get their destination floor 'f_destin' from the pre-parsed static information.
       - If the passenger is at their origin 'f_origin':
         - Estimate the cost for this passenger as:
           1 (board action) + absolute floor distance between f_origin and f_destin + 1 (depart action).
           Cost = 2 + abs(level(f_origin) - level(f_destin)).
       - If the passenger is 'boarded':
         - Estimate the cost for this passenger as:
           absolute floor distance between the current lift floor and f_destin + 1 (depart action).
           Cost = abs(level(current_lift_floor) - level(f_destin)) + 1.
       - If the passenger is neither at origin nor boarded (and not served), this indicates an unexpected state or a passenger not relevant to the goal.
    4. The total heuristic value is the sum of the estimated costs for all unserved passengers.
    5. If all passengers in the goal are served, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are facts that are true in every state.
        self.static = task.static

        # Parse floor ordering and assign levels.
        # Build a map from a floor to the floor immediately below it.
        floor_above_to_below = {}
        all_floors = set()

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, above_floor, below_floor = get_parts(fact)
                floor_above_to_below[above_floor] = below_floor
                all_floors.add(below_floor)
                all_floors.add(above_floor)

        # Find the highest floor: a floor that is in all_floors but is not a value
        # in the floor_above_to_below map (i.e., no floor is immediately above it).
        highest_floor = None
        below_floors = set(floor_above_to_below.values())
        highest_floor_candidates = all_floors - below_floors

        if len(highest_floor_candidates) == 1:
            highest_floor = highest_floor_candidates.pop()
        elif len(all_floors) == 1 and not floor_above_to_below:
             # Case with only one floor and no (above) facts
             highest_floor = list(all_floors)[0]
        else:
            # Handle cases where floor structure is not a simple linear chain starting from a unique highest floor.
            # This heuristic relies on a linear order. If not found, heuristic is invalid.
            # print("Warning: Could not determine unique highest floor from static facts. Floor ordering may be ambiguous or non-linear.")
            self.floor_levels = {} # Indicate failure to parse levels
            highest_floor = None # Ensure we don't proceed with level assignment

        self.floor_levels = {}
        if highest_floor is not None:
            current_floor = highest_floor
            level = len(all_floors) if all_floors else 0 # Assign highest level to highest floor
            # Traverse downwards, assigning levels
            while current_floor is not None and level > 0: # Add level > 0 check for safety
                self.floor_levels[current_floor] = level
                # Find the floor immediately below the current_floor
                next_floor = floor_above_to_below.get(current_floor)
                current_floor = next_floor
                level -= 1

        # Parse passenger destinations from 'destin' facts in static.
        self.passenger_destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.passenger_destinations[passenger] = destination_floor

        # Identify all passengers that need to be served (from goal).
        self.passengers_to_serve = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 self.passengers_to_serve.add(passenger)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # If floor levels weren't parsed correctly, return 0 (uninformative but safe).
        # Only return 0 if there are passengers to serve but we can't estimate.
        if not self.floor_levels and self.passengers_to_serve:
             # print("Heuristic returning 0 due to failed floor level parsing.")
             return 0
        # If there are no passengers to serve, the goal is met or trivial.
        if not self.passengers_to_serve:
             return 0


        # Check if goal is reached (redundant if self.passengers_to_serve check is thorough,
        # but good defensive programming).
        if self.goals <= state:
            return 0

        total_estimated_cost = 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        if current_lift_floor is None:
             # This should not happen in a valid miconic state.
             # print("Warning: Lift location not found in state. Returning 0.")
             return 0

        current_lift_level = self.floor_levels.get(current_lift_floor)
        if current_lift_level is None:
             # This should not happen if floor_levels was built correctly from static facts.
             # print(f"Warning: Lift floor '{current_lift_floor}' not found in floor levels map. Returning 0.")
             return 0

        # Track passengers by their current status (only those relevant to the goal)
        passengers_at_origin = {} # {passenger: origin_floor}
        passengers_boarded = set()
        served_passengers_in_state = set()

        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, passenger, origin_floor = get_parts(fact)
                if passenger in self.passengers_to_serve:
                    passengers_at_origin[passenger] = origin_floor
            elif match(fact, "boarded", "*"):
                _, passenger = get_parts(fact)
                if passenger in self.passengers_to_serve:
                    passengers_boarded.add(passenger)
            elif match(fact, "served", "*"):
                 _, passenger = get_parts(fact)
                 served_passengers_in_state.add(passenger)


        # Compute estimated cost for each unserved passenger from the goal list
        for passenger in self.passengers_to_serve:
            # Check if the passenger is already served in the current state
            if passenger in served_passengers_in_state:
                continue # This passenger is done

            destination_floor = self.passenger_destinations.get(passenger)
            if destination_floor is None:
                 # This passenger is in the goal but has no destination in static facts.
                 # Should not happen in valid problems.
                 # print(f"Warning: Destination not found for goal passenger '{passenger}'. Skipping.")
                 continue

            destination_level = self.floor_levels.get(destination_floor)
            if destination_level is None:
                 # Destination floor not in our parsed levels.
                 # print(f"Warning: Destination floor '{destination_floor}' not found in floor levels map. Skipping passenger '{passenger}'.")
                 continue

            if passenger in passengers_at_origin:
                # Passenger is waiting at origin
                origin_floor = passengers_at_origin[passenger]
                origin_level = self.floor_levels.get(origin_floor)
                if origin_level is None:
                     # print(f"Warning: Origin floor '{origin_floor}' not found in floor levels map. Skipping passenger '{passenger}'.")
                     continue

                # Estimated cost: board (1) + travel origin->destin + depart (1)
                # Travel cost is floor distance.
                estimated_passenger_cost = 1 + abs(origin_level - destination_level) + 1
                total_estimated_cost += estimated_passenger_cost

            elif passenger in passengers_boarded:
                # Passenger is boarded
                # Estimated cost: travel current->destin + depart (1)
                estimated_passenger_cost = abs(current_lift_level - destination_level) + 1
                total_estimated_cost += estimated_passenger_cost

            # If a passenger is in self.passengers_to_serve but not in served_passengers_in_state,
            # and also not in passengers_at_origin or passengers_boarded, this is an inconsistent state.
            # The heuristic implicitly assumes passengers are either at origin, boarded, or served.
            # We don't add cost for such passengers, effectively ignoring them if they are in an unknown state.

        return total_estimated_cost
