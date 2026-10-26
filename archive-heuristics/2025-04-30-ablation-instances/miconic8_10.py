from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of necessary board and depart actions for unserved passengers
    and adds the minimum vertical travel distance the lift must cover to visit
    all floors where pickups or dropoffs are required, starting from the current floor.

    # Assumptions
    - Unit cost for all actions (board, depart, move between adjacent floors).
    - The floor structure is linear, defined by the `above` predicate.
    - The static facts contain `(above f_higher f_lower)` for all pairs of floors
      where `f_higher` is vertically above `f_lower`. This allows determining
      the relative vertical order of all floors.
    - All passengers who need to be served are mentioned in `(destin p d)` static facts.
    - In any reachable state, an unserved passenger is either waiting at their origin
      floor (`(origin p o)`) or is boarded in the lift (`(boarded p)`).

    # Heuristic Initialization
    - Parses static facts to build a mapping from floor names to integer ranks.
      The floor with the most floors above it is assigned rank 1 (lowest),
      and the floor with no floors above it is assigned the highest rank.
    - Extracts passenger destinations from static facts.
    - Identifies the set of all passengers who need to be served based on destinations.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Check if the state is a goal state (all passengers served). If yes, return 0.
    2.  Identify the current floor of the lift.
    3.  Initialize counters for `board` actions needed and `depart` actions needed to 0.
    4.  Initialize sets for required pickup floors and required dropoff floors as empty.
    5.  Iterate through all passengers who need to be served (unserved passengers):
        -   Check if the passenger is currently waiting at their origin floor in the state (`(origin p o)`).
            -   If yes, increment `board` actions needed by 1. Add the origin floor `o` to the set of required pickup floors. Add the passenger's destination floor `d` (looked up from initialization data) to the set of required dropoff floors (as they will eventually need to be dropped off).
        -   Check if the passenger is currently boarded in the lift in the state (`(boarded p)`).
            -   If yes, increment `depart` actions needed by 1. Add the passenger's destination floor `d` to the set of required dropoff floors.
    6.  Calculate the total base actions needed: `board_actions_needed + depart_actions_needed`.
    7.  Combine the current lift floor, all required pickup floors, and all required dropoff floors into a single set of floors that the lift must visit.
    8.  Map this set of floors to their integer ranks using the floor mapping created during initialization.
    9.  Find the minimum and maximum ranks among this set of relevant floors.
    10. Calculate the minimum vertical travel cost: `max_rank - min_rank`. This represents the minimum vertical distance the lift must span to cover all necessary floors.
    11. The total heuristic value is the sum of the total base actions and the minimum vertical travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ranking and passenger destination information.
        """
        self.goals = task.goals # Goal conditions

        # Build floor ranking map from static facts
        # Find all floors mentioned in 'above' facts
        all_above_facts = [get_parts(fact) for fact in task.static if get_parts(fact)[0] == 'above']
        floors = set()
        for fact_parts in all_above_facts:
             floors.add(fact_parts[1]) # f_higher
             floors.add(fact_parts[2]) # f_lower

        # Count how many floors are above each floor based on 'above' facts
        # (above f_higher f_lower) means f_higher is higher than f_lower
        # For a floor `f`, count how many `f_other` exist such that (above f_other f) is true.
        # This count represents how many floors are *strictly* above `f`.
        floors_above_count = {f: 0 for f in floors}
        for f_lower in floors:
             for fact_parts in all_above_facts:
                 f_higher_in_fact = fact_parts[1]
                 f_lower_in_fact = fact_parts[2]
                 if f_lower_in_fact == f_lower:
                     # f_higher_in_fact is above f_lower
                     floors_above_count[f_lower] += 1

        # The floor with count 0 is the highest. The floor with count N-1 is the lowest.
        # Rank 1 is the lowest floor. Rank N is the highest floor.
        num_floors = len(floors)
        self.floor_to_rank = {}
        # Handle case with 0 floors (shouldn't happen in valid problems)
        if num_floors > 0:
            for floor, count_above in floors_above_count.items():
                # Rank = (num_floors - count_above)
                self.floor_to_rank[floor] = num_floors - count_above
        # else: self.floor_to_rank remains empty, heuristic will handle this

        # Store passenger destinations from static facts
        self.passenger_destinations = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                passenger, destination = parts[1], parts[2]
                self.passenger_destinations[passenger] = destination

        # Get the set of all passengers who need to be served (those with destinations)
        self.all_passengers = set(self.passenger_destinations.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Identify current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        # If no lift location found or no floors defined, heuristic is infinite (or large)
        # This shouldn't happen in valid miconic problems.
        if current_lift_floor is None or not self.floor_to_rank:
             # print("Warning: Cannot compute heuristic, lift location or floor ranks missing.")
             return float('inf') # Should not be reachable in valid problems

        current_lift_rank = self.floor_to_rank[current_lift_floor]

        # Identify unserved passengers and their state
        served_passengers = set()
        waiting_passengers = {} # {passenger: origin_floor}
        boarded_passengers = set() # {passenger}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served':
                served_passengers.add(parts[1])
            elif parts[0] == 'origin':
                passenger, origin_floor = parts[1], parts[2]
                waiting_passengers[passenger] = origin_floor
            elif parts[0] == 'boarded':
                boarded_passengers.add(parts[1])

        # Determine unserved passengers
        unserved_passengers = self.all_passengers - served_passengers

        # Calculate base actions (board + depart) and identify required floors
        required_pickup_floors = set()
        required_dropoff_floors = set()
        board_actions_needed = 0
        depart_actions_needed = 0

        for passenger in unserved_passengers:
            # A passenger must be either waiting or boarded if unserved in a valid state
            if passenger in waiting_passengers:
                # Passenger is waiting at origin
                board_actions_needed += 1
                origin_floor = waiting_passengers[passenger]
                required_pickup_floors.add(origin_floor)
                # This passenger will eventually need to depart at destination
                # Ensure destination exists (should be guaranteed by problem definition)
                if passenger in self.passenger_destinations:
                    destination_floor = self.passenger_destinations[passenger]
                    required_dropoff_floors.add(destination_floor)
            elif passenger in boarded_passengers:
                # Passenger is boarded
                depart_actions_needed += 1
                # Ensure destination exists
                if passenger in self.passenger_destinations:
                    destination_floor = self.passenger_destinations[passenger]
                    required_dropoff_floors.add(destination_floor)
            # else: unserved passenger is neither waiting nor boarded - this state is unexpected
            # in a standard miconic problem but might occur with invalid transitions.
            # We ignore such passengers for the heuristic calculation as we don't know
            # where they are or what they need. This might make the heuristic non-zero
            # for states that are effectively unsolvable from that point for that passenger,
            # which is acceptable for a non-admissible heuristic.


        total_base_actions = board_actions_needed + depart_actions_needed

        # Calculate vertical travel cost
        floors_to_visit = required_pickup_floors.union(required_dropoff_floors)

        if not floors_to_visit:
            # No pickups or dropoffs needed for unserved passengers.
            # This should only happen if all passengers are served (handled above),
            # or if unserved passengers are in an unexpected state (ignored above).
            # If it somehow happens and unserved_passengers is not empty,
            # the heuristic will be total_base_actions (which is 0 if no board/depart needed) + 0.
            # This is okay.
            vertical_travel_cost = 0
        else:
            # Include current lift floor in the range calculation
            all_relevant_floors = floors_to_visit.union({current_lift_floor})
            relevant_ranks = [self.floor_to_rank[f] for f in all_relevant_floors if f in self.floor_to_rank] # Defensive check
            if not relevant_ranks: # Should not happen if floors_to_visit is not empty and current_lift_floor is valid
                 vertical_travel_cost = 0
            else:
                min_rank = min(relevant_ranks)
                max_rank = max(relevant_ranks)
                vertical_travel_cost = max_rank - min_rank

        # Total heuristic value
        heuristic_value = total_base_actions + vertical_travel_cost

        return heuristic_value
