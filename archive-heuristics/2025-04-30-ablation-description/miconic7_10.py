from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total cost to serve all passengers by summing
    the minimum independent costs for each unserved passenger. The cost for a
    passenger includes moving the lift to their location (origin or current),
    boarding/departing, and moving the lift to their destination.

    # Assumptions
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic sums the costs for each passenger independently, which may
      overestimate the true cost as lift movements are shared. This is acceptable
      for a greedy best-first search heuristic.

    # Heuristic Initialization
    - Parses the static facts to determine the floor order and assign a numerical
      level to each floor. This allows calculating the distance between any two
      floors.
    - Extracts the destination floor for each passenger from the static facts.
    - Identifies all passengers relevant to the problem (those in goals or static facts).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the current floor of the lift from the state.
    3. Iterate through all relevant passengers identified during initialization.
    4. For each passenger `p`:
       - Check if `(served p)` is true in the current state. If yes, this passenger
         contributes 0 to the heuristic.
       - If `(served p)` is false:
         - Get the destination floor `f_destin` for passenger `p` from the pre-calculated map.
         - Check if `(boarded p)` is true in the current state.
           - If `(boarded p)` is true: The passenger is in the lift. They need to be
             dropped off at their destination `f_destin`. The minimum cost for this
             passenger, considering they are already in the lift, is the distance
             from the current lift floor to `f_destin` plus the cost of the `depart`
             action (1). Add this cost to the total.
         - Check if `(origin p f_origin)` is true in the current state.
           - If `(origin p f_origin)` is true: The passenger is waiting at `f_origin`.
             They need to be picked up at `f_origin` and then dropped off at `f_destin`.
             The minimum cost for this passenger, considering they are waiting, is
             the distance from the current lift floor to `f_origin`, plus the cost
             of the `board` action (1), plus the distance from `f_origin` to `f_destin`,
             plus the cost of the `depart` action (1). Add this cost to the total.
         - If the passenger is not served, not boarded, and not at an origin, this
           indicates an unexpected state for a passenger who is part of the goal.
           Assuming valid problem states, this case should not occur for goal passengers.
    5. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Determine floor order and levels from (above f1 f2) facts.
        # Count how many floors are *below* each floor.
        # The floor with the most floors below it is the highest.
        # The floor with 0 floors below it is the lowest.
        floor_below_counts = {}
        all_floors = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1], get_parts(fact)[2]
                all_floors.add(f_above)
                all_floors.add(f_below)
                # Increment count for the floor that is *above*
                floor_below_counts[f_above] = floor_below_counts.get(f_above, 0) + 1

        # Floors not mentioned as being above anything are the lowest.
        for floor in all_floors:
             if floor not in floor_below_counts:
                 floor_below_counts[floor] = 0 # Lowest floor(s) have 0 floors below them

        # Sort floors by the count of floors below them (ascending) to get order from lowest to highest.
        # Then assign levels starting from 1 for the lowest floor.
        sorted_floors_lowest_to_highest = sorted(floor_below_counts.keys(),
                                                 key=lambda f: floor_below_counts[f])

        self.floor_levels = {}
        level = 1
        for floor in sorted_floors_lowest_to_highest:
             self.floor_levels[floor] = level
             level += 1

        # 2. Extract passenger destinations from static facts.
        self.passenger_destinations = {}
        for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 _, passenger, destination_floor = get_parts(fact)
                 self.passenger_destinations[passenger] = destination_floor

        # 3. Identify all relevant passengers (those in goals or destinations).
        # These are the passengers whose served status matters for the goal.
        self.all_passengers = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 self.all_passengers.add(passenger)
        # Ensure passengers mentioned in destinations are also considered, even if not explicitly in goal served list
        # (though typically goal is served for all relevant passengers)
        self.all_passengers.update(self.passenger_destinations.keys())


    def get_floor_level(self, floor_name):
        """Helper to get the numerical level of a floor."""
        # Based on problem structure, all floors mentioned in above/origin/destin should have a level.
        # Accessing directly with [] will raise KeyError if a floor is missing, which is desired
        # if the problem definition is inconsistent.
        return self.floor_levels[floor_name]

    def get_distance(self, floor1_name, floor2_name):
        """Helper to calculate the distance between two floors."""
        level1 = self.get_floor_level(floor1_name)
        level2 = self.get_floor_level(floor2_name)
        return abs(level1 - level2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Check if goal is reached
        if self.goals <= state:
             return 0

        total_cost = 0

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # Should always find the lift location in a valid state
        if current_lift_floor is None:
             # This indicates an invalid state representation or domain assumption violation
             # Return infinity as this state is likely unreachable or invalid.
             return float('inf')

        # Track state of each passenger relevant to the goal
        passengers_waiting = {} # {passenger: origin_floor}
        passengers_boarded = set()
        passengers_served = set()

        # Build sets/dicts for quick lookup of passenger states
        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                passengers_waiting[passenger] = floor
            elif match(fact, "boarded", "*"):
                _, passenger = get_parts(fact)
                passengers_boarded.add(passenger)
            elif match(fact, "served", "*"):
                _, passenger = get_parts(fact)
                passengers_served.add(passenger)

        # Calculate cost for each unserved passenger that is part of the goal
        for passenger in self.all_passengers:
            if passenger in passengers_served:
                continue # Passenger is already served, cost is 0 for this passenger

            # Get the destination floor for this passenger
            destination_floor = self.passenger_destinations.get(passenger)
            # If a passenger is in self.all_passengers (e.g., from goal) but has no destination
            # in static facts, this is an inconsistent problem definition.
            # We skip this passenger or handle as an error. Assuming valid problems.
            if destination_floor is None:
                 # This should not happen in valid miconic problems where goal passengers have destinations.
                 # If it occurs, it suggests a problem definition error.
                 # For robustness, we could ignore this passenger or return inf.
                 # Let's assume valid problems and continue, although this passenger can't be served.
                 # A more robust heuristic might return inf if a goal passenger has no destination.
                 continue


            if passenger in passengers_boarded:
                # Passenger is boarded, needs to be dropped off at destination
                # Cost = Distance(current_lift_floor, destination_floor) + 1 (depart action)
                cost_for_passenger = self.get_distance(current_lift_floor, destination_floor) + 1
                total_cost += cost_for_passenger

            elif passenger in passengers_waiting:
                # Passenger is waiting at origin_floor, needs pickup and dropoff
                origin_floor = passengers_waiting[passenger]
                # Cost = Distance(current_lift_floor, origin_floor) + 1 (board action) + Distance(origin_floor, destination_floor) + 1 (depart action)
                cost_for_passenger = self.get_distance(current_lift_floor, origin_floor) + 1 + \
                                     self.get_distance(origin_floor, destination_floor) + 1
                total_cost += cost_for_passenger
            # else: The passenger is not served, not boarded, and not waiting at an origin.
            # For a passenger who is part of the goal (in self.all_passengers), this state
            # is unexpected in a standard miconic problem (implies they vanished or were
            # dropped off incorrectly). We assume valid states and problem definitions.
            # If this state occurred for a goal passenger, the heuristic might be inaccurate
            # or could return inf. Given the problem structure, this branch is effectively
            # for passengers already served (handled by the initial check) or passengers
            # not relevant to the goal (not in self.all_passengers).

        return total_cost
