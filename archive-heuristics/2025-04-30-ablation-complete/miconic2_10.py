from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    # Ensure the number of parts matches the number of args for a strict match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It combines the count of unserved passengers (representing the core tasks of pickup and dropoff)
    with an estimate of the lift movement needed to reach the relevant floors.

    # Assumptions
    - Floors are ordered linearly, defined by 'above' predicates.
    - Passenger destinations are fixed and known from the initial state.
    - The cost of each action (board, depart, up, down) is 1.

    # Heuristic Initialization
    - Parses 'above' facts from static information to build a mapping of floors to integer levels,
      allowing calculation of distances between floors.
    - Parses 'destin' facts from the initial state to store the destination floor for each passenger.
    - Identifies all passengers that need to be served based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify which passengers are not yet served by checking the 'served' predicate against the goal passengers.
    2. If all goal passengers are served, the heuristic is 0.
    3. Count the number of passengers who are not served. This contributes directly to the heuristic,
       as each unserved passenger requires at least a board and a depart action sequence.
    4. Determine the current floor of the lift.
    5. Identify the set of floors where passengers are waiting ('origin' predicate). These are pickup stops.
    6. Identify the set of floors where boarded passengers need to be dropped off. These are dropoff stops.
    7. Combine pickup and dropoff floors to get the set of all 'required stops' for the lift.
    8. If there are no required stops, the estimated lift movement is 0.
    9. If there are required stops:
       - Find the integer levels for the current lift floor and all required stop floors using the pre-calculated floor levels map.
       - Calculate the minimum distance from the current lift level to any required floor level.
       - Calculate the maximum distance from the current lift level to any required floor level.
       - The estimated lift movement cost is the sum of the minimum and maximum distances. This non-admissible estimate
         encourages the lift to move towards the "zone" of activity and accounts for the span of floors it needs to cover.
    10. The total heuristic value is the sum of the number of unserved passengers and the estimated lift movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor structure, passenger destinations,
        and the set of passengers to be served.
        """
        super().__init__(task) # Call the base class constructor

        # Parse 'above' facts to build floor levels
        self.floor_levels = {}
        above_map = {} # child -> parent
        below_map = {} # parent -> child
        all_floors = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                above_map[f_below] = f_above
                below_map[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the lowest floor (a floor that is not a value in below_map)
        lowest_floor = None
        # Check if all_floors is not empty before iterating
        if all_floors:
            for floor in all_floors:
                if floor not in below_map.values():
                    lowest_floor = floor
                    break

        # Assign levels starting from the lowest floor
        if lowest_floor:
            current_floor = lowest_floor
            level = 0
            while current_floor is not None:
                self.floor_levels[current_floor] = level
                current_floor = above_map.get(current_floor)

        # Store passenger destinations (from initial state)
        self.passenger_destins = {}
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                passenger, destination_floor = parts[1], parts[2]
                self.passenger_destins[passenger] = destination_floor

        # Store the set of all passengers who need to be served (from goals)
        self.passengers_to_serve = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                self.passengers_to_serve.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify served passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # 2. If all goal passengers are served, heuristic is 0
        if self.passengers_to_serve <= served_passengers:
             return 0

        # 3. Count unserved passengers
        # This is the number of passengers in the goal set that are not in the served set
        num_passengers_not_served = len(self.passengers_to_serve - served_passengers)

        # 4. Determine current lift floor
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # Assuming current_floor is always found in a valid state
        current_level = self.floor_levels.get(current_floor)
        # Assuming current_floor is always in self.floor_levels

        # 5. Identify pickup floors
        pickup_floors = {get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}

        # 6. Identify dropoff floors
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        # Dropoff floors are destinations of boarded passengers
        dropoff_floors = {self.passenger_destins[p] for p in boarded_passengers if p in self.passenger_destins}

        # 7. Combine required stops
        required_stops = pickup_floors | dropoff_floors

        # 8. Calculate movement cost
        movement_cost = 0
        if required_stops:
            # Get levels for required stops, ensuring floors exist in our map
            required_levels = {self.floor_levels[f] for f in required_stops if f in self.floor_levels}

            # Should not be empty if required_stops was not empty and floors are valid
            if required_levels:
                min_dist_to_required = min(abs(current_level - level) for level in required_levels)
                max_dist_to_required = max(abs(current_level - level) for level in required_levels)
                movement_cost = min_dist_to_required + max_dist_to_required
            # else: movement_cost remains 0, which is correct if required_levels is unexpectedly empty

        # 10. Total heuristic
        return num_passengers_not_served + movement_cost
