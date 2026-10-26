from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Splits a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

# Helper function to match PDDL fact patterns
def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the remaining cost by summing three components:
        1. The number of 'board' actions needed (waiting passengers).
        2. The number of 'depart' actions needed (boarded passengers).
        3. An estimate of the lift movement actions required to visit
           all necessary floors (origins of waiting passengers and
           destinations of boarded passengers). The movement cost is
           estimated as the distance from the current lift floor to
           the closest end of the range of necessary floors, plus the
           width of that range.

    Assumptions:
        - Floors are ordered linearly and consecutively, as defined by
          the (above ?f_higher ?f_lower) facts. Specifically, (above f_higher f_lower)
          means f_higher is a floor located above f_lower. The set of (above) facts
          defines a total order on floors.
        - Passenger origins and destinations are static and provided
          in the initial state (as static facts).
        - Each passenger has exactly one origin and one destination.
        - The goal is to serve all passengers listed in the goal.

    Heuristic Initialization:
        - Parses the (above) facts to determine the floor order and
          create a mapping from floor names to integer indices. The
          floor with 0 floors below it (based on (above) facts) gets
          index 0, the next gets index 1, and so on. This is determined
          by counting how many floors appear as the 'higher' floor in
          (above) facts for each floor.
        - Parses the (origin) and (destin) facts from static information
          to store the origin and destination floor for each passenger.
        - Identifies the set of passengers that need to be served based on the goal facts.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the state is a goal state. If yes, return 0.
        2. Identify the current floor of the lift from the state. Convert
           the floor name to its integer index using the precomputed mapping.
        3. Initialize counts for waiting passengers and boarded passengers to 0.
        4. Initialize a set to store the indices of all floors that need
           to be visited (origin floors for waiting passengers and
           destinations of boarded passengers).
        5. Iterate through the set of passengers that need to be served.
        6. For each such passenger:
           a. Check if the passenger is already served (by checking for
              '(served passenger_name)' in the state). If yes, this
              passenger contributes 0 to the heuristic.
           b. If not served, check if the passenger is boarded (by checking
              for '(boarded passenger_name)' in the state). If yes:
              - Increment the boarded passenger count.
              - Add the index of their destination floor to the set of
                necessary floor indices.
           c. If not served and not boarded, the passenger must be waiting
              at their origin. Check if '(origin passenger_name origin_floor)'
              is in the state. If yes:
              - Increment the waiting passenger count.
              - Add the index of their origin floor to the set of
                necessary floor indices.
                 # Note: If a passenger is not served, not boarded, and not at their origin,
                 # this indicates an inconsistent state not expected in valid problem traces.
                 # We ignore such passengers for the heuristic calculation.
        7. Calculate the estimated movement cost:
           a. If the set of necessary floor indices is empty, the movement
              cost is 0 (all remaining passengers are either served or
              boarded and at the current floor, or all passengers are served).
           b. If the set is not empty, find the minimum and maximum floor
              indices in the set.
           c. The estimated movement cost is the distance from the current
              lift floor index to the closest of the min/max necessary floor
              indices, plus the distance between the min and max necessary
              floor indices. This is calculated as:
              `min(abs(current_lift_floor_idx - min_needed_idx), abs(current_lift_floor_idx - max_needed_idx)) + (max_needed_idx - min_needed_idx)`
        8. The total heuristic value is the sum of the waiting passenger count,
           the boarded passenger count, and the estimated movement cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse floor order and create floor_name -> index mapping
        # (above f_higher f_lower) means f_higher is above f_lower.
        # Count how many floors are below each floor to determine its index.
        floor_names = set()
        # Count how many times a floor appears as the *higher* floor in an (above) fact
        # This count corresponds to the number of floors below it.
        floors_below_count = {} # floor_name -> count of floors below it

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                floor_names.add(f_higher)
                floor_names.add(f_lower)
                floors_below_count[f_higher] = floors_below_count.get(f_higher, 0) + 1

        # Sort floors based on the count of floors below them (ascending).
        # The floor with count 0 is the lowest (index 0).
        sorted_floors = sorted(floor_names, key=lambda f: floors_below_count.get(f, 0))
        self.floor_to_index = {floor: i for i, floor in enumerate(sorted_floors)}
        # self.index_to_floor = {i: floor for floor, i in self.floor_to_index.items()} # Not strictly needed for heuristic, but useful

        # 2. Parse passenger origins and destinations
        self.passenger_info = {} # passenger_name -> {'origin': floor_name, 'destin': floor_name}

        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                if passenger not in self.passenger_info:
                    self.passenger_info[passenger] = {}
                self.passenger_info[passenger]['origin'] = floor
            elif match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                if passenger not in self.passenger_info:
                    self.passenger_info[passenger] = {}
                self.passenger_info[passenger]['destin'] = floor

        # 3. Identify passengers that need to be served (from goal)
        self.passengers_to_serve = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
             return 0

        # 2. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break
        # In a valid state, the lift must be somewhere.
        # assert current_lift_floor is not None, "Lift location not found in state"
        current_lift_floor_idx = self.floor_to_index[current_lift_floor]

        # 3. Identify unserved passengers and necessary floors
        waiting_passengers_count = 0
        boarded_passengers_count = 0
        needed_floor_indices = set() # Floors the lift must visit

        # Iterate through all passengers that need to be served according to the goal
        for passenger in self.passengers_to_serve:
            # Check if passenger info is complete (should be if PDDL is valid)
            if passenger not in self.passenger_info or 'origin' not in self.passenger_info[passenger] or 'destin' not in self.passenger_info[passenger]:
                 # This passenger is in the goal but static info is missing. Skip.
                 continue

            origin_floor = self.passenger_info[passenger]['origin']
            destin_floor = self.passenger_info[passenger]['destin']

            # Check passenger status in the current state
            is_served = f'(served {passenger})' in state
            is_boarded = f'(boarded {passenger})' in state
            is_waiting_at_origin = f'(origin {passenger} {origin_floor})' in state

            if is_served:
                # Passenger is served, contributes 0 to heuristic
                continue

            # Passenger is not served
            if is_boarded:
                # Passenger is boarded, needs to be dropped off at destination
                boarded_passengers_count += 1
                needed_floor_indices.add(self.floor_to_index[destin_floor])
            elif is_waiting_at_origin:
                 # Passenger is not served and not boarded, must be waiting at origin
                 waiting_passengers_count += 1
                 needed_floor_indices.add(self.floor_to_index[origin_floor])
            # else: Passenger is not served, not boarded, and not at origin.
            # This indicates an inconsistent state not expected in valid problem traces.
            # We ignore such passengers for the heuristic calculation.


        # 4. Calculate estimated movement cost
        movement_cost = 0
        if needed_floor_indices:
            min_needed_idx = min(needed_floor_indices)
            max_needed_idx = max(needed_floor_indices)

            # Cost to reach the closest end of the needed range
            cost_to_reach_range = min(abs(current_lift_floor_idx - min_needed_idx),
                                      abs(current_lift_floor_idx - max_needed_idx))

            # Cost to traverse the entire needed range
            cost_to_traverse_range = max_needed_idx - min_needed_idx

            movement_cost = cost_to_reach_range + cost_to_traverse_range

        # 5. Total heuristic = board actions + depart actions + movement actions
        # Each waiting passenger needs 1 board action.
        # Each boarded passenger needs 1 depart action.
        # Movement cost is estimated as calculated above.
        total_heuristic = waiting_passengers_count + boarded_passengers_count + movement_cost

        return total_heuristic
