from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort by summing the number of unserved passengers
    (weighted by whether they still need boarding) and the estimated vertical travel distance
    the lift must cover to visit all floors where passengers need to be picked up or dropped off.

    # Assumptions
    - The 'above' predicate defines a linear order of floors.
    - Passenger destinations are static facts defined in the initial state.
    - Passenger initial origins are defined in the initial state.
    - Each unserved passenger requires a board action (if not already boarded) and a depart action.
    - The lift can pick up/drop off multiple passengers at the same floor visit.

    # Heuristic Initialization
    - Precomputes the floor levels based on the 'above' predicates to quickly calculate vertical distances.
    - Stores the initial origin and static destination floors for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all passengers who have not yet reached their destination ('served').
    3. Count the number of unserved passengers who are currently at their origin floor (`num_at_origin`).
    4. Count the number of unserved passengers who are currently boarded (`num_boarded_unserved`).
    5. The base action cost is estimated as `(num_at_origin * 2) + num_boarded_unserved`. This counts 2 actions (board + depart) for passengers still waiting at their origin and 1 action (depart) for passengers already boarded.
    6. Determine the set of 'required floors' the lift must visit:
       - For each unserved passenger currently at their origin floor, add their origin floor to the set of required floors (for pickup).
       - For each unserved passenger currently boarded, add their destination floor to the set of required floors (for dropoff).
       - Note: If a passenger is at their origin, both their origin (pickup) and destination (dropoff) floors are required visits.
    7. If there are no required floors (which should only happen if all passengers are served), the travel cost is 0.
    8. If there are required floors, calculate the estimated lift travel cost:
       - Find the minimum and maximum floor levels among the required floors using the precomputed floor levels.
       - The estimated travel cost is the vertical distance spanning from the current lift floor to the lowest required floor and up to the highest required floor. This is calculated as `max(current_level, max_required_level) - min(current_level, min_required_level)`. This represents the minimum vertical range the lift must traverse.
    9. The total heuristic value is the sum of the base action cost and the estimated lift travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger info.
        """
        # Store passenger origin and destination floors
        self.passenger_info = {}
        # Also collect all floors and 'above' relations
        floor_below = {}
        all_floors = set()

        # Get static facts (destin, above, initial lift-at)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                p, f_d = parts[1], parts[2]
                if p not in self.passenger_info:
                    self.passenger_info[p] = {}
                self.passenger_info[p]['destin'] = f_d
                all_floors.add(f_d)
            elif parts[0] == 'above':
                f1, f2 = parts[1], parts[2]
                floor_below[f1] = f2
                all_floors.add(f1)
                all_floors.add(f2)
            elif parts[0] == 'lift-at':
                 all_floors.add(parts[1])

        # Get initial origin facts from task.initial_state
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'origin':
                  p, f_o = parts[1], parts[2]
                  if p not in self.passenger_info:
                       self.passenger_info[p] = {}
                  self.passenger_info[p]['origin'] = f_o
                  all_floors.add(f_o)

        # Precompute floor levels
        self.floor_levels = {}

        if not all_floors:
             # No floors defined
             return

        # Find the bottom floor: a floor that is a value in floor_below but not a key.
        bottom_floor = None
        floors_that_are_below_others = set(floor_below.values())
        floors_that_are_above_others = set(floor_below.keys())

        potential_bottom_floors = floors_that_are_below_others - floors_that_are_above_others

        if len(potential_bottom_floors) == 1:
             bottom_floor = potential_bottom_floors.pop()
        elif len(all_floors) == 1:
             # Single floor case
             bottom_floor = list(all_floors)[0]
        else:
             # Handle cases with multiple chains or malformed 'above' relations
             # For standard miconic, this case indicates an issue.
             # Fallback: Assign level 1 to all floors.
             for floor in all_floors:
                  self.floor_levels[floor] = 1
             # print("Warning: Could not determine unique bottom floor. Travel cost will be 0.")
             bottom_floor = None # Indicate that floor levels weren't built properly


        if bottom_floor is not None:
            # Assign levels starting from the bottom floor
            current_floor = bottom_floor
            level = 1
            # Build a map from floor to the floor immediately above it
            floor_above = {v: k for k, v in floor_below.items()}

            while current_floor is not None:
                self.floor_levels[current_floor] = level
                # Move up to the floor above the current one
                current_floor = floor_above.get(current_floor)
                level += 1


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Find current lift floor
        f_current = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                f_current = parts[1]
                break

        # If lift location is unknown, heuristic is infinite (or a large value)
        if f_current is None or f_current not in self.floor_levels:
             # This indicates an invalid state representation or floor not in static facts
             return float('inf')

        current_level = self.floor_levels[f_current]

        # 2. Identify unserved passengers and required floors
        num_at_origin = 0
        num_boarded_unserved = 0
        pickup_floors = set()
        dropoff_floors = set()

        # Build sets of passengers in different states for quick lookup
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        boarded_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'boarded'}
        origin_passengers_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'origin'}


        for p, info in self.passenger_info.items():
            # Check if passenger is served
            if p in served_passengers:
                continue # This passenger is done

            # Passenger is unserved
            f_origin = info.get('origin')
            f_destin = info.get('destin')

            if f_origin is None or f_destin is None:
                 # Missing origin/destin info for an unserved passenger
                 # This shouldn't happen if passenger_info is populated correctly from initial state
                 continue

            if p in boarded_passengers:
                # Passenger is unserved and boarded, needs dropoff
                num_boarded_unserved += 1
                dropoff_floors.add(f_destin)
            elif p in origin_passengers_in_state: # Passenger is unserved and at origin
                # Passenger needs pickup and dropoff
                num_at_origin += 1
                pickup_floors.add(f_origin)
                dropoff_floors.add(f_destin)
            # else: Passenger is unserved but neither boarded nor at origin? Invalid state?
            # Assuming valid states where unserved passengers are either at origin or boarded.


        # If all passengers are served, heuristic is 0
        if num_at_origin == 0 and num_boarded_unserved == 0:
            return 0

        # 3. Calculate travel cost
        required_floors = pickup_floors | dropoff_floors

        travel_cost = 0
        if required_floors: # Only calculate travel if there are floors to visit
            # Ensure all required floors are in our floor_levels map
            # This check is important if the fallback level assignment was used
            if not all(f in self.floor_levels for f in required_floors):
                 # This indicates a required floor was not found when building levels
                 # Problem definition issue? Return infinity?
                 return float('inf')

            min_level = min(self.floor_levels[f] for f in required_floors)
            max_level = max(self.floor_levels[f] for f in required_floors)

            # Estimated travel is the span from current floor to the min/max required floor
            travel_cost = max(current_level, max_level) - min(current_level, min_level)

        # 4. Total heuristic
        # Base action cost: 2 actions (board + depart) for each passenger at origin,
        #                   1 action (depart) for each boarded passenger.
        actions_cost = (num_at_origin * 2) + num_boarded_unserved

        return actions_cost + travel_cost

