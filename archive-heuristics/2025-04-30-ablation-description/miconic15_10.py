from fnmatch import fnmatch
# from heuristics.heuristic_base import Heuristic # Uncomment this line in the actual environment

# Dummy Heuristic base class for standalone testing
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
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

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to serve all passengers by summing
    the estimated cost for each individual passenger not yet served.
    The estimated cost for a passenger depends on their current state:
    waiting at origin, boarded, or served. It considers the lift's current
    position and the passenger's origin and destination floors.

    # Assumptions
    - The cost of moving the lift between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The 'above' predicates define a linear order of floors, where (above f_a f_b) means f_a is immediately above f_b.
    - Passenger origin and destination floors are fixed and available in the initial state or static facts.
    - If a passenger is not 'served' and not 'boarded', they are assumed to be waiting at their initial origin floor.

    # Heuristic Initialization
    - Parse the 'above' predicates from static facts to determine the floor order
      and create a mapping from floor names to numerical indices (lowest floor gets index 0).
    - Parse the 'origin' and 'destin' facts from the initial state and static
      facts to store the initial origin and destination floor for each passenger
      who needs to be served (i.e., is in the goal). Note that initial origin
      might not be present in initial_state if the passenger starts 'boarded'.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Initialize the total heuristic cost to 0.
    3. For each passenger that needs to be served (i.e., is listed in the goal):
        a. Check if the passenger is already 'served' in the current state. If yes, the cost for this passenger is 0.
        b. If not served, check if the passenger is 'boarded'.
            i. If 'boarded', the estimated cost for this passenger is the distance
               from the current lift floor to the passenger's destination floor,
               plus 1 action for 'depart'.
            ii. If not 'boarded', the passenger is assumed to be waiting at their
                initial origin floor. The estimated cost for this passenger is the
                distance from the current lift floor to the passenger's initial
                origin floor, plus 1 action for 'board', plus the distance from the
                initial origin floor to the destination floor, plus 1 action for 'depart'.
                This step requires knowing the initial origin floor, which is stored
                during initialization.
        c. Add the estimated cost for this passenger to the total heuristic cost.
    4. Return the total heuristic cost.

    The distance between two floors is the absolute difference in their numerical
    indices derived from the 'above' predicates.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger info.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Determine floor order and create floor_to_number mapping
        above_facts = [fact for fact in static_facts if match(fact, "above", "*", "*")]

        # Build the 'immediately_above' map: f_below -> f_above
        immediately_above = {}
        all_floors_from_above = set()
        for fact in above_facts:
            parts = get_parts(fact)
            if len(parts) == 3: # Ensure fact has correct structure (predicate, arg1, arg2)
                _, f_above, f_below = parts
                immediately_above[f_below] = f_above
                all_floors_from_above.add(f_above)
                all_floors_from_above.add(f_below)

        # Collect all floors mentioned anywhere relevant (above, lift-at, origin, destin).
        all_floors_from_init_static = set(all_floors_from_above)
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if parts and parts[0] in ['lift-at', 'origin', 'destin']:
                 if len(parts) > 2: # origin, destin
                     all_floors_from_init_static.add(parts[2])
                 elif len(parts) == 2: # lift-at
                     all_floors_from_init_static.add(parts[1])


        # Find the lowest floor: a floor 'f' such that no fact (above ?x f) exists.
        # This means 'f' is not the second argument (f_below) in any 'above' fact.
        floors_that_are_below_others = set(immediately_above.keys()) # These floors have something immediately above them
        potential_lowest = all_floors_from_init_static - floors_that_are_below_others

        lowest_floor = None
        if potential_lowest:
             # If there are multiple such floors (shouldn't happen in linear miconic), pick one deterministically (e.g., alphabetically first)
             lowest_floor = sorted(list(potential_lowest))[0]
        elif all_floors_from_init_static:
             # Fallback for cases like a single floor with no 'above' facts, or a cycle (invalid).
             # If all floors are keys in immediately_above, it implies a cycle or disconnected components.
             # For a single floor with no above, potential_lowest will contain it.
             # If there's a cycle like (above f1 f2) (above f2 f1), potential_lowest will be empty.
             # In a valid linear miconic problem, potential_lowest should have exactly one element.
             # If it's empty but there are floors, pick one alphabetically as a fallback.
             lowest_floor = sorted(list(all_floors_from_init_static))[0]


        self.floor_to_number = {}
        if lowest_floor:
            # Build the ordered list starting from the lowest floor
            ordered_floors = [lowest_floor]
            current = lowest_floor
            while True:
                next_f = None
                # Find the floor immediately above the current floor
                for fact in above_facts:
                    _, f_above, f_below = get_parts(fact)
                    if f_below == current:
                        next_f = f_above
                        break
                if next_f and next_f not in ordered_floors:
                    ordered_floors.append(next_f)
                    current = next_f
                else:
                    break # Reached the highest floor or a floor already added

            # Assign numbers based on the ordered list
            for i, floor in enumerate(ordered_floors):
                 self.floor_to_number[floor] = i

        # Ensure all floors mentioned anywhere are in the map, assign numbers if missing
        # This handles floors mentioned only in init/static but not in 'above' (e.g., single floor, or disconnected).
        # Assign sequential numbers starting after the ordered sequence.
        max_num = max(self.floor_to_number.values()) if self.floor_to_number else -1
        for floor in all_floors_from_init_static:
             if floor not in self.floor_to_number:
                  max_num += 1
                  self.floor_to_number[floor] = max_num


        # 2. Store passenger initial origin and destination floors
        self.passenger_info = {}
        # Collect all passengers mentioned in goals
        goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Get origin and destin facts from the initial state and static facts
        # Destin facts are static. Origin facts are in initial_state if not boarded.
        passenger_origins_init = {get_parts(fact)[1]: get_parts(fact)[2] for fact in initial_state if match(fact, "origin", "*", "*")}
        passenger_destins_init = {get_parts(fact)[1]: get_parts(fact)[2] for fact in initial_state if match(fact, "destin", "*", "*")}
        passenger_destins_static = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "destin", "*", "*")}

        # Combine destin info (static overrides initial if present, though shouldn't conflict)
        passenger_destins = {**passenger_destins_init, **passenger_destins_static}

        for passenger in goal_passengers:
             initial_origin_floor = passenger_origins_init.get(passenger)
             destin_floor = passenger_destins.get(passenger)

             # We need destin for all goal passengers.
             # We need initial_origin for passengers who are *not* initially boarded.
             # If a passenger starts boarded, their origin fact is gone from initial_state,
             # so initial_origin_floor will be None. This is expected and handled by the heuristic logic.
             if destin_floor: # Must have a destination to be served
                 self.passenger_info[passenger] = (initial_origin_floor, destin_floor)
             # else: Passenger in goal but no destin found? Problematic, skip or handle error.
             # Assuming valid problems, all goal passengers have a destin.


    def _get_floor_number(self, floor_name):
        """Helper to get the numerical index for a floor name."""
        # All relevant floors should have been added to floor_to_number during init.
        # If a floor name is somehow missing, this indicates a problem setup issue.
        # Returning 0 is a fallback, but might be misleading.
        # A valid miconic problem should have all floors mentioned in init/static covered by the numbering.
        return self.floor_to_number.get(floor_name, 0)


    def _distance(self, floor1_name, floor2_name):
        """Calculate the distance (number of floors between) two floors."""
        # This method is only called with valid floor names by the heuristic logic.
        num1 = self._get_floor_number(floor1_name)
        num2 = self._get_floor_number(floor2_name)
        return abs(num1 - num2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find the current lift floor
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        # If lift_floor is None, the state is invalid unless no passengers need serving.
        passengers_to_serve = [p for p in self.passenger_info if f'(served {p})' not in state]
        if not passengers_to_serve:
            return 0 # Goal state

        if lift_floor is None:
             # Invalid state with passengers needing service. Discourage.
             return float('inf') # Or a large integer

        total_cost = 0

        # Iterate through each passenger we need to serve (those in self.passenger_info and not served)
        for passenger, (initial_origin_floor, destin_floor) in self.passenger_info.items():
            # Check if the passenger is already served
            if f'(served {passenger})' in state:
                continue # Already served, cost is 0 for this passenger

            # Check if the passenger is boarded
            if f'(boarded {passenger})' in state:
                # Passenger is in the lift, needs to go to destination and depart
                cost_for_passenger = self._distance(lift_floor, destin_floor) + 1 # +1 for depart
                total_cost += cost_for_passenger
            else:
                # Passenger is not served and not boarded. Assume they are at their initial origin.
                # We must have a valid initial_origin_floor for this case.
                # If initial_origin_floor is None, it means this passenger started boarded but is now
                # unboarded and unserved. This state implies they were dropped off somewhere
                # that wasn't their destination, which shouldn't happen in a valid plan.
                # This state is likely unreachable or invalid in a standard miconic problem.
                # Assign a high cost.
                if initial_origin_floor is None:
                     # This indicates a problematic state or problem definition.
                     # The heuristic cannot proceed correctly without the origin.
                     return float('inf')

                # Passenger is waiting at initial_origin_floor.
                cost_for_passenger = (
                    self._distance(lift_floor, initial_origin_floor) + 1 # +1 for board
                    + self._distance(initial_origin_floor, destin_floor) + 1 # +1 for depart
                )
                total_cost += cost_for_passenger

        return total_cost
