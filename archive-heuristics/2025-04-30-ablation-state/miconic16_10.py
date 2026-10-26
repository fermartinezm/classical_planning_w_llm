from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the fact has at least as many parts as args for a valid match
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all passengers
    to their destinations. It sums the required board/depart actions for each unserved
    passenger and adds an estimate of the minimum vertical movement required to visit
    all necessary floors (origins for waiting passengers, destinations for boarded passengers).

    # Assumptions
    - Floors are arranged in a single linear sequence defined by 'above' predicates.
    - Passengers need to be boarded at their origin and departed at their destination.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding or departing a passenger is 1.
    - The movement cost is estimated by the distance from the current lift floor to the
      nearest required floor extreme plus the total vertical span of all required floors.

    # Heuristic Initialization
    - Parses static facts to determine the linear order of floors and assign a level
      (integer) to each floor, starting from level 0 for the bottom floor.
    - Stores the destination floor for each passenger.
    - Identifies all passengers in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state satisfies the goal conditions. If yes, the heuristic is 0.
    2. Identify the current floor of the lift by finding the fact `(lift-at ?f)`. If no such
       fact exists, the state is likely invalid or unsolvable, return infinity.
    3. Get the level (integer representation) of the current lift floor using the
       precomputed floor-to-level map. If the floor is not in the map, return infinity.
    4. Identify all unserved passengers. An unserved passenger is one who does not have
       the predicate `(served ?p)` true.
    5. For each unserved passenger, determine their current state: either waiting at
       their origin floor (`(origin ?p ?f)`) or boarded in the lift (`(boarded ?p)`).
    6. Calculate the non-movement cost. This is the sum of minimum board/depart actions
       needed for each unserved passenger. A passenger waiting at their origin needs
       both a `board` action and a `depart` action (cost 2). A passenger already boarded
       only needs a `depart` action (cost 1). Sum these costs for all unserved passengers.
    7. Identify the set of "required floors". These are the origin floors for all
       unserved passengers waiting at their origin, and the destination floors for all
       unserved passengers who are currently boarded.
    8. If the set of required floors is empty, the movement cost is 0.
    9. If the set of required floors is not empty:
       - Get the levels for all required floors using the floor-to-level map. If any
         required floor is not in the map, return infinity (invalid state).
       - Find the minimum and maximum floor levels among the required floors.
       - Estimate the movement cost. This is calculated as the minimum of the vertical
         distance from the current lift floor's level to the minimum required level,
         and the vertical distance from the current lift floor's level to the maximum
         required level, plus the total vertical span between the minimum and maximum
         required levels. This estimates the minimum travel needed to sweep through
         the range of required floors starting from the current position.
    10. The total heuristic value is the sum of the non-movement cost (step 6) and the
        estimated movement cost (step 8 or 9).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting floor levels and passenger destinations."""
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Build floor level map
        all_floors = set()
        floor_immediately_above_map = {} # Map: floor_below -> floor_above
        floors_that_are_below_something_values = set() # Floors that appear as the value in (above x y)

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above' and len(parts) == 3:
                f_above, f_below = parts[1], parts[2]
                all_floors.add(f_above)
                all_floors.add(f_below)
                floor_immediately_above_map[f_below] = f_above
                floors_that_are_below_something_values.add(f_above) # f_above is a value in the map
            # Also collect floors from origin/destin facts to ensure all floors are known
            elif parts[0] in ['origin', 'destin'] and len(parts) == 3:
                 all_floors.add(parts[2])

        # Find the bottom floor: the floor that is not a value in the floor_immediately_above_map
        # (i.e., no floor is immediately below it in the defined chain)
        bottom_floor = None
        candidate_bottom_floors = all_floors - floors_that_are_below_something_values

        if len(candidate_bottom_floors) == 1:
             bottom_floor = list(candidate_bottom_floors)[0]
        elif len(candidate_bottom_floors) > 1:
             # This might indicate multiple floor chains or isolated floors.
             # For a single linear building, there should be exactly one bottom floor.
             # Handle as potentially malformed or complex structure not covered by simple heuristic.
             # print(f"Warning: Found multiple potential bottom floors: {candidate_bottom_floors}. Floor order determination may be inaccurate.")
             # In case of multiple candidates, the level mapping might be ambiguous.
             # We proceed with one, but the map might be incomplete or incorrect.
             bottom_floor = list(candidate_bottom_floors)[0]
        # else: # len == 0, no bottom floor found. Handled by checking bottom_floor is None

        self.floor_to_level = {}
        if bottom_floor is not None:
            # Assign levels starting from the bottom
            current_floor = bottom_floor
            level = 0
            visited_floors = set() # For cycle detection
            while current_floor is not None and current_floor in all_floors:
                if current_floor in visited_floors: # Cycle detection
                    # print(f"Warning: Cycle detected in floor order involving {current_floor}. Floor order determination stopped.")
                    self.floor_to_level = {} # Invalidate map on cycle
                    break
                visited_floors.add(current_floor)

                self.floor_to_level[current_floor] = level
                next_floor = floor_immediately_above_map.get(current_floor)

                # Check if next_floor is actually a floor we know about
                if next_floor is not None and next_floor not in all_floors:
                     # print(f"Warning: Floor {next_floor} found in 'above' relation but not in known floors.")
                     self.floor_to_level = {} # Invalidate map
                     break # Stop if relation points to unknown floor

                current_floor = next_floor
                level += 1

            # Check if all floors found were assigned a level (only relevant if no cycle/unknown floor)
            # If the map was invalidated due to cycle/unknown floor, this check is skipped.
            if self.floor_to_level and len(self.floor_to_level) != len(all_floors):
                 # print(f"Warning: Only {len(self.floor_to_level)} out of {len(all_floors)} floors were assigned a level. Floor order might be incomplete.")
                 # Decide whether to use partial map or invalidate. Let's use partial for now.
                 pass # Keep the partial map

        # 2. Store passenger destinations and identify all passengers
        self.passenger_destinations = {}
        all_passengers_set = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin' and len(parts) == 3:
                passenger, destination = parts[1], parts[2]
                self.passenger_destinations[passenger] = destination
                all_passengers_set.add(passenger)
            elif parts[0] == 'origin' and len(parts) == 3: # Also collect passengers from origin facts
                 passenger = parts[1]
                 all_passengers_set.add(passenger)

        # Identify all passengers in the problem (those mentioned in origin or destin)
        self.all_passengers = frozenset(all_passengers_set)


    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # Handle case where floor levels could not be determined during init
        if not self.floor_to_level:
             # Fallback heuristic: Count unserved passengers
             unserved_count = 0
             # Check against all passengers identified during init
             for p in self.all_passengers:
                 # Check if the served fact exists in the current state
                 if f"(served {p})" not in state:
                     unserved_count += 1
             # Return a value > 0 if unserved passengers exist, 0 otherwise (handled by goal check)
             # Use a simple multiplier.
             return unserved_count * 2 # Arbitrary multiplier > 0

        # 2. Find current lift floor and its level
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown, heuristic is infinite (unsolvable from this state)
        if current_lift_floor is None:
             # print("Error: Lift location (lift-at) not found in state.")
             return float('inf')

        current_lift_level = self.floor_to_level.get(current_lift_floor)
        # If lift is at a floor not in our level map, state is likely invalid
        if current_lift_level is None:
             # print(f"Error: Lift is at floor {current_lift_floor} which has no assigned level.")
             return float('inf')


        # 3-5. Identify unserved passengers and their state, calculate non-movement cost
        unserved_passengers_at_origin = {} # {passenger: origin_floor}
        unserved_boarded_passengers = set() # {passenger}
        served_passengers = set() # Keep track to filter

        # First pass to find served passengers
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served' and len(parts) > 1:
                 served_passengers.add(parts[1])

        # Second pass to find unserved passengers at origin or boarded
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'origin' and len(parts) == 3:
                p, f = parts[1], parts[2]
                # Only consider passengers identified during init and not yet served
                if p in self.all_passengers and p not in served_passengers:
                    unserved_passengers_at_origin[p] = f
            elif parts[0] == 'boarded' and len(parts) == 2:
                p = parts[1]
                # Only consider passengers identified during init and not yet served
                if p in self.all_passengers and p not in served_passengers:
                    unserved_boarded_passengers.add(p)

        # Calculate non-movement actions: 2 per passenger at origin (board, depart), 1 per boarded (depart)
        non_movement_cost = 2 * len(unserved_passengers_at_origin) + 1 * len(unserved_boarded_passengers)

        # 6-9. Calculate movement actions
        required_floors = set()
        for p, origin_floor in unserved_passengers_at_origin.items():
            # Need to visit origin to pick up
            required_floors.add(origin_floor)
            # Need to visit destination to drop off
            destination_floor = self.passenger_destinations.get(p)
            if destination_floor: # Should always exist for passengers in self.all_passengers
                 required_floors.add(destination_floor)
            else:
                 # Passenger in state has no destination? Invalid problem definition.
                 # print(f"Error: Passenger {p} at origin has no destination defined.")
                 return float('inf')


        for p in unserved_boarded_passengers:
            # Need to visit destination to drop off
            destination_floor = self.passenger_destinations.get(p)
            if destination_floor: # Should always exist for passengers in self.all_passengers
                 required_floors.add(destination_floor)
            else:
                 # Boarded passenger has no destination? Invalid problem definition.
                 # print(f"Error: Boarded passenger {p} has no destination defined.")
                 return float('inf')

        movement_cost = 0
        if required_floors:
            required_levels = []
            for f in required_floors:
                 level = self.floor_to_level.get(f)
                 if level is not None:
                      required_levels.append(level)
                 else:
                      # Required floor has no assigned level? Invalid state or init failure.
                      # print(f"Error: Required floor {f} has no assigned level.")
                      return float('inf')

            if required_levels: # Should be true if required_floors was not empty and all had levels
                min_level_R = min(required_levels)
                max_level_R = max(required_levels)

                # Estimate movement cost: distance to nearest required floor extreme + span
                movement_cost = min(abs(current_lift_level - min_level_R), abs(current_lift_level - max_level_R)) + (max_level_R - min_level_R)
            # else: required_floors was not empty, but required_levels became empty? This shouldn't happen
            # if the check for level is done inside the loop and returns inf.

        # 10. Total heuristic
        total_cost = non_movement_cost + movement_cost

        return total_cost
