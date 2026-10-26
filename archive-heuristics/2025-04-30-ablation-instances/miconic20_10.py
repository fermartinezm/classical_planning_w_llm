from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the number of board and depart actions needed for unserved passengers
    and adds an estimate of the minimum vertical movement required for the lift
    to visit all necessary floors (origin floors for waiting passengers and
    destination floors for boarded passengers).

    # Assumptions
    - The lift has sufficient capacity for all passengers.
    - The floor structure is a linear sequence defined by 'above' predicates.
    - The cost of each action (board, depart, up, down) is 1.

    # Heuristic Initialization
    - Parses 'above' facts to determine the linear order of floors and create
      a mapping from floor name to its level (0-indexed from the lowest floor).
    - Parses 'destin' facts to create a mapping from passenger name to their
      destination floor.
    - Collects all passenger names mentioned in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unserved Passengers: Find all passengers for whom the predicate
       '(served ?p)' is not true in the current state. The set of all passengers
       is determined during initialization from the goal state and supplemented
       by passengers found in 'origin' or 'boarded' facts in the current state.

    2. Count Required Actions:
       - For each unserved passenger currently waiting at an origin floor
         '(origin ?p ?f)', count 2 actions (1 board, 1 depart).
       - For each unserved passenger currently boarded '(boarded ?p)', count 1
         action (1 depart).
       - Sum these counts to get the base action cost.

    3. Identify Current Lift Location: Find the floor '?f' where '(lift-at ?f)'
       is true. Determine its level using the precomputed floor level mapping.

    4. Identify Required Stop Floors:
       - Collect all origin floors '?f_origin' for passengers currently waiting
         '(origin ?p ?f_origin)'. These are pickup stops.
       - Collect all destination floors '?f_destin' for passengers currently
         boarded '(boarded ?p)'. These are dropoff stops.
       - Combine these sets to get the set of all floors the lift must visit.

    5. Estimate Movement Cost:
       - If there are no required stop floors, the movement cost is 0.
       - Otherwise, find the minimum and maximum floor levels among the required
         stop floors.
       - The estimated movement cost is the vertical distance from the current
         lift floor level to the nearest required stop floor level, plus the
         total vertical span between the minimum and maximum required stop
         floor levels. This estimates the cost to reach the "action zone" and
         traverse its full height.

    6. Calculate Total Heuristic: Sum the base action cost (from step 2) and
       the estimated movement cost (from step 5).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Build floor level mapping from 'above' facts
        # (above f_higher f_lower) means f_higher is immediately above f_lower
        # We want to build the list from lowest to highest floor.
        # The lowest floor is the one that is NOT the first argument (f_higher)
        # in any 'above' fact.
        floors_that_are_higher = set()
        all_floors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                f_higher, f_lower = parts[1], parts[2]
                floors_that_are_higher.add(f_higher)
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        lowest_floor = None
        for floor in all_floors:
             if floor not in floors_that_are_higher:
                 lowest_floor = floor
                 break

        # Handle potential edge case (e.g., single floor, or malformed problem)
        if lowest_floor is None and all_floors:
             # If no floor is found that isn't a 'f_higher', assume it's a single floor
             # or pick the first one found.
             if len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
             else:
                 # This indicates a problem with the 'above' facts structure
                 # For robustness, pick an arbitrary floor, though this might lead to bad heuristics
                 # print("Warning: Could not determine unique lowest floor from 'above' facts.")
                 lowest_floor = list(all_floors)[0] # Arbitrary pick


        # Build ordered list of floors from lowest to highest
        ordered_floors = []
        current_floor = lowest_floor
        # Map f_lower -> f_higher to easily find the floor above the current one
        lower_to_higher = {}
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "above":
                  lower_to_higher[parts[2]] = parts[1]

        while current_floor is not None:
            ordered_floors.append(current_floor)
            # Find the floor immediately above current_floor using the map
            current_floor = lower_to_higher.get(current_floor)


        # Create floor level mapping
        self.floor_levels = {floor: level for level, floor in enumerate(ordered_floors)}

        # 2. Build passenger destination mapping from 'destin' facts
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor

        # 3. Collect all passenger names mentioned in the goal state
        self.all_passengers = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served":
                  self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Unserved Passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Collect all passengers currently in the state (waiting or boarded)
        passengers_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if parts[0] in ["origin", "boarded"]:
                  passengers_in_state.add(parts[1])

        # Unserved passengers are those in the initial/goal set or currently in state, but not served
        all_relevant_passengers = self.all_passengers | passengers_in_state
        unserved_passengers = {p for p in all_relevant_passengers if p not in served_passengers}


        # If no unserved passengers, it's a goal state
        if not unserved_passengers:
            return 0

        # 2. Count Required Actions and Identify Pickup/Dropoff Stops
        action_cost = 0
        pickup_stops = set()
        dropoff_stops = set()

        for passenger in unserved_passengers:
            is_waiting = False
            is_boarded = False
            origin_floor = None

            # Check if passenger is waiting or boarded in the current state
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "origin" and parts[1] == passenger:
                    is_waiting = True
                    origin_floor = parts[2]
                    break # Found origin
                if parts[0] == "boarded" and parts[1] == passenger:
                    is_boarded = True
                    break # Found boarded

            if is_waiting:
                action_cost += 2 # Needs board and depart
                if origin_floor:
                    pickup_stops.add(origin_floor)
            elif is_boarded:
                action_cost += 1 # Needs depart
                dest_floor = self.destinations.get(passenger)
                if dest_floor:
                     dropoff_stops.add(dest_floor)
                # else: print(f"Warning: Destination not found for boarded passenger {passenger}")
            # else: # Unserved but neither waiting nor boarded - should not happen in valid states


        # 3. Identify Current Lift Location
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        if lift_floor is None:
             # Should not happen in a valid miconic state
             # print("Error: Lift location not found in state.")
             return float('inf') # Indicate unsolvable/invalid state


        # Combine required floors
        required_floors = pickup_stops | dropoff_stops

        # 5. Estimate Movement Cost
        movement_cost = 0
        if required_floors:
            # Filter out any required floors that weren't found in static facts (shouldn't happen)
            required_levels = [self.floor_levels[f] for f in required_floors if f in self.floor_levels]

            if required_levels: # Ensure list is not empty after filtering
                min_level_needed = min(required_levels)
                max_level_needed = max(required_levels)

                current_lift_level = self.floor_levels.get(lift_floor)
                if current_lift_level is None:
                     # Should not happen if lift_floor was found and floor_levels is correct
                     # print(f"Error: Unknown lift floor {lift_floor} level.")
                     return float('inf') # Indicate unsolvable/invalid state


                # Distance from current lift level to the nearest required floor level
                dist_to_min = abs(current_lift_level - min_level_needed)
                dist_to_max = abs(current_lift_level - max_level_needed)
                dist_to_nearest = min(dist_to_min, dist_to_max)

                # Vertical span of required floors
                vertical_span = max_level_needed - min_level_needed

                movement_cost = dist_to_nearest + vertical_span
            # else: print("Warning: Required floors have no known levels.")


        # 6. Calculate Total Heuristic
        total_cost = action_cost + movement_cost

        return total_cost
