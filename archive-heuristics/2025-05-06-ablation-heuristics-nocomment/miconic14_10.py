from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to parse a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        This heuristic estimates the remaining cost by summing three components:
        1. The number of passengers waiting at their origin floor (need board + depart).
        2. The number of passengers currently boarded in the lift (need depart).
        3. An estimate of the minimum lift movement required to visit all floors
           where passengers need to be picked up or dropped off. This movement
           estimate is the vertical distance between the lowest and highest
           floors that the lift must visit, including the current floor.

    Assumptions:
        - Floors are linearly ordered (e.g., f1, f2, f3, ...). The `above`
          predicate defines this order such that if (above f_lower f_higher) is true,
          f_higher is immediately above f_lower.
        - Passenger origins and destinations are static.
        - The goal is to serve all passengers.

    Heuristic Initialization:
        - Parses the `(above ?f1 ?f2)` facts from the static information to
          determine the linear order of floors and assign an integer level
          to each floor name. It assumes `(above f_lower f_higher)` means
          `f_higher` is immediately above `f_lower`. It finds the lowest floor
          (one not appearing as the first argument (`f_lower`) in any `above` fact,
          meaning no floor is immediately below it) and assigns level 1,
          incrementing levels for floors immediately above. This mapping
          is stored in `self.floor_levels`.
        - Parses the `(destin ?p ?f)` facts from the static information to
          store the destination floor for each passenger in `self.destinations`.
        - Collects all passenger names from the destinations.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state from the node.
        2. Check if the goal state is reached (all passengers served). If yes, return 0.
        3. Find the current floor of the lift by looking for the `(lift-at ?f)` fact in the state.
        4. Initialize counters for waiting and boarded passengers to zero.
        5. Initialize a set `stops_needed` to store the floors the lift must visit.
        6. Create dictionaries/sets to track the current status and location of each passenger from the state facts (`origin`, `boarded`, `served`).
        7. Iterate through all known passengers (`self.all_passengers`):
           - If the passenger is not marked as `served` in the current state:
             - Check if the passenger is `waiting` (has an `(origin ...)` fact). If yes:
               - Increment waiting passenger count.
               - Add their origin floor to `stops_needed`.
             - Check if the passenger is `boarded` (has a `(boarded ...)` fact). If yes:
               - Increment boarded passenger count.
               - Find the destination `f_destin` for passenger `?p` using the precomputed `self.destinations`.
               - Add `f_destin` to `stops_needed`.
        8. If there are no waiting or boarded passengers (meaning all unserved passengers are in an unexpected state, or there are no unserved passengers left to process after the goal check), return 0. This case should ideally not be reached for solvable problems before the goal check returns 0.
        9. Calculate the movement cost:
           - Get the level of the current lift floor using `self.floor_levels`.
           - Get the levels for all floors in the `stops_needed` set using `self.floor_levels`.
           - Combine the current lift level and the levels of `stops_needed` floors into a single set of relevant levels.
           - If `relevant_levels` is not empty, the movement cost estimate is the difference between the maximum and minimum levels in this set. Otherwise, it's 0.
        10. The total heuristic value is the sum of the waiting passenger count, the boarded passenger count, and the movement cost estimate.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse floor levels from (above f_lower f_higher) facts
        self.floor_levels = {}
        lower_to_higher_map = {} # f_lower -> f_higher
        all_floors = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                # Interpretation: (above f_lower f_higher) means f_higher is immediately above f_lower
                _, f_lower, f_higher = get_parts(fact) # f_lower is first arg, f_higher is second arg
                lower_to_higher_map[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)

        lowest_floor = None
        # A floor is the lowest if no other floor is immediately below it,
        # i.e., it does not appear as a key in the lower_to_higher_map.
        floors_that_are_lower_than_others = set(lower_to_higher_map.keys())
        for floor in all_floors:
            if floor not in floors_that_are_lower_than_others:
                lowest_floor = floor
                break # Assuming a single lowest floor in a linear structure

        if lowest_floor:
            # Assign levels starting from the lowest floor
            current_floor = lowest_floor
            level = 1
            # Traverse upwards using the lower_to_higher_map
            while current_floor in lower_to_higher_map:
                self.floor_levels[current_floor] = level
                current_floor = lower_to_higher_map[current_floor]
                level += 1
            # Assign level to the highest floor (the last one found)
            self.floor_levels[current_floor] = level
        # else: floor_levels remains empty if no floors or non-linear structure

        # 2. Store passenger destinations from (destin p f) facts
        self.destinations = {}
        self.all_passengers = set()
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.destinations[passenger] = destination_floor
                self.all_passengers.add(passenger)

    def __call__(self, node):
        state = node.state

        # Check if goal is reached (all passengers served)
        # This is the most efficient check for h=0
        if self.goals <= state:
             return 0

        current_lift_floor = None
        waiting_passengers_count = 0
        boarded_passengers_count = 0
        stops_needed = set() # Floors the lift must visit

        # Track passenger status from current state
        passenger_is_served = set()
        passenger_is_boarded = set()
        passenger_origin = {} # passenger -> origin_floor

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "lift-at":
                current_lift_floor = parts[1]
            elif predicate == "origin":
                passenger = parts[1]
                origin_floor = parts[2]
                passenger_origin[passenger] = origin_floor
            elif predicate == "boarded":
                passenger_is_boarded.add(parts[1])
            elif predicate == "served":
                passenger_is_served.add(parts[1])

        # Identify unserved passengers and required stops
        unserved_passengers = self.all_passengers - passenger_is_served

        for passenger in unserved_passengers:
            if passenger in passenger_origin: # Passenger is waiting
                waiting_passengers_count += 1
                origin_floor = passenger_origin[passenger]
                if origin_floor in self.floor_levels: # Ensure floor is known
                    stops_needed.add(origin_floor)
            elif passenger in passenger_is_boarded: # Passenger is boarded
                boarded_passengers_count += 1
                destination_floor = self.destinations.get(passenger)
                if destination_floor and destination_floor in self.floor_levels: # Ensure destination floor is known
                    stops_needed.add(destination_floor)
            # else: Passenger is unserved but not waiting or boarded. This shouldn't happen in valid states.
            # If it does, they are not counted towards waiting/boarded, and their floor isn't added to stops_needed.

        # If no passengers need pickup or dropoff, but goal is not reached,
        # this implies an issue with the state or problem definition.
        # However, if unserved_passengers is not empty, stops_needed should be non-empty
        # assuming all origin/destin floors are in floor_levels.
        # If stops_needed is empty here, it likely means all unserved passengers
        # are at floors not parsed in floor_levels, which points to a problem definition issue.
        # Let's return a minimal heuristic if stops_needed is unexpectedly empty
        # while there are still unserved passengers.
        if not stops_needed and unserved_passengers:
             # Fallback: Count unserved passengers. Each needs at least 1 action (depart if boarded, board+depart if waiting)
             # This is a weak fallback, ideally stops_needed is populated correctly.
             return len(unserved_passengers)


        # Calculate movement cost
        movement_cost = 0
        relevant_levels = set()

        # Add levels of required stops
        if stops_needed:
             # Filter out stops whose levels weren't parsed (problematic floors)
             known_stops_needed = {f for f in stops_needed if f in self.floor_levels}
             if known_stops_needed:
                 relevant_levels.update({self.floor_levels[f] for f in known_stops_needed})

        # Add level of current lift floor
        if current_lift_floor and current_lift_floor in self.floor_levels:
             relevant_levels.add(self.floor_levels[current_lift_floor])
        # else: current_lift_floor is None or unknown, indicates state issue.
        # If relevant_levels is empty here, it means no known stops and no known lift location.
        # This is highly unlikely in a valid state unless all floors are unknown.

        if relevant_levels:
            min_level = min(relevant_levels)
            max_level = max(relevant_levels)
            movement_cost = max_level - min_level
        # else: movement_cost remains 0 if relevant_levels is empty (e.g., no floors parsed)


        # Total heuristic value
        # Sum of waiting passengers (need board+depart) + boarded passengers (need depart) + movement range
        heuristic_value = waiting_passengers_count + boarded_passengers_count + movement_cost

        return heuristic_value
