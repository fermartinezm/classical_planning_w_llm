from fnmatch import fnmatch

# Assuming heuristics.heuristic_base import Heuristic is available
# If not, a minimal base class might be needed depending on the planner's structure.
# For this response, we assume the Heuristic base class is provided externally
# as implied by the problem description and example code structure.


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
    # Basic check: number of parts must match number of args unless wildcards are used extensively
    # A more robust check is done by zip and fnmatch
    if len(parts) != len(args) and '*' not in args: # Simple check for obvious mismatches
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the estimated lift movement cost to visit all necessary floors (origins for unboarded,
    destinations for boarded) and the number of board and depart actions needed.

    # Assumptions
    - Floors are ordered linearly by the 'above' predicate, defining a single chain.
    - The lift can carry multiple passengers.
    - The estimated lift movement is the minimum travel distance to cover the range of floors
      that need visiting, starting from the current floor.

    # Heuristic Initialization
    - Extracts the floor ordering from 'above' predicates and creates a mapping
      from floor names to numerical levels. Handles potential issues in 'above' facts
      by falling back to alphabetical order if necessary.
    - Stores the destination floor for each passenger by scanning initial state facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the numerical level for each floor based on the 'above' facts.
       This is done once during initialization by finding the lowest floor in the chain
       defined by `(above floor_above floor_below)` facts and assigning increasing levels upwards.
       If the chain is malformed or incomplete, floor levels are assigned alphabetically as a fallback.
    2. During initialization, scan the initial state facts to find the destination floor
       for each passenger using `(destin ?p ?f)` facts.
    3. For a given state (in the `__call__` method), identify the current floor of the lift.
    4. Identify all passengers who are not yet 'served' by checking the goal conditions
       and the current state facts.
    5. Separate unserved passengers into those waiting at their origin (unboarded, identified by `(origin ?p ?f)`)
       and those who are boarded (identified by `(boarded ?p)`).
    6. Identify the set of origin floors for unboarded passengers (`F_pickup`).
    7. Identify the set of destination floors for boarded passengers (`F_dropoff`).
    8. Combine these sets to get the set of all floors the lift must visit (`F_visit = F_pickup U F_dropoff`).
    9. If `F_visit` is empty, it means all unserved passengers are either already at their destination
       (which shouldn't happen if they are unserved) or are in a state not covered (e.g., dropped off
       at the wrong floor, which is not possible in this domain). If `F_visit` is empty and the goal
       is not reached, this might indicate an unreachable state or a problem definition issue.
       However, if `F_visit` is empty, it implies no further pickups or specific dropoffs are required
       at distinct floors relative to the current state of unserved passengers. In a solvable state
       where the goal is not reached, `F_visit` should typically not be empty unless all remaining
       unserved passengers are boarded and the lift is already at their destination floor (in which case
       only depart actions are needed, and movement_cost would be 0). If `F_visit` is empty, the movement
       cost is 0.
    10. If `F_visit` is not empty, calculate the minimum and maximum floor levels among floors in `F_visit`.
    11. Estimate the lift movement cost: This is the distance from the current lift floor
        to the closest extreme floor level in `F_visit` (either min or max), plus the distance
        between the min and max floor levels in `F_visit`.
        Cost = `min(|current_level - min_visit_level|, |current_level - max_visit_level|) + (max_visit_level - min_visit_level)`.
    12. Count the number of unboarded unserved passengers (each needs a 'board' action, cost 1).
    13. Count the number of boarded unserved passengers (each needs a 'depart' action, cost 1).
    14. The total heuristic value is the sum of the estimated lift movement cost,
        the number of board actions, and the number of depart actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Determine floor levels from 'above' facts.
        # Build a map where key is the lower floor and value is the floor directly above it.
        # (above floor_above floor_below) means floor_above is directly above floor_below.
        floor_below_to_above_map = {}
        all_floors_set = set()

        # Collect floors and build the direct above map from static facts
        for fact in self.static_facts:
            if match(fact, "above", "*", "*"):
                _, floor_above, floor_below = get_parts(fact)
                floor_below_to_above_map[floor_below] = floor_above
                all_floors_set.add(floor_above)
                all_floors_set.add(floor_below)

        # Also collect floors from initial state facts (lift-at, origin, destin)
        for fact in task.initial_state:
             parts = get_parts(fact)
             predicate = parts[0]
             if predicate in ["lift-at", "origin", "destin"]:
                 if len(parts) > 2: # Ensure there's a floor argument
                     all_floors_set.add(parts[-1])

        # Find the lowest floor: it's a floor that is in all_floors_set but is not a value
        # in the floor_below_to_above_map (i.e., no floor is directly below it in the chain).
        floors_that_are_above_something = set(floor_below_to_above_map.values())
        potential_lowest_floors = all_floors_set - floors_that_are_above_something

        lowest_floor = None
        if len(potential_lowest_floors) == 1:
            lowest_floor = list(potential_lowest_floors)[0]
        elif len(all_floors_set) == 1:
             # Single floor case, no 'above' facts
             lowest_floor = list(all_floors_set)[0]
        else:
             # Problem with 'above' facts or multiple chains. Fallback to alphabetical.
             # print(f"Warning: Could not determine unique lowest floor from 'above' facts chain ({potential_lowest_floors}). Falling back to alphabetical levels.")
             sorted_floors = sorted(list(all_floors_set))
             self.floor_levels = {floor: i + 1 for i, floor in enumerate(sorted_floors)}
             # Proceed with destination mapping and return
             self._map_passenger_destinations(task.initial_state, self.static_facts)
             return # Exit __init__ after fallback

        # Assign levels starting from the lowest floor
        self.floor_levels = {}
        current_floor = lowest_floor
        level = 1
        while current_floor is not None:
            if current_floor in self.floor_levels: # Cycle detection
                 # print(f"Warning: Detected cycle in floor 'above' relationships involving {current_floor}. Falling back to alphabetical levels.")
                 sorted_floors = sorted(list(all_floors_set))
                 self.floor_levels = {floor: i + 1 for i, floor in enumerate(sorted_floors)}
                 break # Exit while loop

            self.floor_levels[current_floor] = level
            # Find the floor directly above the current_floor
            current_floor = floor_below_to_above_map.get(current_floor)
            level += 1

        # If the chain didn't cover all floors found, something is wrong.
        if len(self.floor_levels) != len(all_floors_set):
             # print(f"Warning: Floor chain mapping covered {len(self.floor_levels)} floors, but {len(all_floors_set)} floors were found. Falling back to alphabetical levels.")
             sorted_floors = sorted(list(all_floors_set))
             self.floor_levels = {floor: i + 1 for i, floor in enumerate(sorted_floors)}


        # 2. Store destination floor for each passenger from the initial state.
        self._map_passenger_destinations(task.initial_state, self.static_facts)

    def _map_passenger_destinations(self, initial_state, static_facts):
        """Helper to map passengers to destinations."""
        self.passenger_destinations = {}
        # Destinations are typically in the initial state
        for fact in initial_state:
             if match(fact, "destin", "*", "*"):
                 _, passenger, destination_floor = get_parts(fact)
                 self.passenger_destinations[passenger] = destination_floor
        # Also check static facts just in case
        for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 _, passenger, destination_floor = get_parts(fact)
                 self.passenger_destinations[passenger] = destination_floor


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break
        # If lift location is not found, something is wrong with the state representation
        if current_lift_floor is None or current_lift_floor not in self.floor_levels:
             # This shouldn't happen in a valid state if floor_levels is correctly built
             # print(f"Warning: Lift location {current_lift_floor} not found or level unknown.")
             return float('inf') # Should not occur in valid states

        current_lift_level = self.floor_levels[current_lift_floor]

        # Identify unserved passengers and their state (boarded or at origin)
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}

        unserved_passengers = set(self.passenger_destinations.keys()) - served_passengers

        floors_to_visit = set()
        num_board_actions = 0
        num_depart_actions = 0

        for passenger in unserved_passengers:
            dest_floor = self.passenger_destinations.get(passenger)
            if dest_floor is None:
                 # Should not happen if passenger_destinations is complete
                 continue # Skip this passenger if destination is unknown

            if passenger in boarded_passengers:
                # Passenger is boarded, needs to be dropped off at destination
                floors_to_visit.add(dest_floor)
                num_depart_actions += 1
            elif passenger in origin_locations:
                # Passenger is waiting at origin, needs pickup and dropoff
                origin_floor = origin_locations[passenger]
                floors_to_visit.add(origin_floor)
                floors_to_visit.add(dest_floor)
                num_board_actions += 1
                num_depart_actions += 1
            # else: Passenger is neither served, boarded, nor at an origin? (Should not happen in valid states)


        # Calculate estimated lift movement cost
        movement_cost = 0
        if floors_to_visit:
            # Filter out any floors not found in floor_levels (shouldn't happen with robust init)
            valid_visit_levels = [self.floor_levels[f] for f in floors_to_visit if f in self.floor_levels]

            if valid_visit_levels: # Ensure there are valid levels to visit
                min_visit_level = min(valid_visit_levels)
                max_visit_level = max(valid_visit_levels)

                # Cost to reach the range + cost to traverse the range
                cost_to_reach_min = abs(current_lift_level - min_visit_level)
                cost_to_reach_max = abs(current_lift_level - max_visit_level)
                range_span = max_visit_level - min_visit_level

                movement_cost = min(cost_to_reach_min, cost_to_reach_max) + range_span
            # else: Should not happen if floors_to_visit is not empty and floor_levels is correct


        # Total heuristic is movement cost + board actions + depart actions
        total_heuristic_cost = movement_cost + num_board_actions + num_depart_actions

        return total_heuristic_cost
