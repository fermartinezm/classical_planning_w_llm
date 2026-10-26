import math

class miconicHeuristic:
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
        Estimates the number of actions required to reach the goal state.
        The estimate is the sum of three components:
        1. Lift movement cost: The minimum number of floor movements needed
           to traverse the range of floors containing the current lift position
           and all floors where passengers need to be picked up or dropped off.
        2. Boarding cost: The number of passengers waiting at their origin floors.
        3. Departing cost: The number of passengers currently boarded in the lift.

    Assumptions:
        - The domain defines a linear order of floors using the `(above f_lower f_higher)` predicate
          in the static facts, where `f_lower` is immediately below `f_higher`.
        - All floors mentioned in the problem instance (initial state, goals, static)
          are part of this linear structure.
        - Passengers needing service are either waiting at their origin or boarded.
        - The heuristic is designed for Greedy Best-First Search and does not need to be admissible.

    Heuristic Initialization:
        The constructor processes the static facts of the planning task:
        - It identifies all floors and builds a mapping (`floor_above`) representing
          the immediate 'above' relationship between floors.
        - It finds the lowest floor in the hierarchy.
        - It traverses the floor hierarchy from the lowest to the highest floor,
          creating a mapping (`floor_to_level`) from floor names to numerical levels (0-indexed).
        - It stores the destination floor for each passenger from the `(destin p d)` facts.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Check if the state is a goal state (all passengers served). If yes, the heuristic is 0.
        2. Identify the current floor of the lift by finding the `(lift-at f)` fact in the state.
        3. Identify passengers waiting at their origin floors by finding `(origin p o)` facts.
        4. Identify passengers currently boarded in the lift by finding `(boarded p)` facts.
        5. Determine the set of 'required' floors: this includes the origin floors of all waiting passengers
           and the destination floors of all boarded passengers.
        6. Calculate the lift movement cost:
           - If there are no required floors, the movement cost is 0.
           - Otherwise, find the numerical levels for the current lift floor and all required floors
             using the `floor_to_level` map.
           - The movement cost is the difference between the maximum and minimum levels among
             this combined set of floors (current lift floor + required floors). This represents
             the minimum span the lift must cover.
        7. Calculate the boarding cost: This is simply the number of waiting passengers.
        8. Calculate the departing cost: This is simply the number of boarded passengers.
        9. The total heuristic value is the sum of the movement cost, boarding cost, and departing cost.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static domain information.

        Args:
            task: The planning task object (instance of Task class).
        """
        self.task = task
        self.floor_to_level = {}
        self.destin_map = {}
        self.all_passengers = set() # Passengers mentioned in destin facts

        floor_above = {} # Map from lower floor to higher floor
        all_floors_in_above = set() # Floors mentioned in above facts

        # Process static facts to build floor structure and passenger destinations
        for fact_str in task.static:
            parts = fact_str[1:-1].split()
            predicate = parts[0]
            if predicate == 'above':
                f_lower = parts[1]
                f_higher = parts[2]
                all_floors_in_above.add(f_lower)
                all_floors_in_above.add(f_higher)
                floor_above[f_lower] = f_higher
            elif predicate == 'destin':
                p = parts[1]
                d = parts[2]
                self.all_passengers.add(p)
                self.destin_map[p] = d
            # Note: origin facts are typically in the initial state, not static.
            # We rely on destin facts in static to get the set of all passengers.

        # Build floor order and level mapping
        ordered_floors = []
        lowest_floor = None

        if all_floors_in_above:
            # Find the lowest floor: it's in all_floors_in_above but not a value in floor_above
            floors_that_are_below_others = set(floor_above.values())
            potential_lowest = all_floors_in_above - floors_that_are_below_others

            if len(potential_lowest) == 1:
                 lowest_floor = list(potential_lowest)[0]
            elif len(all_floors_in_above) == 1: # Case with single floor mentioned in above
                 lowest_floor = list(all_floors_in_above)[0]
            # else: Multiple lowest floors or no floors in above facts - indicates invalid structure

            # Traverse upwards from lowest floor
            current_floor = lowest_floor
            level = 0
            while current_floor is not None:
                ordered_floors.append(current_floor)
                self.floor_to_level[current_floor] = level
                level += 1
                current_floor = floor_above.get(current_floor)

        # Add any floors from destinations or initial state that weren't part of the 'above' chain
        # This handles single-floor problems or destinations/origins on floors not explicitly in 'above'
        all_floors_mentioned = set(all_floors_in_above) | set(self.destin_map.values())
        if hasattr(task, 'initial_state'):
             for fact_str in task.initial_state:
                 parts = fact_str[1:-1].split()
                 if len(parts) > 1 and parts[0] == 'lift-at':
                      all_floors_mentioned.add(parts[1]) # lift-at ?f
                 if len(parts) > 2 and parts[0] == 'origin':
                      all_floors_mentioned.add(parts[2]) # origin ?p ?f

        # If floor_to_level is still empty but there are floors mentioned,
        # it means no 'above' facts defined a structure. Assign arbitrary levels.
        if not self.floor_to_level and all_floors_mentioned:
             # Sort floors alphabetically for deterministic level assignment
             for i, floor in enumerate(sorted(list(all_floors_mentioned))):
                  self.floor_to_level[floor] = i
        # Note: If floors exist but are not connected by 'above' and not handled above,
        # they won't be in floor_to_level and might cause KeyError later.
        # Assuming valid miconic problems have a single connected floor structure defined by 'above'.


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            An estimate of the remaining actions to reach the goal.
        """
        # Check if goal is reached (optional, but good practice for a heuristic)
        # The search algorithm will also check this, but returning 0 is correct.
        if self.task.goal_reached(state):
            return 0

        current_lift_floor = None
        waiting_passengers = [] # List of (passenger, origin_floor)
        boarded_passengers = [] # List of passenger

        # Extract state information
        for fact_str in state:
            parts = fact_str[1:-1].split()
            predicate = parts[0]
            if predicate == 'lift-at':
                current_lift_floor = parts[1]
            elif predicate == 'origin':
                p = parts[1]
                o = parts[2]
                waiting_passengers.append((p, o))
            elif predicate == 'boarded':
                p = parts[1]
                boarded_passengers.append(p)
            # served passengers are implicitly handled by not being in waiting/boarded lists

        # Determine required floors
        pickup_floors = {o for (p, o) in waiting_passengers}
        # Only consider boarded passengers whose destination is known from static info
        dropoff_floors = {self.destin_map[p] for p in boarded_passengers if p in self.destin_map}
        required_floors = pickup_floors | dropoff_floors

        # Calculate movement cost
        movement_cost = 0
        # If there are floors to visit or the lift is at a floor, calculate span
        # We need the current floor's level and the levels of all required floors
        floors_to_consider_for_span = set(required_floors)
        if current_lift_floor:
             floors_to_consider_for_span.add(current_lift_floor)

        # Get levels for all relevant floors, ensuring they are in our map
        # If a floor is not in the map, it suggests an issue with the PDDL or parsing.
        # For valid miconic, all floors in state should be in self.floor_to_level.
        mapped_floors_to_consider = {f for f in floors_to_consider_for_span if f in self.floor_to_level}

        if mapped_floors_to_consider:
            relevant_levels = {self.floor_to_level[f] for f in mapped_floors_to_consider}
            min_level_span = min(relevant_levels)
            max_level_span = max(relevant_levels)
            movement_cost = max_level_span - min_level_span
        # else: No relevant floors could be mapped. This shouldn't happen in valid miconic.


        # Calculate board cost
        board_cost = len(waiting_passengers)

        # Calculate depart cost
        depart_cost = len(boarded_passengers)

        # Total heuristic estimate
        h = movement_cost + board_cost + depart_cost

        return h
