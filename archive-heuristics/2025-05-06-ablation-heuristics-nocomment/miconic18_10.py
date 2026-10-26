from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to parse a PDDL fact string into a list of parts."""
    # Remove outer parentheses and split by space
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions required to reach the goal state
    by considering the lift's current position, the location of unserved
    passengers (waiting or boarded), and the need to visit relevant floors
    for pickups and dropoffs.

    This heuristic is non-admissible and designed for greedy best-first search
    to minimize expanded nodes.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.

        Heuristic Initialization:
        - Parses the 'above' predicates to determine the linear ordering of floors
          and creates a mapping from floor name to its index.
        - Parses the 'destin' predicates to store the destination floor for each
          passenger.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Determine floor ordering and create floor_to_index map
        floor_above = {}
        all_floors = set()
        for fact in static_facts:
            if fact.startswith('(above '):
                f1, f2 = get_parts(fact)[1:]
                floor_above[f1] = f2
                all_floors.add(f1)
                all_floors.add(f2)

        # Find the lowest floor (a floor that is not above any other floor)
        # Correction: Find the lowest floor (a floor that no other floor is above)
        # i.e., a floor f such that there is no (above ?other f) fact.
        # In our floor_above map (f1 -> f2 means f2 is above f1), this means
        # finding f such that f is not a value in floor_above.
        lowest_floor = None
        floors_that_are_above_another = set(floor_above.keys())
        floors_that_are_below_another = set(floor_above.values())

        # The lowest floor is one that is below another floor, but no floor is below it.
        # i.e., it's a value in floor_above, but not a key in floor_above.
        # No, the lowest floor is the one that is a key in floor_above but not a value.
        # Example: (above f1 f2), (above f2 f3). f1 is key, not value. f2 is key and value. f3 is value, not key.
        # Lowest floor is f1. Highest floor is f3.
        # Let's re-read the PDDL: (up f1 f2) precond (above f1 f2). f2 is higher than f1.
        # (above f1 f2) means f1 is immediately below f2.
        # So, f1 is the lowest floor if no (above ?f f1) exists.
        # Let's build floor_below map: floor_below[f2] = f1 if (above f1 f2)
        floor_below = {}
        all_floors = set()
        for fact in static_facts:
            if fact.startswith('(above '):
                f1, f2 = get_parts(fact)[1:]
                floor_below[f2] = f1
                all_floors.add(f1)
                all_floors.add(f2)

        if not all_floors: # Handle case with no floors or no above facts
             self.floor_to_index = {}
             # Assume a single floor if any exists
             for fact in task.initial_state:
                 if fact.startswith('(lift-at '):
                     f = get_parts(fact)[1]
                     self.floor_to_index[f] = 0
                     break
             # Add any other floors mentioned in initial state if not already added
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 if len(parts) > 2 and parts[0] in ['origin', 'destin']:
                     f = parts[2]
                     if f not in self.floor_to_index:
                         self.floor_to_index[f] = 0 # Assign 0 if no ordering
             # Add floors from goals if not already added
             for goal in self.goals:
                 parts = get_parts(goal)
                 if len(parts) > 2 and parts[0] == 'served': # served doesn't mention floor
                      pass # Skip served goals for floor detection
                 elif len(parts) > 2 and parts[0] == 'at': # Assuming 'at' might appear in goals for miconic (though not in example)
                      f = parts[2]
                      if f not in self.floor_to_index:
                          self.floor_to_index[f] = 0 # Assign 0 if no ordering

        else:
            lowest_floor = None
            floors_with_something_below = set(floor_below.values())
            for f in all_floors:
                if f not in floor_below: # This floor is not above any other floor
                     # Correction: This floor has nothing immediately below it. It's the lowest.
                     lowest_floor = f
                     break

            self.floor_to_index = {}
            current_floor = lowest_floor
            index = 0
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                index += 1
                current_floor = floor_above.get(current_floor) # Use floor_above to go up

        # 2. Store passenger destinations
        self.passenger_to_destin = {}
        for fact in static_facts:
            if fact.startswith('(destin '):
                p, f = get_parts(fact)[1:]
                self.passenger_to_destin[p] = f

        # Store all passenger names for easy iteration
        self.all_passengers = set(self.passenger_to_destin.keys())


    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current floor of the lift.
        2. Identify all passengers who have not yet been served (based on the goal).
        3. If all passengers are served, the heuristic is 0.
        4. Identify 'active floors':
           - Floors where unserved passengers are waiting to be picked up.
           - Destination floors for unserved passengers who are currently boarded.
        5. Count the number of unserved passengers waiting ('num_waiting').
        6. Count the number of unserved passengers who are boarded ('num_boarded').
        7. If there are no active floors:
           - This implies all unserved passengers are boarded and are currently
             at their destination floor (which must be the lift's current floor).
           - The remaining actions are just 'depart' for each boarded passenger.
           - The heuristic is the number of boarded unserved passengers ('num_boarded').
        8. If there are active floors:
           - Determine the minimum and maximum floor indices among the active floors.
           - Estimate the travel cost: This is the minimum distance required for the
             lift to travel from its current floor to cover the range of active floors.
             A simple estimate is the distance to the closer extreme active floor
             plus the distance between the minimum and maximum active floors.
             `travel = min(abs(current_idx - min_idx), abs(current_idx - max_idx)) + (max_idx - min_idx)`
           - Estimate the action cost: Each waiting passenger needs a 'board' action
             and a 'depart' action (2 actions). Each boarded passenger needs a 'depart'
             action (1 action).
             `actions = num_waiting * 2 + num_boarded`
           - The total heuristic is the sum of the estimated travel cost and the
             estimated action cost.

        Assumptions:
        - Floors are linearly ordered as defined by the 'above' predicates.
        - All actions have a cost of 1.
        - The heuristic assumes a simplified model where the lift might visit
          floors in a non-optimal order but must cover the range of necessary floors.
        - Passengers only need to be picked up once and dropped off once.
        """
        state = node.state

        # 1. Identify current lift floor
        f_current = None
        for fact in state:
            if fact.startswith('(lift-at '):
                f_current = get_parts(fact)[1]
                break

        if f_current is None:
             # This should not happen in a valid miconic state, but handle defensively
             # If lift location is unknown, maybe return infinity or a large number?
             # Or assume a default floor? Let's assume it's a solvable state and lift is somewhere.
             # For this heuristic, we need the lift location. If not found, something is wrong.
             # Returning a large value encourages search away from this state.
             # However, the problem description implies valid states will be passed.
             # Let's assume f_current is always found.
             pass # f_current must be found if state is valid

        current_idx = self.floor_to_index.get(f_current, 0) # Default to 0 if floor not in map (shouldn't happen)


        # 2. Identify unserved passengers
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if fact.startswith('(served ')}
        unserved = {p for p in self.all_passengers if p not in served_passengers_in_state}

        # 3. If all passengers are served, heuristic is 0
        if not unserved:
            return 0

        # 4. Identify active floors and count waiting/boarded unserved passengers
        active_floors = set()
        num_waiting = 0
        num_boarded = 0
        passenger_origin_map = {}

        # First pass to build origin map and count boarded
        for fact in state:
            if fact.startswith('(origin '):
                p, f = get_parts(fact)[1:]
                passenger_origin_map[p] = f
            elif fact.startswith('(boarded '):
                p = get_parts(fact)[1]
                if p in unserved:
                    num_boarded += 1

        # Second pass to populate active_floors based on unserved status
        for p in unserved:
            if p in passenger_origin_map: # Passenger is waiting
                active_floors.add(passenger_origin_map[p])
                num_waiting += 1
            elif '(boarded {})'.format(p) in state: # Passenger is boarded (and not waiting)
                 # Need to drop them off at their destination
                 active_floors.add(self.passenger_to_destin[p])
            # Note: num_boarded was already counted in the first pass

        # 7. Handle case with no active floors
        if not active_floors:
            # This case implies all unserved passengers are boarded AND are at their destination floor.
            # The only remaining actions are 'depart' for each boarded unserved passenger.
            # The number of such passengers is num_boarded.
            return num_boarded

        # 8. Calculate heuristic if there are active floors
        active_indices = sorted([self.floor_to_index[f] for f in active_floors if f in self.floor_to_index])
        if not active_indices: # Should not happen if active_floors is not empty and floor_to_index is correct
             return float('inf') # Or a large number indicating a potential issue

        min_idx = active_indices[0]
        max_idx = active_indices[-1]

        # Estimated travel cost
        # Distance to closer extreme + range distance
        travel = min(abs(current_idx - min_idx), abs(current_idx - max_idx)) + (max_idx - min_idx)

        # Estimated action cost
        # Each waiting needs board (1) + depart (1) = 2 actions
        # Each boarded needs depart (1) action
        actions = num_waiting * 2 + num_boarded

        # Total heuristic
        return travel + actions

