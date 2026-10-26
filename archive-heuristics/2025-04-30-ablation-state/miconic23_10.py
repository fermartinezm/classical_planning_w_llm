from fnmatch import fnmatch
# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and strip leading/trailing whitespace
    fact = str(fact).strip()
    # Basic check if it looks like a fact string (starts with '(' and ends with ')')
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    else:
        # If it doesn't look like a fact, return as is or handle appropriately.
        # Based on context, facts are expected to be strings like '(predicate arg1 arg2)'.
        # Returning the split string might work for simple cases but is risky.
        # Assume valid fact string format for inputs to get_parts.
        return fact.split() # Fallback: just split the string


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the estimated cost for each unserved passenger independently.
    The estimated cost for a passenger includes:
    1. Movement cost for the lift to reach the passenger's origin floor (if waiting).
    2. Cost of the 'board' action (if waiting).
    3. Movement cost for the lift to travel from the origin floor to the destination floor (if waiting).
    4. Movement cost for the lift to travel from the current lift floor to the destination floor (if boarded).
    5. Cost of the 'depart' action (if boarded or waiting).

    Note that movement costs are calculated as the absolute difference in floor indices.
    This heuristic is likely non-admissible as it overestimates by summing independent costs,
    ignoring that the lift can transport multiple passengers and visit floors efficiently.
    However, it aims to guide a greedy best-first search effectively by capturing
    the remaining "work" for each passenger including necessary travel.

    # Assumptions
    - The PDDL defines floor order using `(above f_higher f_lower)` facts, forming a total order.
    - All passengers need to be served (reach their destination and depart) to satisfy the goal.
    - Action costs are uniform (implicitly 1).

    # Heuristic Initialization
    - Parses `(above f_higher f_lower)` facts from static information to create a mapping from floor names to numerical indices, representing their vertical order.
    - Parses `(origin p f)` facts from static information to map each passenger to their origin floor name.
    - Parses `(destin p f)` facts from static information to map each passenger to their destination floor name.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Find the current floor of the lift from the state fact `(lift-at ?f)`. Convert the floor name to its numerical index using the pre-calculated mapping.
    3. Iterate through all passengers identified during initialization.
    4. For each passenger `p`:
       a. Check if the fact `(served p)` is present in the current state. If yes, this passenger is served, add 0 to the total cost for this passenger, and continue to the next passenger.
       b. If the passenger is not served, check if the fact `(boarded p)` is present in the current state.
       c. If `(boarded p)` is true:
          - This passenger is currently in the lift.
          - Get the passenger's destination floor name and convert it to its index (`destin_idx`).
          - The estimated cost for this passenger is the movement cost from the current lift floor (`f_current_idx`) to the destination floor (`abs(f_current_idx - destin_idx)`) plus the cost of the 'depart' action (1).
          - Add `abs(f_current_idx - destin_idx) + 1` to the total heuristic cost.
       d. If `(boarded p)` is false:
          - This passenger is waiting at their origin floor.
          - Get the passenger's origin floor name and convert it to its index (`origin_idx`).
          - Get the passenger's destination floor name and convert it to its index (`destin_idx`).
          - The estimated cost for this passenger is the movement cost from the current lift floor (`f_current_idx`) to the origin floor (`abs(f_current_idx - origin_idx)`) plus the cost of the 'board' action (1), plus the movement cost from the origin floor to the destination floor (`abs(origin_idx - destin_idx)`) plus the cost of the 'depart' action (1).
          - Add `abs(f_current_idx - origin_idx) + abs(origin_idx - destin_idx) + 2` to the total heuristic cost.
    5. Return the total calculated heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger origins, and destinations.
        """
        super().__init__(task)

        # Map floor names to numerical indices based on the 'above' relation.
        self.floor_to_index = {}
        self.index_to_floor = {}
        above_pairs = set()
        all_floors = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'above' and len(parts) == 3:
                f_higher, f_lower = parts[1], parts[2]
                above_pairs.add((f_higher, f_lower))
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        if not all_floors:
             # Handle case with no floors (shouldn't happen in valid miconic)
             # If no floors, no passengers can exist or move. Heuristic will be 0.
             # Initialize empty maps.
             return

        # Find the lowest floor (a floor that is never the first element in an 'above' pair)
        higher_floors = {f_higher for f_higher, _ in above_pairs}
        lowest_floor = None
        for floor in all_floors:
            if floor not in higher_floors:
                lowest_floor = floor
                break

        if lowest_floor is None:
             # This might happen if the 'above' facts don't form a single chain or are cyclic.
             # For standard miconic, there's a clear lowest floor.
             # As a fallback, just sort floors alphabetically, though this is not PDDL-defined order.
             sorted_floors = sorted(list(all_floors))
             for i, floor in enumerate(sorted_floors):
                 self.floor_to_index[floor] = i
                 self.index_to_floor[i] = floor
        else:
            # Build the ordered list of floors starting from the lowest
            ordered_floors = [lowest_floor]
            current_floor = lowest_floor
            # Create a mapping from lower floor to higher floor for easy lookup
            lower_to_higher = {f_lower: f_higher for f_higher, f_lower in above_pairs}

            while len(ordered_floors) < len(all_floors):
                next_floor = lower_to_higher.get(current_floor)
                if next_floor and next_floor not in ordered_floors:
                    ordered_floors.append(next_floor)
                    current_floor = next_floor
                else:
                    # This might indicate a problem with the 'above' facts not forming a single chain
                    # or missing floors. For standard miconic, this loop should complete.
                    # If it doesn't, the floor list will be incomplete.
                    break # Exit the loop if we can't find the next floor

            # Create the index mapping
            for i, floor in enumerate(ordered_floors):
                self.floor_to_index[floor] = i
                self.index_to_floor[i] = floor


        # Map passengers to their origin and destination floor names
        self.passenger_origin = {}
        self.passenger_destin = {}

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'origin' and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.passenger_origin[p] = f
            elif parts and parts[0] == 'destin' and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.passenger_destin[p] = f


        # Collect all unique passengers who have an origin or destination defined
        self.all_passengers = set(self.passenger_origin.keys()) | set(self.passenger_destin.keys())


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # Find the current lift floor
        current_lift_floor_name = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at' and len(parts) == 2:
                current_lift_floor_name = parts[1]
                break

        if current_lift_floor_name is None:
             # This should not happen in a valid miconic state, but handle defensively
             return float('inf') # Should not reach here in a solvable problem

        # Get the index of the current lift floor
        f_current_idx = self.floor_to_index.get(current_lift_floor_name)
        if f_current_idx is None:
             # This should not happen if floor mapping was built correctly
             return float('inf') # Should not reach here

        total_cost = 0

        # Iterate through all passengers and calculate their contribution to the heuristic
        for passenger in self.all_passengers:
            # Check if the passenger is already served
            if f"(served {passenger})" in state:
                continue # Passenger is served, no more actions needed for them

            # Passenger is not served. Check if they are boarded or waiting.
            is_boarded = f"(boarded {passenger})" in state

            if is_boarded:
                # Passenger is boarded, needs to reach destination and depart
                destin_floor_name = self.passenger_destin.get(passenger)
                if destin_floor_name is None:
                     # Should not happen in a valid problem
                     return float('inf')

                destin_idx = self.floor_to_index.get(destin_floor_name)
                if destin_idx is None:
                     return float('inf')

                # Cost for a boarded passenger: move to destination + depart
                # Movement cost from current lift floor to destination floor
                movement_cost = abs(f_current_idx - destin_idx)
                depart_cost = 1
                total_cost += movement_cost + depart_cost

            else:
                # Passenger is waiting at origin, needs to board, travel, and depart
                origin_floor_name = self.passenger_origin.get(passenger)
                destin_floor_name = self.passenger_destin.get(passenger)

                if origin_floor_name is None or destin_floor_name is None:
                     # Should not happen in a valid problem
                     return float('inf')

                origin_idx = self.floor_to_index.get(origin_floor_name)
                destin_idx = self.floor_to_index.get(destin_floor_name)

                if origin_idx is None or destin_idx is None:
                     return float('inf')

                # Cost for a waiting passenger:
                # move from current lift floor to origin + board + move from origin to destination + depart
                movement_to_origin_cost = abs(f_current_idx - origin_idx)
                board_cost = 1
                movement_origin_to_destin_cost = abs(origin_idx - destin_idx)
                depart_cost = 1

                total_cost += movement_to_origin_cost + board_cost + movement_origin_to_destin_cost + depart_cost

        return total_cost
