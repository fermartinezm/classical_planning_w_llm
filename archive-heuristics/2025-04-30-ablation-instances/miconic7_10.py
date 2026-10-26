from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to transport all
    unserved passengers to their destinations. It sums the estimated non-move
    actions (board and depart) and the estimated minimum move actions for the lift.

    # Assumptions
    - Each unserved passenger needs one final 'depart' action at their destination.
    - Each unserved passenger currently at their origin needs one 'board' action.
    - The minimum number of move actions required is the difference between the
      highest and lowest floor numbers that need to be visited (either for pickup
      or dropoff), considering the lift's current floor. This assumes the lift
      can visit all necessary floors in one continuous movement range.
    - Floor numbers are derived from the static 'above' predicates, establishing
      a total order of floors.

    # Heuristic Initialization
    - Extract the floor ordering from the static 'above' facts to create a mapping
      from floor objects (e.g., 'f1', 'f2') to numerical floor levels.
    - Extract the destination floor for each passenger from the static 'destin' facts.
    - Identify all passengers present in the problem instance.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all passengers that have not yet been served. A passenger is
       unserved if the fact '(served <passenger>)' is not present in the state.
    2. If there are no unserved passengers, the state is a goal state, and the
       heuristic value is 0.
    3. Calculate the estimated non-move actions:
       - Count the number of unserved passengers currently at their origin floor.
         Each of these passengers needs a 'board' action.
       - Count the total number of unserved passengers. Each of these needs a
         'depart' action eventually.
       - The total non-move cost is the sum of these two counts.
    4. Calculate the estimated move actions:
       - Determine the lift's current floor.
       - Identify the set of floors that must be visited:
         - For each unserved passenger at their origin, their origin floor must be visited (for pickup).
         - For each unserved passenger who is currently boarded, their destination floor must be visited (for dropoff).
       - Map these required floors and the lift's current floor to their numerical floor levels using the pre-calculated mapping.
       - The estimated move cost is the difference between the maximum and minimum
         numerical floor levels among the lift's current floor and all required stop floors.
    5. The total heuristic value is the sum of the estimated non-move actions and
       the estimated move actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor numbering based on 'above' predicates.
        - Passenger destinations.
        - List of all passengers.
        """
        self.goals = task.goals  # Goal conditions (used to identify all passengers)
        self.static_facts = task.static # Static facts (used for floor order and destinations)
        self.initial_state = task.initial_state # Initial state (used to identify all passengers)

        # 1. Build floor numbering
        self.floor_to_number = {}
        self.number_to_floor = {}
        above_facts_parts = [get_parts(fact) for fact in self.static_facts if match(fact, "above", "*", "*")]

        # Find all floor objects involved in 'above' relations
        all_floors = set()
        for parts in above_facts_parts:
             if len(parts) == 3: # (above f_higher f_lower)
                 all_floors.add(parts[1]) # f_higher
                 all_floors.add(parts[2]) # f_lower

        if not all_floors:
             # This case indicates an issue with the domain/instance if floors exist but no 'above' facts.
             # In a valid miconic domain, there should be 'above' facts defining floor order.
             print("Warning: No floors found or no 'above' facts in static. Cannot build floor map.")
             # The heuristic will return a fallback value if floor map is empty.
             return

        # Find the lowest floor (a floor f for which no (above ?any f) exists)
        # In (above f_higher f_lower), f_lower is below f_higher.
        # The lowest floor is one that is never the second argument (f_lower) in any (above ?any f) fact.
        floors_that_are_lower = {parts[2] for parts in above_facts_parts if len(parts) == 3}
        lowest_floor = None
        for floor in all_floors:
            if floor not in floors_that_are_lower:
                lowest_floor = floor
                break

        if lowest_floor is None:
             # This indicates a problem like a cycle or disconnected floors in 'above' relations.
             print("Warning: Could not determine lowest floor from 'above' facts.")
             # The heuristic will return a fallback value if floor map is empty.
             return

        # Build map by following the 'above' chain upwards
        # (above f_higher f_lower) means f_higher is one level above f_lower
        # We need a map from lower floor -> higher floor
        above_map = {parts[2]: parts[1] for parts in above_facts_parts if len(parts) == 3}

        current_floor = lowest_floor
        current_number = 1
        while current_floor:
            self.floor_to_number[current_floor] = current_number
            self.number_to_floor[current_number] = current_floor
            # Find the floor directly above current_floor
            current_floor = above_map.get(current_floor)
            current_number += 1

        # 2. Extract passenger destinations
        self.passenger_to_dest = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "destin":
                passenger, destination = parts[1], parts[2]
                self.passenger_to_dest[passenger] = destination

        # 3. Identify all passengers
        self.all_passengers = set()
        # Look in initial state and static facts for objects appearing as passengers
        # Check arguments of predicates that involve passengers
        for fact in task.initial_state | task.static_facts:
             parts = get_parts(fact)
             if len(parts) > 1:
                 predicate = parts[0]
                 # Predicates with passenger as the first argument
                 if predicate in ["origin", "destin", "boarded", "served"]:
                     if len(parts) > 1: self.all_passengers.add(parts[1])
                 # Add other predicates if they involve passengers in other positions
                 # (e.g., a predicate like (in-lift ?person ?lift)) - not applicable here

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unserved passengers
        unserved_passengers = {p for p in self.all_passengers if f'(served {p})' not in state}

        # 2. Goal check
        if not unserved_passengers:
            return 0  # All passengers are served

        # Ensure floor numbering was successful
        if not self.floor_to_number:
             # Fallback heuristic if floor map is not available
             # Return number of unserved passengers as a basic estimate
             return len(unserved_passengers)

        # 3. Calculate non-move actions
        h_board = 0
        pickup_floors = set()
        for p in unserved_passengers:
            # Check if passenger p is at their origin floor
            origin_floor = None
            for fact in state:
                 parts = get_parts(fact)
                 if len(parts) == 3 and parts[0] == "origin" and parts[1] == p:
                      origin_floor = parts[2]
                      break # Found origin fact for this passenger

            if origin_floor:
                h_board += 1 # Needs a board action
                pickup_floors.add(origin_floor)

        h_depart = len(unserved_passengers) # Each unserved passenger needs a depart action eventually
        h_non_move = h_board + h_depart

        # 4. Calculate move actions
        dropoff_floors = set()
        for p in unserved_passengers:
            if f'(boarded {p})' in state:
                # Passenger is boarded, needs to be dropped off at destination
                dest_f = self.passenger_to_dest.get(p)
                if dest_f: # Ensure destination is known from static facts
                    dropoff_floors.add(dest_f)

        required_stops = pickup_floors.union(dropoff_floors)

        h_moves = 0
        if required_stops:
            # Find the lift's current floor
            current_f = None
            for fact in state:
                parts = get_parts(fact)
                if len(parts) == 2 and parts[0] == "lift-at":
                    current_f = parts[1]
                    break # Found lift location

            if current_f and current_f in self.floor_to_number:
                 # Get floor numbers for all required stops that are in our map
                 required_stop_numbers = {self.floor_to_number[f] for f in required_stops if f in self.floor_to_number}

                 # Add current floor number to the set of relevant floors for range calculation
                 all_relevant_numbers = required_stop_numbers.union({self.floor_to_number[current_f]})

                 if all_relevant_numbers: # Should not be empty if required_stops is not empty and current_f is mapped
                     min_num = min(all_relevant_numbers)
                     max_num = max(all_relevant_numbers)
                     h_moves = max_num - min_num
            # else: h_moves remains 0 if current_f is not found or not in map (shouldn't happen in valid state)

        # 5. Total heuristic
        total_cost = h_non_move + h_moves

        return total_cost
