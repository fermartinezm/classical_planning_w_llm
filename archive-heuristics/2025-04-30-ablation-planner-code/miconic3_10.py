# from heuristics.heuristic_base import Heuristic # Assuming Heuristic base class is available

from fnmatch import fnmatch

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# class miconicHeuristic(Heuristic): # Inherit from Heuristic if available
class miconicHeuristic: # Assuming no explicit base class needed for the output
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining cost by summing the minimum actions
    required to serve each unserved passenger independently. For each unserved
    passenger, it calculates the cost to get the lift to their current location
    (origin or current lift floor if boarded), perform the necessary action
    (board or depart), move the lift to their destination, and perform the final
    action (depart). Movement cost is estimated by the absolute difference in
    floor indices.

    # Assumptions
    - Each action (board, depart, up, down) has a cost of 1.
    - The cost for each passenger can be calculated independently and summed up.
      This ignores the possibility of picking up/dropping off multiple passengers
      on a single trip or floor visit, leading to an overestimation (non-admissible).
    - The state representation includes facts for passengers that are
      currently at their origin (`origin`), inside the lift (`boarded`), or
      already served (`served`). The heuristic calculates cost only for passengers
      explicitly listed with `origin` or `boarded` status in the state facts.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static `destin` facts.
    - Determines the order of floors based on the static `above` facts and creates
      a mapping from floor name to an integer index (0-based) representing its
      position from the bottom floor upwards.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift from the `lift-at` fact.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through all facts in the current state.
    4. For each fact:
       - If the fact is `(served ?p)`, this passenger is done; add 0 cost for them.
       - If the fact is `(boarded ?p)`, this passenger is in the lift.
         - Get their destination floor `f_destin` using the pre-calculated map.
         - Calculate the floor distance from the current lift floor to `f_destin`: `abs(index(f_destin) - index(current_lift_floor))`.
         - Add this distance + 1 (for the `depart` action) to the total cost.
       - If the fact is `(origin ?p ?f_origin)`, this passenger is waiting at `f_origin`.
         - Get their destination floor `f_destin` using the pre-calculated map.
         - Calculate the floor distance from the current lift floor to `f_origin`: `abs(index(f_origin) - index(current_lift_floor))`.
         - Calculate the floor distance from `f_origin` to `f_destin`: `abs(index(f_destin) - index(f_origin))`.
         - Add the first distance (to origin) + 1 (for `board`) + the second distance (to destination) + 1 (for `depart`) to the total cost.
    5. The total accumulated cost is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.

        Args:
            task: The planning task object containing goals and static facts.
        """
        self.goals = task.goals  # Goal conditions (e.g., (served p1))
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each passenger.
        self.goal_locations = {}
        # The goal is a conjunction of (served ?p) facts
        # We need the destination for each passenger mentioned in the goal
        goal_passengers = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served" and len(parts) > 1:
                 goal_passengers.add(parts[1])

        # Find destinations for all passengers mentioned in the goal
        for passenger in goal_passengers:
             destin_fact = next((fact for fact in static_facts if match(fact, "destin", passenger, "*")), None)
             if destin_fact:
                 _, p, floor = get_parts(destin_fact)
                 self.goal_locations[p] = floor
             # else: This passenger is in the goal but has no destin fact? Problematic instance.


        # Build floor order and index mapping from 'above' facts.
        # (above f_above f_below) means f_above is physically above f_below.
        # We need to find the bottom floor and build the list upwards.
        above_map = {} # Maps floor_below -> floor_above
        all_floors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) > 2:
                floor_above, floor_below = parts[1], parts[2]
                above_map[floor_below] = floor_above
                all_floors.add(floor_above)
                all_floors.add(floor_below)

        self.floor_to_index = {}
        self.floor_list = []

        if all_floors:
            # Find the lowest floor: it's a floor that is in all_floors but is *not* a value in above_map.
            # This means no floor is immediately below it.
            floors_that_are_above_something = set(above_map.values()) # These are the first args in (above ?f_above ?f_below)

            potential_lowest = list(all_floors - floors_that_are_above_something)

            lowest_floor = None
            if len(potential_lowest) == 1:
                 lowest_floor = potential_lowest[0]
            elif len(all_floors) == 1: # Case with only one floor and no above facts
                 lowest_floor = list(all_floors)[0]
            else:
                 # If still ambiguous, this might indicate a non-linear structure or error.
                 # For this domain, linear is expected.
                 # As a last resort, sort alphabetically, but this is unreliable.
                 # print("Warning: Could not uniquely determine lowest floor. Sorting alphabetically.")
                 sorted_floors = sorted(list(all_floors))
                 if sorted_floors:
                      lowest_floor = sorted_floors[0]


            if lowest_floor is not None:
                current = lowest_floor
                index = 0
                # Traverse upwards using the above_map (floor_below -> floor_above)
                while current is not None:
                    self.floor_list.append(current)
                    self.floor_to_index[current] = index
                    index += 1
                    # Find the floor immediately above 'current'
                    current = above_map.get(current)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining actions to reach a goal state.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Find the current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) > 1:
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # Should not happen in a valid state, but handle defensively
             # print(f"Warning: Lift location '{current_lift_floor}' not found or unknown in state.")
             return float('inf') # Cannot proceed without valid lift location

        current_lift_index = self.floor_to_index[current_lift_floor]

        total_cost = 0

        # Iterate through state facts to find unserved passengers
        # We assume state facts list the status of all passengers relevant to the goal
        # that are not yet served (either origin or boarded), plus served passengers
        # and the lift location.
        # We only calculate cost for passengers found with 'origin' or 'boarded' status.
        passengers_processed = set() # Ensure we don't double count if a passenger appears multiple times (shouldn't happen in valid state)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "boarded" and len(parts) > 1:
                passenger = parts[1]
                if passenger in passengers_processed: continue
                passengers_processed.add(passenger)

                # Passenger is in the lift, needs to go to destination and depart
                destin_floor = self.goal_locations.get(passenger)

                if destin_floor is None:
                    # Passenger in state but no destination found? Problematic instance.
                    # print(f"Warning: Boarded passenger '{passenger}' has no destination.")
                    continue # Cannot calculate cost for this passenger

                destin_index = self.floor_to_index.get(destin_floor)
                if destin_index is None:
                    # print(f"Warning: Unknown destination floor '{destin_floor}' for passenger '{passenger}'.")
                    continue # Cannot calculate cost

                # Cost = moves to destination + depart action
                moves = abs(destin_index - current_lift_index)
                depart_action = 1
                total_cost += moves + depart_action

            elif predicate == "origin" and len(parts) > 2:
                passenger = parts[1]
                origin_floor = parts[2]
                if passenger in passengers_processed: continue
                passengers_processed.add(passenger)

                # Passenger is waiting at origin, needs pickup, transport, and dropoff
                destin_floor = self.goal_locations.get(passenger)

                if destin_floor is None:
                    # Passenger in state but no destination found? Problematic instance.
                    # print(f"Warning: Passenger '{passenger}' at origin '{origin_floor}' has no destination.")
                    continue # Cannot calculate cost

                origin_index = self.floor_to_index.get(origin_floor)
                destin_index = self.floor_to_index.get(destin_floor)

                if origin_index is None or destin_index is None:
                    # print(f"Warning: Unknown origin '{origin_floor}' or destination '{destin_floor}' for passenger '{passenger}'.")
                    continue # Cannot calculate cost

                # Cost = moves to origin + board action + moves to destination + depart action
                moves_to_origin = abs(origin_index - current_lift_index)
                board_action = 1
                moves_to_destin = abs(destin_index - origin_index)
                depart_action = 1

                total_cost += moves_to_origin + board_action + moves_to_destin + depart_action

            # If predicate is "served", cost for this passenger is 0, so we do nothing.
            # If predicate is "lift-at", we already processed it.
            # Other predicates are ignored.

        return total_cost
