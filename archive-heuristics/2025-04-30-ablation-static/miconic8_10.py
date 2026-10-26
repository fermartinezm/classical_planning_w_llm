from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists in the environment
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and strip potential whitespace
    fact_str = str(fact).strip()
    # Check if it looks like a PDDL fact (starts with '(' and ends with ')')
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # This case indicates an unexpected format in the state representation
         # For standard PDDL facts in the state, this check should pass.
         # Return empty list or handle error as appropriate for the planner's state representation.
         # Assuming valid PDDL fact strings in the state.
         return [] # Return empty list for non-fact strings

    # Remove parentheses and split by whitespace
    return fact_str[1:-1].split()

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
    # Check if each part matches the corresponding pattern argument (using fnmatch for wildcards)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort to serve all passengers.
    It counts the number of board/depart actions needed for unserved passengers
    and adds an estimate of the vertical travel distance required for the lift
    to visit all necessary pickup and dropoff floors.

    # Assumptions
    - All passengers need to be served (reach their destination).
    - The floor structure is a linear chain defined by `(above f_i f_j)` facts.
    - Passengers are either at their origin floor or boarded in any valid state.
    - The cost of each action (move, board, depart) is 1.

    # Heuristic Initialization
    - Extract the floor order and map floor names to integer levels based on `(above f_i f_j)` facts.
    - Extract the destination floor for each passenger from `(destin ?p ?d)` facts.
    - Identify all passengers in the problem from initial state or static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift. Get its corresponding level using the precomputed floor level map.
    2. Initialize the heuristic value `h` to 0.
    3. Create a set `required_floors` to store floors the lift must visit to pick up or drop off passengers.
    4. Iterate through all passengers identified during initialization:
       - Check if the passenger is already served (`(served p)` is present in the current state). If yes, this passenger contributes 0 to the heuristic, so continue to the next passenger.
       - If the passenger is not served, check their current status in the state:
         - If `(origin p o)` is present: The passenger is waiting at floor `o`. They need to be boarded and later departed. Add 2 to `h` (representing the board and depart actions). Add floor `o` to the `required_floors` set.
         - If `(boarded p)` is present: The passenger is inside the lift. They need to be departed at their destination. Add 1 to `h` (representing the depart action). Find the passenger's destination `d` (stored during initialization) and add floor `d` to the `required_floors` set.
         - (Note: In a valid miconic state, an unserved passenger must be either at their origin or boarded).
    5. After iterating through all passengers, if the `required_floors` set is empty, it implies that all passengers who needed pickup or dropoff have already been served. In this case, the heuristic value is 0 (as initialized and potentially incremented only for served passengers, which is then skipped). Return 0.
    6. If `required_floors` is not empty, the lift still needs to visit these floors. Estimate the minimum vertical travel:
       - Get the integer levels for all floors in `required_floors` using the precomputed map.
       - Find the minimum (`min_req_level`) and maximum (`max_req_level`) levels among these required floors.
       - Get the level of the current lift floor (`current_lift_level`).
       - The estimated number of move actions is the total vertical span covered by the current lift position and all required floors. This is calculated as `max(current_lift_level, max_req_level) - min(current_lift_level, min_req_level)`.
       - Add this estimated move count to `h`.
    7. Return the final calculated value of `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting floor levels, passenger destinations, and the set of all passengers."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Build floor levels mapping
        self.floor_levels = {}
        floor_below_map = {}
        all_floors = set()

        # Collect all 'above' relationships and all mentioned floors
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f_above, f_below = parts[1], parts[2]
                floor_below_map[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the bottom floor (a floor that is not the second argument of any 'above' fact)
        bottom_floor = None
        floors_that_are_below_others = set(floor_below_map.values())
        for floor in all_floors:
            if floor not in floors_that_are_below_others:
                bottom_floor = floor
                break

        # Build the level map starting from the bottom floor
        if bottom_floor and all_floors:
            current_floor = bottom_floor
            level = 1
            while current_floor in all_floors:
                self.floor_levels[current_floor] = level
                next_floor = None
                # Find the floor immediately above current_floor
                # This is the floor f_above such that (above f_above current_floor) exists
                for f_above, f_below in floor_below_map.items():
                    if f_below == current_floor:
                        next_floor = f_above
                        break # Assuming a linear chain, there's only one floor immediately above

                if next_floor is None:
                    # Reached the top floor (no floor is immediately above this one)
                    break

                current_floor = next_floor
                level += 1
        # If floor_levels is empty, it implies no floors or malformed 'above' facts.
        # The heuristic will need to handle this gracefully in __call__.


        # 2. Extract passenger destinations and identify all passengers
        self.goal_locations = {}
        self.all_passengers = set()

        # Passengers and destinations are defined in static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and len(parts) >= 2: # Check length before accessing indices
                if parts[0] == "destin" and len(parts) == 3:
                    p, d = parts[1], parts[2]
                    self.goal_locations[p] = d
                    self.all_passengers.add(p)
                elif parts[0] == "origin" and len(parts) == 3:
                    p, o = parts[1], parts[2]
                    self.all_passengers.add(p)

        # Ensure all passengers mentioned in goals are included (redundant for miconic but robust)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served" and len(parts) == 2:
                  p = parts[1]
                  self.all_passengers.add(p)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # The current state is a frozenset of fact strings

        # Check if the goal is reached
        if self.goals <= state:
            return 0

        h = 0
        required_floors = set()
        current_lift_floor = None

        # Find the current lift location
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        # If floor levels weren't built or lift floor is not in map, state is problematic
        if not self.floor_levels or current_lift_floor is None or current_lift_floor not in self.floor_levels:
             # This indicates a problem with the domain definition or state representation
             # Returning infinity signals this state is likely not on a path to the goal
             return float('inf') # Should not happen in valid miconic states

        current_lift_level = self.floor_levels[current_lift_floor]

        # Pre-process state facts for quick lookups
        state_facts_dict = {}
        for fact in state:
             parts = get_parts(fact)
             if parts:
                  predicate = parts[0]
                  args = tuple(parts[1:])
                  if predicate not in state_facts_dict:
                       state_facts_dict[predicate] = set()
                  state_facts_dict[predicate].add(args)

        def is_fact_in_state(predicate, *args):
             return args in state_facts_dict.get(predicate, set())

        # Process each passenger to find unserved ones and required stops
        for passenger in self.all_passengers:
            # Check if the passenger is already served
            if is_fact_in_state("served", passenger):
                continue # This passenger is done

            # Passenger is not served
            # Check if the passenger is waiting at their origin floor
            found_origin = False
            for args in state_facts_dict.get("origin", set()):
                 if len(args) == 2 and args[0] == passenger:
                      origin_floor = args[1]
                      h += 2 # Estimate 1 board + 1 depart action
                      required_floors.add(origin_floor)
                      found_origin = True
                      break # Assuming a passenger has only one origin fact

            if found_origin:
                 continue # Move to the next passenger

            # If not at origin and not served, check if the passenger is inside the lift
            if is_fact_in_state("boarded", passenger):
                h += 1 # Estimate 1 depart action
                dest_floor = self.goal_locations.get(passenger)
                if dest_floor: # Passenger must have a destination
                    required_floors.add(dest_floor)
                # else: Invalid state? Boarded passenger without destination?

            # Note: A valid unserved passenger must be either at origin or boarded.
            # If neither, it might indicate an invalid state or a passenger not part of the problem goal.
            # We only consider passengers identified in init/goals/static.

        # If no required floors, all relevant passengers are served
        if not required_floors:
            return h # Should be 0 if goal is reached, >0 otherwise (which shouldn't happen if required_floors is empty)

        # Calculate estimated moves based on the span of required floors and current lift position
        required_levels = [self.floor_levels[f] for f in required_floors if f in self.floor_levels]

        # This check is a safeguard, should not be needed if required_floors is not empty
        # and floor_levels contains all floors from the problem.
        if not required_levels:
             # This could happen if a required floor wasn't in the floor_levels map,
             # which implies an issue with __init__ or the PDDL.
             return float('inf') # Indicate problematic state

        min_req_level = min(required_levels)
        max_req_level = max(required_levels)

        # Estimated moves is the vertical distance covering the current lift floor and all required floors
        estimated_moves = max(current_lift_level, max_req_level) - min(current_lift_level, min_req_level)

        h += estimated_moves

        return h
