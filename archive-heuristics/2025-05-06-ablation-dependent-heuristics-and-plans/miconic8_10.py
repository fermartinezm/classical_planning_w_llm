from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input defensively
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the required non-move actions (board and depart)
    for each unserved passenger and adds an estimate of the minimum move
    actions needed to visit all floors where actions are required.

    # Assumptions
    - Floors are ordered linearly and contiguously. The `above` predicates
      define this order.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The lift can carry multiple passengers.
    - Passengers only need to be picked up at their origin and dropped off
      at their destination.

    # Heuristic Initialization
    - Parses the `above` predicates from static facts to determine the
      linear order of floors and assign a numerical level to each floor.
    - Stores the origin and destination floor for each passenger by
      parsing `origin` and `destin` facts from the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift.
    2. Identify all passengers who have not yet been served (i.e., `(served ?p)`
       is not in the state).
    3. For each unserved passenger:
       - If the passenger is at their origin floor (`(origin ?p ?f)` is true),
         they need a `board` action at that floor and a `depart` action at
         their destination floor. Add 2 to the non-move action count.
       - If the passenger is boarded (`(boarded ?p)` is true), they need a
         `depart` action at their destination floor. Add 1 to the non-move
         action count.
    4. Identify the set of "required floors":
       - Include the origin floor for every unserved passenger currently at
         their origin.
       - Include the destination floor for every unserved passenger currently
         boarded.
    5. If the set of required floors is empty, the heuristic is 0 (goal state).
    6. If required floors exist:
       - Find the minimum and maximum floor levels among the required floors.
       - Calculate the minimum number of move actions required to visit all
         these floors starting from the current lift floor. This is estimated
         as the distance from the current floor to the closer of the two
         extreme required floors (min or max level), plus the distance
         between the min and max required floor levels.
         `move_cost = min(abs(current_level - min_req_level), abs(current_level - max_req_level)) + (max_req_level - min_req_level)`
    7. The total heuristic value is the sum of the total non-move actions
       and the estimated minimum move actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger info.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Map passenger to their origin and destination floors
        self.passenger_origin = {}
        self.passenger_destin = {}

        # Extract passenger origins and destinations from initial state and goals
        # Note: Origin is typically in initial state, Destin in goals.
        # We parse both initial state and static facts for origins, and goals for destinations.
        # Static facts might contain (origin p f) for some problems.
        for fact in task.initial_state | task.static:
             parts = get_parts(fact)
             if parts and parts[0] == "origin":
                 p, f = parts[1], parts[2]
                 self.passenger_origin[p] = f

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                p = parts[1]
                # Find the destination for this passenger from static facts
                # (destin p f) is usually a static fact.
                destin_fact = next((f for f in self.static_facts if match(f, "destin", p, "*")), None)
                if destin_fact:
                    self.passenger_destin[p] = get_parts(destin_fact)[2]
                # If not found in static, check initial state (less common for destin)
                elif task.initial_state:
                     destin_fact = next((f for f in task.initial_state if match(f, "destin", p, "*")), None)
                     if destin_fact:
                         self.passenger_destin[p] = get_parts(destin_fact)[2]


        # Determine floor order and assign levels
        self.floor_levels = {}
        floor_immediately_above_map = {}
        all_floors = set()

        # Collect all floor names and build the 'immediately above' map
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                floor_immediately_above_map[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)
            # Also collect floors from other predicates just in case
            elif parts and len(parts) > 1 and parts[-1].startswith('f'):
                 all_floors.add(parts[-1])
            elif parts and len(parts) > 2 and parts[0] in ['origin', 'destin'] and parts[2].startswith('f'):
                 all_floors.add(parts[2])
            elif parts and len(parts) > 1 and parts[0] == 'lift-at' and parts[1].startswith('f'):
                 all_floors.add(parts[1])


        # Find the bottom floor (a floor that is not a key in floor_immediately_above_map)
        bottom_floor = None
        # Convert set to list to iterate; set iteration order is not guaranteed
        all_floors_list = list(all_floors)
        for floor in all_floors_list:
            if floor not in floor_immediately_above_map:
                # Check if this floor is actually present in any 'above' fact as the lower floor
                # This handles cases with disconnected floors or single floors, though unlikely in miconic
                is_lower_floor_in_above = any(get_parts(f)[2] == floor for f in self.static_facts if get_parts(f) and get_parts(f)[0] == "above")
                if not is_lower_floor_in_above and len(all_floors) > 1:
                     # This floor is not the lower floor in any 'above' fact, and it's not the bottom floor
                     # This case might indicate an issue with the problem definition or a floor
                     # that is isolated. For simplicity, we assume a single connected set of floors.
                     # If there's only one floor, it's the bottom floor.
                     if len(all_floors) == 1:
                         bottom_floor = floor
                         break
                     continue # Skip this floor, it's not the bottom of the main stack

                bottom_floor = floor
                break # Found the bottom floor

        if bottom_floor is None and all_floors:
             # This might happen if floors form a cycle or are completely disconnected
             # In a standard miconic problem, there's a clear bottom floor.
             # As a fallback, if we can't find a floor that's never the lower floor,
             # we might pick one arbitrarily or handle this as an unsolvable state (h=infinity).
             # For simplicity, assume the standard structure and raise an error or handle gracefully.
             # Let's assume the first floor found is the bottom if only one exists.
             if len(all_floors) == 1:
                 bottom_floor = list(all_floors)[0]
             else:
                 # Fallback: If multiple floors but no clear bottom, this heuristic might be unreliable.
                 # We could try to find a floor that appears least often as the upper floor?
                 # Or just pick one and hope for the best, or return infinity.
                 # Let's assume a valid structure and proceed, or return a large value if structure is weird.
                 # For now, let's assume bottom_floor is found correctly in valid problems.
                 pass # bottom_floor remains None if not found

        if bottom_floor is not None:
            # Assign levels starting from the bottom floor
            current_floor = bottom_floor
            level = 0
            while current_floor is not None:
                self.floor_levels[current_floor] = level
                level += 1
                current_floor = floor_immediately_above_map.get(current_floor)

        # Check if all floors found have been assigned a level.
        # If not, there might be disconnected components or errors in 'above' facts.
        # For this heuristic, we assume a single vertical stack.
        if len(self.floor_levels) != len(all_floors):
             # Handle error or return infinity? For greedy search, a large value is fine.
             # print(f"Warning: Could not assign levels to all floors. Found {len(all_floors)} floors, leveled {len(self.floor_levels)}")
             # print(f"All floors: {all_floors}")
             # print(f"Leveled floors: {self.floor_levels.keys()}")
             # print(f"Above map: {floor_immediately_above_map}")
             # This heuristic might be inaccurate or fail if the floor structure is complex.
             # We proceed with the levels we found, assuming relevant floors are leveled.
             pass


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # This should not happen in a valid miconic state, but handle defensively
             # print("Error: Lift location not found in state.")
             return float('inf') # Or a large value

        current_level = self.floor_levels.get(current_lift_floor)
        if current_level is None:
             # This might happen if the lift is on a floor not included in 'above' facts
             # and not assigned a level. Heuristic cannot compute distance.
             # print(f"Error: Lift is at floor {current_lift_floor} which has no assigned level.")
             return float('inf') # Or a large value


        unserved_passengers = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                p = parts[1]
                if goal not in state:
                    unserved_passengers.add(p)

        total_non_move_actions = 0
        required_floors = set()

        for p in unserved_passengers:
            p_origin = self.passenger_origin.get(p)
            p_destin = self.passenger_destin.get(p)

            if p_origin is None or p_destin is None:
                 # Should not happen in valid problems, but handle defensively
                 # print(f"Warning: Origin or destination not found for passenger {p}")
                 continue # Cannot compute heuristic for this passenger

            # Check if passenger is at origin
            if f"(origin {p} {p_origin})" in state:
                total_non_move_actions += 2 # Needs board and depart
                required_floors.add(p_origin)
                required_floors.add(p_destin) # Needs to go to destination eventually
            # Check if passenger is boarded
            elif f"(boarded {p})" in state:
                total_non_move_actions += 1 # Needs depart
                required_floors.add(p_destin)

            # Note: If passenger is neither at origin nor boarded, and not served,
            # this state might be invalid or represent a passenger who was dropped
            # off at the wrong floor (not possible with current actions).
            # Assuming valid states only have passengers at origin, boarded, or served.


        # Calculate move actions
        if not required_floors:
            # All passengers served or no passengers needed service initially
            return 0 # Goal state handled above, but this is a safety check

        # Get levels for required floors, skipping any floor not in our level map
        required_levels = [self.floor_levels[f] for f in required_floors if f in self.floor_levels]

        if not required_levels:
             # This could happen if required floors were not assigned levels during init
             # print(f"Error: Required floors {required_floors} have no assigned levels.")
             return float('inf') # Cannot compute distance

        min_req_level = min(required_levels)
        max_req_level = max(required_levels)

        # Minimum moves to visit all required floors starting from current floor
        # Go to the closer extreme, then sweep across the range.
        move_cost = min(abs(current_level - min_req_level), abs(current_level - max_req_level)) + (max_req_level - min_req_level)

        return total_non_move_actions + move_cost

