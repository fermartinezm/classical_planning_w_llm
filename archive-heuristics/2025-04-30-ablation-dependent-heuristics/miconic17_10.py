import re

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the cost to reach the goal state (all passengers served) by summing
    the number of required 'board' actions, the number of required 'depart' actions,
    and an estimate of the lift movement cost. The movement cost is estimated
    based on the current lift position and the range of floors that need to be
    visited for pickups and dropoffs.

    Assumptions:
    - The PDDL instance is well-formed for the miconic domain.
    - Floors are ordered linearly, and the 'above' predicates define this order.
      It attempts to parse floor names like 'f<number>' to determine levels.
      If this fails or no 'above' facts exist, it falls back to assigning level 0
      to all floors, effectively ignoring movement cost.
    - All passengers mentioned in 'origin' or 'boarded' facts have a corresponding
      'destin' fact in the static information.
    - The state representation includes '(lift-at ?f)' for the current lift position.

    Heuristic Initialization:
    - Stores the task object.
    - Parses static facts to build a dictionary mapping passengers to their
      destination floors (`self.destinations`).
    - Collects all passengers mentioned in static, initial, or goal states.
    - Collects all floors mentioned in static, initial, or goal states.
    - Attempts to determine floor levels (`self.floor_levels`):
      - First, tries to parse floor names like 'f<number>' and use the number
        (relative to the minimum number found) as the level. This is the preferred
        method for efficiency if applicable.
      - If any floor name does not match the 'f<number>' pattern, or if no
        'above' facts are present in the static information (which are needed
        to validate the numerical order or build levels via graph), it falls back
        to assigning level 0 to all identified floors. This simplifies the
        movement cost calculation to 0.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state satisfies the goal conditions (all passengers served). Iterate through all known passengers and check if `(served p)` is in the state. If all are served, return 0.
    2. Find the current floor of the lift by searching for a fact starting with `(lift-at `. If not found, return infinity (indicating an unexpected or invalid state).
    3. Initialize counters for unboarded and boarded unserved passengers, and sets for required pickup and dropoff floors.
    4. Build temporary lookup structures (dictionaries/sets) for `origin`, `boarded`, and `served` facts present in the current state for efficient access.
    5. Iterate through all known passengers. If a passenger is not in the `served` set:
       - Check if the passenger is in the `boarded` set. If yes, increment the boarded count and add their destination floor (looked up from `self.destinations`) to the `dropoff_floors` set.
       - If the passenger is not boarded, check if they are in the `origin_map`. If yes, increment the unboarded count, add their origin floor to the `pickup_floors` set, and add their destination floor to the `dropoff_floors` set.
    6. Combine `pickup_floors` and `dropoff_floors` into a single set `required_floors`.
    7. Calculate the estimated cost:
       - `board_actions` = count of unboarded passengers.
       - `depart_actions` = count of boarded unserved passengers.
       - `movement_cost`:
         - If `required_floors` is empty, movement cost is 0.
         - If `required_floors` is not empty and floor levels were successfully determined (i.e., `self._floor_levels_reliable` is True):
           - Get the levels for all `required_floors` and the `current_f`. Handle potential missing floors by assigning a default level (e.g., 0), although this indicates a potential issue with the problem definition or level parsing.
           - Find the minimum (`min_level`) and maximum (`max_level`) levels among the `required_levels`.
           - The movement cost is estimated as the minimum travel distance to visit all floors in the range [min_level, max_level] starting from `current_level`. This is calculated as `(max_level - min_level) + min(abs(current_level - min_level), abs(current_level - max_level))`.
         - If floor levels could not be determined reliably (`self._floor_levels_reliable` is False), the movement cost is 0.
    8. The total heuristic value is the sum of `board_actions`, `depart_actions`, and `movement_cost`.
    """
    def __init__(self, task):
        self.task = task
        self.destinations = {}
        self.floor_levels = {}
        self.all_passengers = set()
        self._floor_levels_reliable = False # Flag to indicate if movement cost is meaningful

        # Extract destinations and collect all passengers
        for fact in task.static:
            if fact.startswith('(destin '):
                parts = fact.strip('()').split()
                passenger = parts[1]
                floor = parts[2]
                self.destinations[passenger] = floor
                self.all_passengers.add(passenger)

        # Collect all floors mentioned in static, initial, and goal states
        floors = set()
        above_relations_exist = False
        for fact in task.static:
            if fact.startswith('(above '):
                above_relations_exist = True
                parts = fact.strip('()').split()
                floors.add(parts[1])
                floors.add(parts[2])

        for fact in task.initial_state | task.goals:
             parts = fact.strip('()').split()
             for part in parts[1:]:
                 # Simple check: assume objects starting with 'f' are floors
                 if part.startswith('f'):
                     floors.add(part)

        if not floors:
             # No floors found at all - assign level 0 to any potential floors found
             potential_floors = set()
             for fact in task.initial_state | task.goals:
                 parts = fact.strip('()').split()
                 for part in parts[1:]:
                     if part.startswith('f'):
                         potential_floors.add(part)
             for floor in potential_floors:
                 self.floor_levels[floor] = 0
             self._floor_levels_reliable = False # Cannot use movement cost

        else:
            # Attempt to parse floor names like f<number>
            floor_numbers = {}
            can_use_number_levels = True
            for floor in floors:
                 match = re.match(r'f(\d+)', floor)
                 if match:
                     floor_numbers[floor] = int(match.group(1))
                 else:
                     can_use_number_levels = False
                     break # Cannot use number-based levels if any floor name doesn't match

            # We can only use number-based levels reliably if all floors match the pattern AND
            # there are 'above' relations present in static facts. The 'above' relations
            # implicitly validate that the numerical order corresponds to the physical order.
            # If no 'above' facts, we can't be sure f2 is above f1 even if they are named that way.
            if can_use_number_levels and floor_numbers and above_relations_exist:
                min_floor_num = min(floor_numbers.values())
                for floor, num in floor_numbers.items():
                    self.floor_levels[floor] = num - min_floor_num # Assign level 0 to the lowest floor number
                self._floor_levels_reliable = True # Number-based levels are reliable
            else:
                 # Fallback: Assign level 0 to all floors if f<number> parsing failed
                 # or if no 'above' relations exist to establish order.
                 for floor in floors:
                     self.floor_levels[floor] = 0
                 self._floor_levels_reliable = False # Cannot use movement cost reliably


        # Ensure all passengers from initial state/goal are in self.all_passengers
        for fact in task.initial_state | task.goals:
             parts = fact.strip('()').split()
             if parts[0] in ('origin', 'destin', 'boarded', 'served'):
                 if len(parts) > 1 and parts[1].startswith('p'): # Simple check for passenger naming
                     self.all_passengers.add(parts[1])


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # 1. Check if goal state
        is_goal = True
        for p in self.all_passengers:
            if f'(served {p})' not in state:
                is_goal = False
                break
        if is_goal:
            return 0

        # 2. Identify current floor
        current_f = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_f = fact.strip('()').split()[1]
                break
        if current_f is None:
             # Should not happen in a valid state according to PDDL semantics
             # Return infinity to prune this branch in search
             return float('inf')

        # 3. Identify unserved passengers (implicitly done in step 5)

        # 4. & 5. Determine required floors and passenger status
        pickup_floors = set()
        dropoff_floors = set()
        unboarded_passengers_count = 0
        boarded_passengers_count = 0

        # Build temporary lookup structures for efficient access
        origin_map = {} # passenger -> origin_floor
        boarded_set = set() # set of boarded passengers
        served_set = set() # set of served passengers

        for fact in state:
            if fact.startswith('(origin '):
                parts = fact.strip('()').split()
                p = parts[1]
                f = parts[2]
                origin_map[p] = f
            elif fact.startswith('(boarded '):
                parts = fact.strip('()').split()
                p = parts[1]
                boarded_set.add(p)
            elif fact.startswith('(served '):
                 parts = fact.strip('()').split()
                 p = parts[1]
                 served_set.add(p)

        for p in self.all_passengers:
            if p not in served_set:
                if p in boarded_set:
                    # Passenger is boarded and unserved, needs drop-off
                    boarded_passengers_count += 1
                    dest_f = self.destinations.get(p)
                    if dest_f: # Destination should exist for any passenger needing service
                        dropoff_floors.add(dest_f)
                    # else: This indicates an issue with the problem definition
                elif p in origin_map:
                    # Passenger is unboarded and unserved, needs pick-up and drop-off
                    unboarded_passengers_count += 1
                    origin_f = origin_map[p]
                    dest_f = self.destinations.get(p)
                    if origin_f: # Origin should exist
                        pickup_floors.add(origin_f)
                    if dest_f: # Destination should exist
                        dropoff_floors.add(dest_f)
                    # else: This indicates an issue with the problem definition
                # else: unserved but not boarded and not at origin? Invalid state?
                # Assume valid states where unserved passengers are either at origin or boarded.

        required_floors = pickup_floors | dropoff_floors

        # 6. Calculate estimated cost
        board_actions = unboarded_passengers_count
        depart_actions = boarded_passengers_count

        movement_cost = 0
        if required_floors and self._floor_levels_reliable:
            # Ensure all required floors and current floor have a level
            # If not, this indicates an issue with floor parsing or problem definition
            # Assign a default level (e.g., 0) if missing, though this might skew heuristic
            required_levels = []
            for f in required_floors:
                 level = self.floor_levels.get(f)
                 if level is None:
                      # Floor not found in parsed levels - fallback to level 0
                      level = 0
                      # print(f"Warning: Floor {f} not found in floor_levels. Using level 0.")
                 required_levels.append(level)

            current_level = self.floor_levels.get(current_f)
            if current_level is None:
                 # Current lift floor not found in levels - fallback to level 0
                 current_level = 0
                 # print(f"Warning: Current floor {current_f} not found in floor_levels. Using level 0.")


            min_level = min(required_levels)
            max_level = max(required_levels)

            # Movement cost calculation: distance to cover the range [min_level, max_level]
            # starting from current_level.
            # This is the distance from the current floor to the closest end of the range,
            # plus the distance to traverse the entire range.
            # Cost = (max_level - min_level) + min(abs(current_level - min_level), abs(current_level - max_level))
            movement_cost = (max_level - min_level) + min(abs(current_level - min_level), abs(current_level - max_level))

        # Total heuristic
        h = board_actions + depart_actions + movement_cost

        return h
