# Assume Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of boarding/departing actions needed for unserved passengers
    and adds an estimate of the lift movement cost required to visit all necessary
    floors (origins for unboarded, destinations for boarded).

    # Assumptions
    - Floors are totally ordered, and the `(above f_higher f_lower)` facts define
      this order.
    - Passengers only need to be picked up at their origin and dropped off at their
      destination.
    - The lift can carry multiple passengers.
    - The goal is always to serve a specific set of passengers.
    - The 'above' facts in the static part of the domain/problem define the complete
      linear ordering of floors.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the initial state.
    - Determines the total order of floors based on `(above ...)` facts in the
      static part of the domain/problem and creates a mapping from floor name to index.
      It identifies the lowest floor as the one not appearing as the 'higher' floor
      in any 'above' fact, and then builds the ordered list upwards. If no 'above'
      facts exist, it defaults to sorting floors found in the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify Current State Information:**
       - Find the current floor of the lift by looking for the `(lift-at ?f)` fact.
       - Identify all passengers who are part of the goal (those mentioned in `(served ?p)` goals).
       - Identify which goal passengers are already served by checking for `(served ?p)` facts in the current state.

    2. **Calculate Action Costs and Required Floors:**
       - Initialize counts for `unboarded_count` and `boarded_count` to zero.
       - Initialize sets `pickup_floors` and `dropoff_floors` to store floors the lift must visit.
       - Iterate through each goal passenger:
         - If the passenger is already served, skip them.
         - If the passenger is not served:
           - Check if the passenger is at their origin floor (`(origin p f)`). If yes, increment `unboarded_count` (representing the needed `board` action) and add their origin floor `f` to `pickup_floors`.
           - Check if the passenger is currently boarded (`(boarded p)`). If yes, increment `boarded_count` (representing the needed `depart` action) and add their destination floor (retrieved from initialization) to `dropoff_floors`.

    3. **Calculate Base Heuristic:**
       - The base heuristic is the sum of the required boarding and departing actions: `h = unboarded_count + boarded_count`.

    4. **Estimate Lift Movement Cost:**
       - Combine `pickup_floors` and `dropoff_floors` into a single set `required_floors`. These are all the floors the lift must visit to pick up or drop off unserved passengers.
       - If `required_floors` is empty, no further movement is needed for pickups/dropoffs, and the movement cost is 0.
       - If `required_floors` is not empty:
         - Find the minimum and maximum floor indices among the `required_floors` using the pre-calculated `floor_to_index` map. Let these be `min_idx_req` and `max_idx_req`.
         - Get the index of the current lift floor, `idx_current`.
         - Calculate the distance between the minimum and maximum required floors: `dist_min_max = max_idx_req - min_idx_req`.
         - Calculate the distance from the current lift floor to the minimum required floor: `dist_current_min = abs(idx_current - min_idx_req)`.
         - Calculate the distance from the current lift floor to the maximum required floor: `dist_current_max = abs(idx_current - max_idx_req)`.
         - The estimated movement cost is the minimum distance to reach either end of the required floor range plus the distance to traverse the entire range: `movement_cost = min(dist_current_min, dist_current_max) + dist_min_max`.
         - Add this `movement_cost` to the total heuristic `h`.

    5. **Return Heuristic Value:**
       - Return the calculated value `h`. The heuristic is 0 if and only if all goal passengers are served.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and floor order.
        """
        # self.task = task # Not strictly needed if we store required info
        self.goals = task.goals

        # Extract passenger destinations from initial state
        self.passenger_destinations = {}
        for fact in task.initial_state:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.passenger_destinations[passenger] = destination_floor

        # Determine floor order and create floor-to-index map
        all_floors_mentioned = set()
        # Map lower floor to higher floor based on (above f_higher f_lower)
        higher_than_map = {}
        # Keep track of floors that appear as f_higher in (above f_higher f_lower)
        floors_that_are_higher = set()

        for fact in task.static:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                all_floors_mentioned.add(f_higher)
                all_floors_mentioned.add(f_lower)
                higher_than_map[f_lower] = f_higher
                floors_that_are_higher.add(f_higher)

        # Find the lowest floor: a floor that is mentioned but is not the 'higher' floor in any 'above' fact.
        lowest_floor = None
        for floor in all_floors_mentioned:
            if floor not in floors_that_are_higher:
                lowest_floor = floor
                break # Found the lowest

        if lowest_floor is None and all_floors_mentioned:
             # This case might occur if there's only one floor or the 'above' facts
             # don't form a clear chain starting from a unique lowest floor.
             # If only one floor, the order doesn't matter.
             if len(all_floors_mentioned) == 1:
                 self.ordered_floors = list(all_floors_mentioned)
             else:
                 # Fallback: If a clear lowest floor isn't found (e.g., disconnected floors, or complex above relations)
                 # default to sorting floors alphabetically or numerically if possible.
                 # Given the typical miconic structure, this fallback indicates an unexpected problem file.
                 # Sorting numerically is common for f1, f2, ...
                 try:
                     self.ordered_floors = sorted(list(all_floors_mentioned), key=lambda f: int(f[1:]))
                 except ValueError:
                      # If floor names are not f<number>, sort alphabetically
                      self.ordered_floors = sorted(list(all_floors_mentioned))

        elif lowest_floor is not None:
             # Build ordered list starting from lowest_floor by following the chain f_lower -> f_higher
             self.ordered_floors = [lowest_floor]
             current = lowest_floor
             while current in higher_than_map:
                 next_floor = higher_than_map[current]
                 self.ordered_floors.append(next_floor)
                 current = next_floor
        else: # No floors mentioned in static facts (e.g., single floor problem with no above facts)
             # Try to get floors from initial state facts like (lift-at f) or (origin p f)
             all_floors_mentioned_in_init = set()
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 if parts and parts[0] in ["lift-at", "origin", "destin"]:
                     if len(parts) > 1:
                         # The floor is usually the last argument
                         all_floors_mentioned_in_init.add(parts[-1])
             # Default sort if no above facts
             try:
                 self.ordered_floors = sorted(list(all_floors_mentioned_in_init), key=lambda f: int(f[1:]))
             except ValueError:
                 self.ordered_floors = sorted(list(all_floors_mentioned_in_init))


        # Create floor-to-index map
        self.floor_to_index = {floor: i for i, floor in enumerate(self.ordered_floors)}


    def get_floor_index(self, floor_name):
        """Helper to get the index of a floor."""
        # Return -1 or handle error if floor not found - should not happen for valid floors
        return self.floor_to_index.get(floor_name, -1)

    def get_distance(self, floor1_name, floor2_name):
        """Helper to get the distance between two floors."""
        idx1 = self.get_floor_index(floor1_name)
        idx2 = self.get_floor_index(floor2_name)
        # Check for -1 in production code if get_floor_index can return -1
        return abs(idx1 - idx2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # 1. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        if current_lift_floor is None:
             # Should not happen in a valid state reachable from initial state
             # where lift-at is always true for some floor.
             # Return a large value indicating this is likely an invalid or unsolvable state.
             return float('inf')

        # 2. Identify goal passengers
        goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # 3. Calculate action costs and identify required floors
        unboarded_count = 0
        boarded_count = 0
        pickup_floors = set()
        dropoff_floors = set()

        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for passenger in goal_passengers:
            if passenger in served_passengers:
                continue # Passenger is already served

            # Check if passenger is at origin
            is_at_origin = False
            origin_floor = None
            # Iterate through state to find origin fact for this passenger
            for fact in state:
                 if match(fact, "origin", passenger, "*"):
                     _, _, origin_floor = get_parts(fact)
                     is_at_origin = True
                     break

            if is_at_origin:
                unboarded_count += 1
                if origin_floor: # origin_floor should be found if is_at_origin is True
                    pickup_floors.add(origin_floor)
            elif f"(boarded {passenger})" in state:
                boarded_count += 1
                # Destination floor was stored in __init__
                if passenger in self.passenger_destinations:
                    dropoff_floors.add(self.passenger_destinations[passenger])
                # else: Passenger is boarded but destination unknown/not a goal passenger?
                # Assuming valid problem setup, this case implies the passenger is boarded
                # and their destination is in self.passenger_destinations.

        # 6. Calculate base heuristic from actions
        h = unboarded_count + boarded_count

        # 7. Calculate movement cost
        required_floors = pickup_floors | dropoff_floors

        if required_floors:
            # Find min and max index among required floors
            required_indices = [self.get_floor_index(f) for f in required_floors if self.get_floor_index(f) != -1]
            if required_indices: # Ensure there are valid floors in the set
                min_idx_req = min(required_indices)
                max_idx_req = max(required_indices)

                idx_current = self.get_floor_index(current_lift_floor)

                # Calculate distances
                dist_min_max = max_idx_req - min_idx_req # Distance between min and max required floors
                dist_current_min = abs(idx_current - min_idx_req) # Distance from current to min required
                dist_current_max = abs(idx_current - max_idx_req) # Distance from current to max required

                # Estimated movement cost: reach one end of the required range, then traverse the range
                movement_cost = min(dist_current_min, dist_current_max) + dist_min_max

                h += movement_cost
            # else: required_floors contained only invalid floor names? Should not happen.


        # Heuristic is 0 only for goal states.
        # If h > 0, it's not a goal state. If h == 0, it should be a goal state.
        # This is true because h=0 implies unboarded_count=0, boarded_count=0, and required_floors is empty.
        # unboarded_count=0 and boarded_count=0 means all goal passengers are neither at origin nor boarded.
        # In a solvable problem, this implies they must be served.

        return h
