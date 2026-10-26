from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and strip potential leading/trailing whitespace
    fact_str = str(fact).strip()
    # Remove outer parentheses and split by spaces
    return fact_str[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments,
    # and then check if each part matches the corresponding argument pattern.
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the estimated number of board and depart actions
    needed for unserved passengers and adds an estimate of the lift's
    required movement to visit all necessary floors.

    # Assumptions
    - Actions have unit cost.
    - The floor ordering is defined by the `above` predicates, indicating
      immediately adjacent floors.
    - Passengers are either at their origin, boarded, or served.
    - Objects starting with 'p' are passengers, and objects starting with 'f'
      are floors. This is inferred from typical Miconic instances and PDDL
      naming conventions, as type information is not directly available
      in the provided Task object structure.

    # Heuristic Initialization
    - Parses static facts (`task.static`) to determine the floor ordering
      and create a mapping from floor names to numerical levels.
    - Parses static facts (`task.static`) to map each passenger to their
      destination floor.
    - Collects all passenger names and floor names from the initial state
      and goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Iterate through all known passengers to determine their status:
       - If a passenger `p` is `(served p)`, they contribute 0 to the heuristic.
       - If a passenger `p` is not `(served p)`:
         - Increment the count of unserved passengers.
         - If `(boarded p)` is true, the passenger is boarded and needs to depart at their destination. Add their destination floor to the set of 'required stops'.
         - If `(origin p o)` is true (and not boarded), the passenger is waiting at `o`. Increment the count of unboarded unserved passengers. Add their origin floor `o` to the set of 'required stops'.
    3. If there are no unserved passengers, the state is a goal state, and the heuristic is 0.
    4. Calculate the base heuristic value: This is the sum of board actions needed (one for each unboarded unserved passenger) and depart actions needed (one for each unserved passenger). Base = `num_unboarded_unserved + num_unserved`.
    5. Calculate the estimated movement cost:
       - Find the numerical levels for the current lift floor and all floors in the 'required stops' set using the pre-calculated floor-to-level map.
       - Find the minimum and maximum levels among the 'required stops'.
       - The estimated movement cost is the minimum number of steps required for the lift to travel from its current level to visit all levels within the range [min_required_level, max_required_level]. This is estimated as the cost to reach either the min or max required level, plus the distance to sweep across the entire range. Specifically, `min(abs(current_level - min_required_level) + (max_required_level - min_required_level), abs(current_level - max_required_level) + (max_required_level - min_required_level))`.
    6. The total heuristic value is the sum of the base heuristic and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about
        floor ordering and passenger destinations.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Collect all floors and passengers mentioned in the problem
        all_objects = set()
        for fact in task.initial_state | task.goals:
             all_objects.update(get_parts(fact)[1:]) # Add all arguments as objects

        self.all_floors = sorted([obj for obj in all_objects if obj.startswith('f')])
        self.all_passengers = sorted([obj for obj in all_objects if obj.startswith('p')])

        # Build floor level mapping from 'above' predicates
        self.floor_to_level = {}
        if len(self.all_floors) == 1:
            self.floor_to_level[self.all_floors[0]] = 0
        elif self.all_floors:
            floor_above_map = {}
            # Find floors that are immediately above others
            floors_that_are_above = set()
            for fact in self.static_facts:
                if match(fact, "above", "*", "*"):
                    f_high, f_low = get_parts(fact)[1:]
                    floor_above_map[f_low] = f_high
                    floors_that_are_above.add(f_high)

            # Find the lowest floor (a floor that is not above any other floor)
            lowest_floor = None
            for floor in self.all_floors:
                 if floor not in floors_that_are_above:
                     lowest_floor = floor
                     break

            if lowest_floor is None and self.all_floors:
                 # This case might happen if the above predicates form a cycle or are incomplete,
                 # or if the lowest floor is not mentioned as f_low in any fact but is present.
                 # A safer way is to find a floor that is a key in floor_above_map but not a value.
                 # Or simply pick the first floor in the sorted list as a fallback, though this is fragile.
                 # Let's try finding a key that isn't a value.
                 floors_that_are_below = set(floor_above_map.keys())
                 floors_that_are_above_values = set(floor_above_map.values())
                 potential_lowest = floors_that_are_below - floors_that_are_above_values
                 if potential_lowest:
                     lowest_floor = sorted(list(potential_lowest))[0] # Pick one deterministically
                 else:
                     # Fallback: assume the first floor in the sorted list is the lowest
                     # This is a weak assumption but handles cases where above facts are minimal
                     lowest_floor = self.all_floors[0]


            # Build the level map by traversing up from the lowest floor
            if lowest_floor:
                current_floor = lowest_floor
                level = 0
                while current_floor in self.all_floors: # Ensure we don't go beyond known floors
                    self.floor_to_level[current_floor] = level
                    if current_floor in floor_above_map:
                        current_floor = floor_above_map[current_floor]
                        level += 1
                    else:
                        break # Reached the highest floor

        # Map passengers to their destination floors
        self.passenger_to_destin = {}
        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                p, d = get_parts(fact)[1:]
                self.passenger_to_destin[p] = d

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # Should not happen in a valid miconic state, but handle defensively
             return float('inf') # Or some large value indicating an invalid state

        num_unserved = 0
        num_unboarded_unserved = 0
        required_stops = set() # Floors the lift must visit

        # Identify unserved passengers and required stops
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_facts = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}

        for passenger in self.all_passengers:
            if passenger not in served_passengers:
                num_unserved += 1
                destin_floor = self.passenger_to_destin.get(passenger) # Get destination from pre-calculated map

                if passenger in boarded_passengers:
                    # Passenger is boarded, needs to go to destination
                    if destin_floor:
                        required_stops.add(destin_floor)
                elif passenger in origin_facts:
                    # Passenger is unboarded, waiting at origin
                    num_unboarded_unserved += 1
                    origin_floor = origin_facts[passenger]
                    required_stops.add(origin_floor)
                    # Also need to add destination floor to required stops, as they will need to be dropped off later
                    if destin_floor:
                         required_stops.add(destin_floor)


        # If no unserved passengers were found but goal is not reached, something is wrong.
        # This check is redundant if goals <= state check passed, but good for robustness.
        if num_unserved == 0:
             return 0

        # Base heuristic: count board and depart actions needed
        # Each unboarded unserved needs 1 board. Each unserved needs 1 depart.
        base_heuristic = num_unboarded_unserved + num_unserved

        # Movement cost
        movement_cost = 0
        if required_stops:
            required_levels = [self.floor_to_level.get(f, None) for f in required_stops]
            # Filter out any floors not found in the level map (shouldn't happen if parsing is correct)
            required_levels = [level for level in required_levels if level is not None]

            if required_levels:
                current_level = self.floor_to_level.get(current_lift_floor, None)
                if current_level is not None:
                    min_level_required = min(required_levels)
                    max_level_required = max(required_levels)

                    # Estimate movement cost as minimum of sweeping up or sweeping down
                    # Cost to reach min_level + range_size OR Cost to reach max_level + range_size
                    range_size = max_level_required - min_level_required
                    cost_sweep_up = abs(current_level - min_level_required) + range_size
                    cost_sweep_down = abs(current_level - max_level_required) + range_size
                    movement_cost = min(cost_sweep_up, cost_sweep_down)
                else:
                    # Current lift floor not in level map - should not happen
                    movement_cost = float('inf') # Indicate problematic state
            # else: required_stops was not empty, but none of the floors were in the level map - indicates parsing issue.
            # movement_cost remains 0, which might be misleading but prevents infinite loop.

        # Total heuristic is sum of base actions and estimated movement
        return base_heuristic + movement_cost

