from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, though state facts should be strings in parentheses
        return []
    return fact[1:-1].split()

# Helper function to match facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts is sufficient and if the parts match the pattern
    if len(parts) < len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts[:len(args)], args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the necessary board/depart actions and estimates
    the lift movement cost based on the range of floors that need to be visited
    to pick up or drop off passengers.

    # Assumptions
    - Floors are linearly ordered and can be mapped to integer levels based
      on their names (e.g., f1, f2, ... fn correspond to levels 1, 2, ... n).
      This assumes floor names are consistently in the format 'f<number>'.
    - The lift has unlimited capacity.
    - The estimated move cost assumes a strategy where the lift travels
      to one extreme of the required floors and then sweeps to the other.

    # Heuristic Initialization
    - Identifies all floor objects by parsing facts in the initial state,
      goals, and static facts.
    - Creates a mapping from floor names to integer levels by sorting the
      identified floor names numerically (based on the number part of 'f<number>').
    - Stores the destination floor for each passenger by parsing `destin`
      facts from the initial state.
    - Identifies all passenger objects by parsing relevant facts in the
      initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the lift's current floor and its corresponding level using
       the `(lift-at ?f)` fact.
    2. Identify all passengers who are currently served using the `(served ?p)` fact.
    3. Initialize counters for board actions needed and depart actions needed,
       and a set for required floors.
    4. Identify passengers currently at their origin floor and passengers
       currently boarded by parsing the state facts.
    5. Iterate through all known passengers:
       - If a passenger is not in the set of served passengers:
         - If the passenger is at their origin floor:
           - Increment board actions needed by 1.
           - Add the passenger's origin floor to the set of required floors.
           - Increment depart actions needed by 1 (they will need to depart later).
           - Add the passenger's destination floor to the set of required floors.
         - Else if the passenger is currently boarded:
           - Increment depart actions needed by 1.
           - Add the passenger's destination floor to the set of required floors.
    6. If there are no required floors, the move cost is 0.
    7. If there are required floors, find the minimum and maximum floor levels
       among them using the floor-to-level mapping.
    8. Estimate the move cost: Calculate the distance from the lift's current
       floor level to the nearest extreme of the required floor range, plus
       the distance spanning the required floor range.
       - `current_level = level(lift_floor)`
       - `min_level = min(level(f) for f in required_floors)`
       - `max_level = max(level(f) for f in required_floors)`
       - If `current_level < min_level`: `move_cost = (min_level - current_level) + (max_level - min_level)`
       - If `current_level > max_level`: `move_cost = (current_level - max_level) + (max_level - min_level)`
       - If `min_level <= current_level <= max_level`: `move_cost = min(current_level - min_level, max_level - current_level) + (max_level - min_level)`
    9. The total heuristic value is the sum of the number of board actions needed,
       the number of depart actions needed, and the estimated move cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering and passenger goals.
        """
        self.goals = task.goals
        initial_state = task.initial_state
        static_facts = task.static

        # 1. Determine floor ordering and create level mapping
        floor_names = set()
        all_facts = set(initial_state) | set(self.goals) | set(static_facts)

        for fact in all_facts:
             parts = get_parts(fact)
             # Check for predicates that involve floors
             if parts and parts[0] in ['origin', 'destin', 'above', 'lift-at']:
                 # Floors are arguments. Check if they look like floor names.
                 for part in parts[1:]:
                     if isinstance(part, str) and part.startswith('f') and part[1:].isdigit():
                          floor_names.add(part)

        # Sort floor names based on the number part (e.g., f1, f10, f2 -> f1, f2, f10)
        try:
            sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))
        except ValueError:
             print(f"Warning: Floor names {floor_names} do not follow 'f<number>' format. Sorting alphabetically.")
             sorted_floor_names = sorted(list(floor_names))

        self.floor_to_level = {floor: i + 1 for i, floor in enumerate(sorted_floor_names)}

        # 2. Store passenger destinations and identify all passengers
        self.passenger_destinations = {}
        self.all_passengers = set()

        for fact in initial_state:
             predicate, *args = get_parts(fact)
             if predicate == "destin":
                  passenger, destination_floor = args
                  self.passenger_destinations[passenger] = destination_floor
                  self.all_passengers.add(passenger)
             elif predicate == "origin":
                  passenger, origin_floor = args
                  self.all_passengers.add(passenger)
             elif predicate == "boarded":
                  passenger = args[0]
                  self.all_passengers.add(passenger)

        # Also add passengers from goals if any are missed (e.g., goal (served p1) but p1 not in initial state facts)
        for goal in self.goals:
             predicate, *args = get_parts(goal)
             if predicate == "served":
                  self.all_passengers.add(args[0])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Find the lift's current floor and level
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown and goal is not reached, return a large value.
        # This handles potential invalid states or unexpected goal conditions.
        if lift_floor is None:
             if self.goals <= state:
                  return 0 # Goal reached
             else:
                  # Should not happen in valid miconic states
                  print(f"Warning: Lift location not found in state: {state}")
                  return 1000000 # Large penalty

        lift_level = self.floor_to_level[lift_floor]

        # 2. Identify served passengers
        served_passengers = set()
        for fact in state:
             predicate, *args = get_parts(fact)
             if predicate == "served":
                  served_passengers.add(args[0])

        # 3-5. Calculate tasks and identify required floors
        board_actions_needed = 0
        depart_actions_needed = 0
        required_floors = set() # Floors the lift must visit

        # Identify passengers currently at their origin floor and passengers currently boarded
        passengers_at_origin = set()
        passengers_boarded = set()

        for fact in state:
             if match(fact, "origin", "*", "*"):
                  p = get_parts(fact)[1]
                  passengers_at_origin.add(p)
             elif match(fact, "boarded", "*"):
                  p = get_parts(fact)[1]
                  passengers_boarded.add(p)


        for passenger in self.all_passengers:
             if passenger not in served_passengers:
                  # If passenger is at origin, they need pickup and dropoff
                  if passenger in passengers_at_origin:
                       board_actions_needed += 1
                       # Find origin floor for this passenger
                       origin_floor = None
                       for fact in state:
                            if match(fact, "origin", passenger, "*"):
                                 origin_floor = get_parts(fact)[2]
                                 break
                       if origin_floor:
                            required_floors.add(origin_floor)

                       # They will also need a dropoff later
                       depart_actions_needed += 1
                       destination_floor = self.passenger_destinations.get(passenger)
                       if destination_floor:
                            required_floors.add(destination_floor)

                  # Elif passenger is boarded, they only need dropoff
                  elif passenger in passengers_boarded:
                       depart_actions_needed += 1
                       destination_floor = self.passenger_destinations.get(passenger)
                       if destination_floor:
                            required_floors.add(destination_floor)

        # 6-8. Calculate move cost
        move_cost = 0
        if required_floors:
            required_levels = [self.floor_to_level[f] for f in required_floors]
            min_level = min(required_levels)
            max_level = max(required_levels)

            current_level = self.floor_to_level[lift_floor]

            # Calculate distance to the range [min_level, max_level] and the range size
            range_dist = max_level - min_level

            if current_level < min_level:
                # Must go up to min_level, then sweep up to max_level
                move_cost = (min_level - current_level) + range_dist
            elif current_level > max_level:
                # Must go down to max_level, then sweep down to min_level
                move_cost = (current_level - max_level) + range_dist
            else: # current_level is within [min_level, max_level]
                # Go to min_level (cost current_level - min_level), then sweep up (cost range_dist)
                cost_go_min_sweep_up = (current_level - min_level) + range_dist
                # Go to max_level (cost max_level - current_level), then sweep down (cost range_dist)
                cost_go_max_sweep_down = (max_level - current_level) + range_dist
                move_cost = min(cost_go_min_sweep_up, cost_go_max_sweep_down)

        # 9. Total heuristic value
        total_heuristic = board_actions_needed + depart_actions_needed + move_cost

        return total_heuristic
