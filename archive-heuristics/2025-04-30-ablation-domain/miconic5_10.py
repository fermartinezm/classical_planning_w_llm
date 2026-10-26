from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions (moves, board, depart)
    required to serve all passengers. It considers the vertical travel needed
    to visit all floors where passengers need to be picked up or dropped off,
    plus the individual board and depart actions.

    # Assumptions
    - The domain uses standard miconic actions: move (up/down), board, depart.
    - Floor levels are implicitly defined by the 'above' predicates, forming a total order.
    - Each move action changes the floor by one level.
    - Each board/depart action involves one passenger.

    # Heuristic Initialization
    - Parses the static facts to determine the floor ordering and create a mapping
      from floor names to integer levels.
    - Parses the static facts to store passenger destinations.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Parse State:** Identify the current lift location, waiting passengers
       and their origin floors, and boarded passengers and their destination floors.
       Also, identify served passengers (these require no further actions).

    2. **Identify Required Stops:** Determine the set of floors the lift must visit
       to serve the remaining passengers. This includes the origin floor for every
       waiting passenger and the destination floor for every boarded passenger.

    3. **Calculate Travel Cost:**
       - If there are no required stops, the travel cost is 0.
       - If there are required stops, find the minimum and maximum floor levels
         among these stops.
       - Calculate the current lift floor level.
       - The estimated travel cost is the minimum number of moves required to
         travel from the current lift level to cover the range between the minimum
         and maximum required stop levels. This is calculated as:
         - If the current level is below the minimum required level: `max_level - current_level`.
         - If the current level is above the maximum required level: `current_level - min_level`.
         - If the current level is within the range [min_level, max_level]:
           `(max_level - min_level) + min(current_level - min_level, max_level - current_level)`.

    4. **Count Board/Depart Actions:**
       - Count the number of passengers who are currently waiting at their origin.
         Each requires one 'board' action.
       - Count the number of passengers who are currently boarded and not yet served.
         Each requires one 'depart' action at their destination.

    5. **Sum Costs:** The total heuristic value is the sum of the estimated travel
       cost, the number of required board actions, and the number of required
       depart actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Build floor level mapping from 'above' facts
        floors = set()
        above_pairs = []
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                floors.add(f_above)
                floors.add(f_below)
                above_pairs.append((f_above, f_below))

        # Correct logic for floor levels:
        # (above f_i f_j) means f_i is at a higher level than f_j.
        # Count how many floors are *below* a given floor f.
        # A floor f_other is below f if (above f f_other) is true.
        # The number of floors below f determines its level.
        floor_below_counts = {f: 0 for f in floors}
        for f_above, f_below in above_pairs:
            # f_above is above f_below, so f_below is one of the floors below f_above
            floor_below_counts[f_above] += 1

        # The floor with the lowest count_below is the lowest floor (level 1).
        # The floor with k floors below it is level k+1.
        # Sort floors by count_below to get them in increasing order of level.
        sorted_floors_by_below_count = sorted(list(floors), key=lambda f: floor_below_counts[f])

        self.floor_to_level = {f: i + 1 for i, f in enumerate(sorted_floors_by_below_count)}

        # 2. Store passenger destinations
        self.passenger_destins = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.passenger_destins[passenger] = floor

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Parse State
        lift_current_floor = None
        passenger_origins = {} # {p: f_origin} for waiting passengers
        passenger_boarded = set() # {p} for boarded passengers
        served_passengers = set() # {p} for served passengers

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "lift-at":
                lift_current_floor = parts[1]
            elif predicate == "origin":
                p, f = parts[1], parts[2]
                passenger_origins[p] = f
            elif predicate == "boarded":
                p = parts[1]
                passenger_boarded.add(p)
            elif predicate == "served":
                p = parts[1]
                served_passengers.add(p)

        # Identify unserved passengers
        all_passengers = set(self.passenger_destins.keys())
        unserved_passengers = all_passengers - served_passengers

        # 2. Identify Required Stops
        floors_to_stop = set()
        passengers_waiting_unserved = set()
        passengers_boarded_unserved = set()

        for p in unserved_passengers:
            if p in passenger_origins: # Passenger is waiting
                floors_to_stop.add(passenger_origins[p])
                passengers_waiting_unserved.add(p)
            elif p in passenger_boarded: # Passenger is boarded and not served
                floors_to_stop.add(self.passenger_destins[p])
                passengers_boarded_unserved.add(p)
            # else: passenger is neither waiting nor boarded -> must be served (handled by unserved_passengers set)

        # 3. Calculate Travel Cost
        travel_cost = 0
        if floors_to_stop:
            levels_to_stop = {self.floor_to_level[f] for f in floors_to_stop}
            min_l = min(levels_to_stop)
            max_l = max(levels_to_stop)
            l_current = self.floor_to_level[lift_current_floor]

            if l_current < min_l:
                travel_cost = max_l - l_current
            elif l_current > max_l:
                travel_cost = l_current - min_l
            else: # min_l <= l_current <= max_l
                travel_cost = (max_l - min_l) + min(l_current - min_l, max_l - l_current)

        # 4. Count Board/Depart Actions
        board_actions_needed = len(passengers_waiting_unserved)
        depart_actions_needed = len(passengers_boarded_unserved)

        # 5. Sum Costs
        total_cost = travel_cost + board_actions_needed + depart_actions_needed

        return total_cost
