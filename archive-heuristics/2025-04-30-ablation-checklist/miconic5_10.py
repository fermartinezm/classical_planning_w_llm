from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys

# Increase recursion depth for floor mapping in large problems
sys.setrecursionlimit(2000)

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It counts the necessary board and depart actions and adds
    an estimate of the vertical travel distance the lift needs to cover
    to visit all required floors (origins for waiting passengers,
    destinations for boarded passengers).

    # Assumptions
    - Floors are linearly ordered by the `above` predicate.
    - All passengers must be served to reach the goal.
    - The cost of each action (board, depart, up, down) is 1.

    # Heuristic Initialization
    - Parses static facts to determine the floor order and map floor objects
      to integer floor numbers.
    - Stores the destination floor for each passenger from static facts.
    - Identifies the set of goal facts (all passengers served).

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Unserved Passengers:** Determine which passengers are not yet
        in the `(served ?p)` state. These are the passengers that still need
        to be transported.

    2.  **Count Required Board/Depart Actions:**
        - For each unserved passenger currently at their origin floor
          (`(origin ?p ?f)`), one `board` action is needed. Add 1 to the heuristic.
        - For each unserved passenger currently boarded (`(boarded ?p)`),
          one `depart` action is needed. Add 1 to the heuristic.
        - The total count of these actions is a lower bound on non-movement actions.

    3.  **Identify Required Floors:** Determine the set of floors the lift
        *must* visit to serve the remaining passengers:
        - The origin floor for every unserved passenger still waiting there.
        - The destination floor for every unserved passenger who is currently boarded.

    4.  **Estimate Travel Cost:** Calculate the minimum number of `up` or `down`
        actions required for the lift to travel from its current floor to visit
        all the required floors identified in step 3.
        - Find the lift's current floor.
        - Map all required floors and the current lift floor to their integer
          floor numbers using the pre-calculated mapping.
        - If there are no required floors, the travel cost is 0.
        - If there are required floors, find the minimum (`min_needed_num`) and
          maximum (`max_needed_num`) floor numbers among them.
        - The lift must travel from its current floor number (`current_floor_num`)
          to cover the vertical range from `min_needed_num` to `max_needed_num`.
        - A reasonable non-admissible estimate for this travel is the distance
          between the lowest and highest needed floors (`max_needed_num - min_needed_num`)
          plus the minimum distance from the current floor to either the lowest
          or highest needed floor (`min(|current_floor_num - min_needed_num|, |current_floor_num - max_needed_num|)`).
          This accounts for getting to the "span" of needed floors and then traversing the span.

    5.  **Sum Costs:** The total heuristic value is the sum of the required
        board/depart actions (step 2) and the estimated travel cost (step 4).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions (all passengers served).
        - Static facts (`above` relationships and passenger destinations).
        - Build the floor-to-integer mapping.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store destination floor for each passenger from static facts.
        self.destin_map = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destin_map[passenger] = floor

        # Build the floor-to-integer mapping based on 'above' predicates.
        self.floor_to_int = self._build_floor_mapping(static_facts)

        # Identify all passengers from goals (assuming all passengers need to be served)
        self.all_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def _build_floor_mapping(self, static_facts):
        """
        Parses 'above' facts to create a mapping from floor object names
        to integer floor numbers (1-based).
        Assumes floors form a single linear sequence.
        """
        floor_above_map = {}
        all_floors = set()
        floors_that_are_above = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_high, f_low = get_parts(fact)
                floor_above_map[f_low] = f_high
                all_floors.add(f_high)
                all_floors.add(f_low)
                floors_that_are_above.add(f_high)

        if not all_floors:
             # Handle case with no floors or no above facts (e.g., empty problem)
             return {}

        # Find the lowest floor: a floor that is not the value of any above mapping
        # (i.e., no floor is immediately below it).
        all_floors_list = list(all_floors)
        floors_below_others = set(floor_above_map.values())
        lowest_floor = None
        for floor in all_floors_list:
            if floor not in floors_below_others:
                lowest_floor = floor
                break

        if lowest_floor is None:
             # This might happen in cyclic or disconnected 'above' relations,
             # or if there's only one floor. If only one floor, it's the lowest.
             if len(all_floors) == 1:
                 lowest_floor = list(all_floors)[0]
             else:
                 # Fallback: If we can't find a clear lowest floor,
                 # maybe just sort alphabetically? Or raise an error?
                 # For standard miconic, finding the lowest should work.
                 # Let's assume standard structure and error if not found.
                 raise ValueError("Could not determine lowest floor from 'above' facts.")


        # Build the ordered list of floors starting from the lowest
        floors_ordered = [lowest_floor]
        current_floor = lowest_floor
        while current_floor in floor_above_map:
            current_floor = floor_above_map[current_floor]
            floors_ordered.append(current_floor)

        # Create the floor name to integer mapping
        floor_to_int_map = {floor: i + 1 for i, floor in enumerate(floors_ordered)}

        return floor_to_int_map

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If the goal is reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        # Find the current lift floor.
        current_lift_floor_str = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor_str = get_parts(fact)[1]
                break

        if current_lift_floor_str is None:
             # This should not happen in a valid miconic state, but handle defensively.
             # If lift location is unknown, maybe return infinity or a large number?
             # For this heuristic, let's assume a valid state with lift-at.
             # If it could happen, we'd need a different approach or a very high cost.
             # Let's return a large value indicating a problematic state for this heuristic.
             return float('inf')


        current_floor_num = self.floor_to_int.get(current_lift_floor_str)
        if current_floor_num is None:
             # Should not happen if floor mapping is built correctly and state is valid
             return float('inf')


        heuristic_cost = 0
        needed_floors_str_set = set()

        # Identify unserved passengers and count board/depart actions needed
        unserved_passengers = [p for p in self.all_passengers if f"(served {p})" not in state]

        for passenger in unserved_passengers:
            is_waiting = False
            is_boarded = False
            origin_floor = None

            # Check if waiting at origin
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    is_waiting = True
                    origin_floor = get_parts(fact)[2]
                    break

            # Check if boarded
            if not is_waiting and f"(boarded {passenger})" in state:
                 is_boarded = True

            if is_waiting:
                # Passenger needs to be boarded and then departed
                heuristic_cost += 2 # 1 for board, 1 for depart
                needed_floors_str_set.add(origin_floor)
                # Destination floor is also needed for this passenger's journey
                destin_floor = self.destin_map.get(passenger)
                if destin_floor:
                    needed_floors_str_set.add(destin_floor)

            elif is_boarded:
                # Passenger needs to be departed
                heuristic_cost += 1 # 1 for depart
                # Destination floor is needed for this passenger
                destin_floor = self.destin_map.get(passenger)
                if destin_floor:
                    needed_floors_str_set.add(destin_floor)
            # else: passenger is neither waiting nor boarded (e.g., already served, or in an unexpected state)
            # We only consider passengers needing board/depart actions.

        # Calculate estimated travel cost
        travel_cost = 0
        if needed_floors_str_set:
            needed_floor_nums = {self.floor_to_int[f] for f in needed_floors_str_set if f in self.floor_to_int}

            if needed_floor_nums: # Ensure the set is not empty after filtering
                min_needed_num = min(needed_floor_nums)
                max_needed_num = max(needed_floor_nums)

                if min_needed_num == max_needed_num:
                    # Only one floor needed, travel is distance to that floor
                    travel_cost = abs(current_floor_num - min_needed_num)
                else:
                    # Multiple floors needed, estimate travel to cover the range
                    # Distance to cover the range + distance from current floor to nearest end of range
                    range_distance = max_needed_num - min_needed_num
                    dist_to_min = abs(current_floor_num - min_needed_num)
                    dist_to_max = abs(current_floor_num - max_needed_num)
                    travel_cost = range_distance + min(dist_to_min, dist_to_max)

        heuristic_cost += travel_cost

        return heuristic_cost

