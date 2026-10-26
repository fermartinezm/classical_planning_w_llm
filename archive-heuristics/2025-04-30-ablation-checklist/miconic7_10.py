from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
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
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(args) > len(parts) and args[-1] != '*':
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It counts the necessary board and depart actions for unserved
    passengers and adds an estimate of the required lift travel.

    # Assumptions
    - Each unserved passenger requires one 'board' action (if at origin)
      and one 'depart' action (if boarded).
    - Lift travel is estimated based on the range of floors that need to be
      visited to pick up waiting passengers and drop off boarded passengers.
    - The lift has infinite capacity.

    # Heuristic Initialization
    - Extract static information: passenger destinations and the floor ordering
      defined by the `above` predicate.
    - Create a mapping from floor names (e.g., 'f1') to numerical floor levels.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Unserved Passengers:** Determine which passengers are not yet
        in the `(served ?p)` state. These are the passengers that still require
        actions.

    2.  **Count Board/Depart Actions:** For each unserved passenger:
        *   If the passenger is at their origin floor `(origin ?p ?f)`, they
            need to be boarded. Add 1 to the heuristic cost. Note their origin
            floor as a required stop for the lift.
        *   If the passenger is `(boarded ?p)`, they need to be departed at
            their destination floor. Add 1 to the heuristic cost. Note their
            destination floor as a required stop for the lift.

    3.  **Identify Required Floors:** Collect all unique origin floors of waiting
        passengers and all unique destination floors of boarded passengers. These
        are the floors the lift *must* visit.

    4.  **Estimate Travel Cost:** If there are required floors to visit:
        *   Find the lift's current floor.
        *   Find the minimum and maximum floor numbers among the required floors.
        *   Estimate the travel cost as the distance from the current floor to
            the furthest required floor, plus the distance needed to traverse
            the entire range of required floors. This is a non-admissible
            estimate designed to encourage moving towards the required floors
            and covering the necessary range. Specifically, calculate
            `max(abs(current_floor - min_req), abs(current_floor - max_req)) + (max_req - min_req)`.

    5.  **Sum Costs:** The total heuristic value is the sum of the board/depart
        action counts and the estimated travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Goal locations for each passenger.
        - Floor ordering from `above` predicates to create a floor-to-number mapping.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each passenger.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is (served ?p), need to find the destination from static facts
            predicate, passenger = get_parts(goal)
            if predicate == "served":
                 # Find the corresponding (destin ?p ?f) fact in static_facts
                 for fact in static_facts:
                     if match(fact, "destin", passenger, "*"):
                         _, p, dest_floor = get_parts(fact)
                         self.goal_locations[p] = dest_floor
                         break # Found destination for this passenger

        # Build floor name to number mapping based on (above f1 f2) facts
        # (above f1 f2) means f1 is immediately above f2
        below_map = {} # maps f1 -> f2 (floor immediately below f1)
        all_floors = set()
        floors_with_floor_below = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_above, f_below = get_parts(fact)
                below_map[f_above] = f_below
                floors_with_floor_below.add(f_above)
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the highest floor (a floor that is never a key in below_map)
        highest_floor = None
        for floor in all_floors:
            if floor not in floors_with_floor_below:
                 highest_floor = floor
                 break

        # Build the ordered list of floors from highest to lowest
        ordered_floors_desc = []
        current_floor = highest_floor
        while current_floor is not None:
            ordered_floors_desc.append(current_floor)
            current_floor = below_map.get(current_floor)

        # Reverse to get floors from lowest to highest
        ordered_floors_asc = list(reversed(ordered_floors_desc))

        # Create the floor name to number mapping (1-based indexing)
        self.floor_to_num = {floor_name: i + 1 for i, floor_name in enumerate(ordered_floors_asc)}
        self.num_to_floor = {i + 1: floor_name for i, floor_name in enumerate(ordered_floors_asc)} # Useful for debugging

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find the current lift floor
        current_lift_floor_name = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor_name = get_parts(fact)[1]
                break

        if current_lift_floor_name is None:
             # This should not happen in a valid miconic state, but handle defensively
             return float('inf') # Cannot proceed without knowing lift location

        current_lift_floor_num = self.floor_to_num[current_lift_floor_name]

        heuristic_cost = 0  # Initialize action cost counter.
        floors_to_visit_nums = set() # Floors the lift must visit (origins for waiting, destinations for boarded)

        # Identify unserved passengers and their required actions/stops
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        all_passengers = set(self.goal_locations.keys()) # Get all passengers from goal locations

        unserved_passengers = all_passengers - served_passengers

        for passenger in unserved_passengers:
            # Check if passenger is waiting at origin
            is_waiting = False
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    origin_floor_name = get_parts(fact)[2]
                    heuristic_cost += 1 # Cost for board action
                    floors_to_visit_nums.add(self.floor_to_num[origin_floor_name])
                    is_waiting = True
                    break # Found origin, move to next passenger state check

            # Check if passenger is boarded (and not served)
            is_boarded = False
            for fact in state:
                 if match(fact, "boarded", passenger):
                     # Passenger is boarded, needs to depart at destination
                     dest_floor_name = self.goal_locations[passenger]
                     heuristic_cost += 1 # Cost for depart action
                     floors_to_visit_nums.add(self.floor_to_num[dest_floor_name])
                     is_boarded = True
                     break # Found boarded state

            # A passenger should be either waiting or boarded if unserved
            # assert is_waiting or is_boarded, f"Unserved passenger {passenger} is neither waiting nor boarded in state: {state}"


        # If no floors need visiting, all relevant passengers are served or will be served
        # without further stops (e.g., already at destination and boarded, which is impossible due to effects)
        # or the state is somehow malformed. If floors_to_visit_nums is empty, the heuristic is just the action count.
        if not floors_to_visit_nums:
            return heuristic_cost

        # Estimate travel cost
        min_floor_num_to_visit = min(floors_to_visit_nums)
        max_floor_num_to_visit = max(floors_to_visit_nums)

        # Travel cost estimate: distance to the furthest required floor from current,
        # plus the distance to traverse the entire range of required floors.
        travel_cost = max(abs(current_lift_floor_num - min_floor_num_to_visit),
                          abs(current_lift_floor_num - max_floor_num_to_visit)) + \
                      (max_floor_num_to_visit - min_floor_num_to_visit)

        heuristic_cost += travel_cost

        return heuristic_cost

