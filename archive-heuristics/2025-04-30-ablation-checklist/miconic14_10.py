from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    passengers who are not yet served. It considers the lift's current
    position, the origin floors of unboarded passengers, and the destination
    floors of boarded passengers.

    # Assumptions
    - Floors are linearly ordered, defined by the `above` predicate.
    - Each unserved passenger needs to be boarded once and departed once.
    - The travel cost is estimated by the vertical distance between the
      minimum and maximum floor levels that need to be visited (including
      the current lift floor).

    # Heuristic Initialization
    - Parses the `above` facts to create a mapping from floor names to
      numerical levels.
    - Parses the `destin` facts to store the destination floor for each passenger.
    - Identifies the set of passengers that need to be served based on the goal.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Goal Passengers:** Determine which passengers must be `served`
        in the goal state.
    2.  **Find Current Lift Location:** Get the floor where the lift is currently
        located from the state.
    3.  **Identify Relevant Passengers:** Iterate through the goal passengers.
        If a passenger is not yet `served`:
        *   Check if they are `boarded`.
        *   Check if they are at their `origin` floor.
    4.  **Determine Required Stops:** Collect the set of floor levels that the
        lift *must* visit to serve the relevant passengers:
        *   The current lift floor.
        *   The origin floor for each unserved, unboarded passenger.
        *   The destination floor for each unserved, boarded passenger.
    5.  **Estimate Travel Cost:** Calculate the vertical distance the lift needs
        to cover. This is the difference between the maximum and minimum floor
        levels among the required stops. This estimates the minimum number of
        `up` or `down` actions needed to visit all necessary floors.
    6.  **Count Board/Depart Actions:**
        *   Count the number of unserved, unboarded passengers. Each needs a
            `board` action.
        *   Count the number of unserved, boarded passengers. Each needs a
            `depart` action.
    7.  **Sum Costs:** The heuristic value is the sum of the estimated travel
        cost, the number of `board` actions needed, and the number of `depart`
        actions needed. If all goal passengers are served, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering, passenger
        destinations, and goal passengers from the task.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Build floor ordering and mapping from floor name to level.
        # Find all floors mentioned in `above` facts.
        floors = set()
        above_map = {} # Maps floor -> floor_immediately_above
        below_map = {} # Maps floor -> floor_immediately_below

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:3]
                floors.add(f_above)
                floors.add(f_below)
                above_map[f_below] = f_above
                below_map[f_above] = f_below

        # Find the lowest floor (a floor that is not immediately above any other floor).
        # Or more robustly, a floor that is not the *second* argument of any `above` fact.
        all_floors_below = set(below_map.values())
        lowest_floor = None
        for floor in floors:
             if floor not in all_floors_below:
                 lowest_floor = floor
                 break

        if lowest_floor is None and floors:
             # Handle cases where there might be a cycle or no clear bottom (unlikely in miconic)
             # Or if there's only one floor. Assume the first floor found is the lowest if no 'above' facts define a bottom.
             lowest_floor = next(iter(floors)) if floors else None


        self.floor_to_level = {}
        if lowest_floor:
            current_floor = lowest_floor
            level = 1
            while current_floor:
                self.floor_to_level[current_floor] = level
                current_floor = above_map.get(current_floor)
                level += 1

        # 2. Store destination floor for each passenger.
        self.passenger_to_destin_floor = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                passenger, floor = get_parts(fact)[1:3]
                self.passenger_to_destin_floor[passenger] = floor

        # 3. Identify goal passengers.
        self.goal_passengers = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                passenger = get_parts(goal)[1]
                self.goal_passengers.add(passenger)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is already reached
        if self.goals <= state:
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # This should not happen in a valid miconic state, but handle defensively
             return float('inf') # Cannot proceed without lift location

        current_lift_level = self.floor_to_level.get(current_lift_floor, 0) # Default to 0 if floor not mapped

        required_floors_levels = {current_lift_level}
        num_board_needed = 0
        num_depart_needed = 0

        # Track passenger states for quick lookup
        is_served = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        is_boarded = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}


        for passenger in self.goal_passengers:
            if passenger in is_served:
                continue # This passenger is already served

            if passenger in is_boarded:
                # Passenger is boarded but not served, needs to depart at destination
                num_depart_needed += 1
                destin_floor = self.passenger_to_destin_floor.get(passenger)
                if destin_floor:
                    destin_level = self.floor_to_level.get(destin_floor, 0)
                    required_floors_levels.add(destin_level)
                # else: defensive, should not happen if problem is well-formed

            elif passenger in origin_locations:
                 # Passenger is at origin but not boarded and not served, needs to be boarded
                 num_board_needed += 1
                 origin_floor = origin_locations[passenger]
                 origin_level = self.floor_to_level.get(origin_floor, 0)
                 required_floors_levels.add(origin_level)
            # else: Passenger is not served, not boarded, and not at origin.
            # This state shouldn't be reachable from initial state via valid actions
            # unless they were previously boarded and then dropped off somewhere not their destin.
            # We assume valid states where unserved passengers are either at origin or boarded.


        # If no passengers need serving, the heuristic is 0 (already checked goal state)
        # or if no relevant floors were found other than the current lift floor.
        if len(required_floors_levels) == 1 and current_lift_level in required_floors_levels and num_board_needed == 0 and num_depart_needed == 0:
             return 0


        # Calculate travel distance
        min_level = min(required_floors_levels)
        max_level = max(required_floors_levels)
        travel_distance = max_level - min_level

        # Total heuristic estimate
        total_cost = travel_distance + num_board_needed + num_depart_needed

        return total_cost

