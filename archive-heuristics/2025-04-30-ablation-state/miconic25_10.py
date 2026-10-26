from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the required board/depart actions for each unserved passenger and adds
    an estimate of the minimum lift movement cost to visit all necessary floors
    (origins of waiting passengers and destinations of boarded passengers).

    # Assumptions
    - Floors are ordered linearly based on the `above` predicates.
    - Each unserved passenger requires at least one board and one depart action (if waiting)
      or one depart action (if boarded).
    - Lift movement cost is estimated by the distance needed to travel from the current
      floor to the lowest required floor, then sweeping up to the highest required floor,
      or vice versa.

    # Heuristic Initialization
    - Parses the `above` predicates from static facts to determine the floor order
      and create a mapping from floor names to integer levels. Floors not part of
      the main chain or only mentioned in other predicates are assigned level 0.
    - Stores the destination floor for each passenger from the `destin` predicates
      found in the static facts.
    - Stores the set of passengers who need to be served according to the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Current State Information:**
        -   Find the current floor of the lift by looking for the `(lift-at ?f)` fact in the state. If not found, return infinity (or a large value), unless the goal is already reached.
        -   Identify all passengers currently in the state who are *not* marked as `(served ?p)`. These are the unserved passengers we need to account for. Filter this list to include only those passengers whose `(served ?p)` is a goal condition.
        -   For each unserved goal passenger, determine their current status:
            -   Are they waiting at their origin floor? (Check for `(origin ?p ?f_origin)` fact).
            -   Are they boarded in the lift? (Check for `(boarded ?p)` fact).
    2.  **Calculate Action Cost Component:**
        -   Initialize a variable `action_cost` to 0.
        -   For each unserved goal passenger who is waiting at their origin: Add 2 to `action_cost`. This accounts for the necessary `board` action and the eventual `depart` action.
        -   For each unserved goal passenger who is boarded: Add 1 to `action_cost`. This accounts for the necessary `depart` action.
    3.  **Identify Required Floors for Lift Travel:**
        -   Create a set `required_floors`.
        -   For each unserved goal passenger who is waiting at their origin, add their origin floor (`f_origin`) to `required_floors`. The lift must visit this floor to pick them up.
        -   For each unserved goal passenger who is boarded, add their destination floor (`f_destin`) to `required_floors`. The lift must visit this floor to drop them off.
    4.  **Calculate Lift Movement Cost Component:**
        -   Initialize a variable `movement_cost` to 0.
        -   If `required_floors` is empty (meaning all relevant goal passengers are served or in a state not requiring a specific floor visit), the `movement_cost` remains 0.
        -   If `required_floors` is not empty:
            -   Get the integer level for the current lift floor using the pre-calculated `floor_levels` map. Use level 0 as a default if the floor is unexpectedly not found.
            -   Find the minimum and maximum integer levels among all floors in `required_floors` using the `floor_levels` map. Use level 0 as a default if a floor is unexpectedly not found.
            -   Estimate the minimum travel distance required to visit all floors between the minimum and maximum required levels, starting from the current lift level. This is calculated as `min(abs(current_level - min_req_level), abs(current_level - max_req_level)) + (max_req_level - min_req_level)`. This represents the cost of traveling to the nearest extreme required floor and then sweeping across the range of required floors.
            -   Assign this calculated value to `movement_cost`.
    5.  **Combine Costs:**
        -   The total heuristic value is the sum of `action_cost` and `movement_cost`.
    6.  **Return Heuristic Value:**
        -   Return the calculated total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals # Goal conditions (e.g., (served p1), (served p2), ...)
        static_facts = task.static # Facts that are not affected by actions.

        # Parse floor order and create floor_levels map
        self.floor_levels = self._parse_floor_order(static_facts)

        # Store destination floor for each passenger
        self.passenger_destinations = {}
        for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 _, passenger, floor = get_parts(fact)
                 self.passenger_destinations[passenger] = floor

        # Store the set of passengers who need to be served according to the goal
        self.goal_served_passengers = {
            get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")
        }


    def _parse_floor_order(self, static_facts):
        """
        Parses the (above f1 f2) facts to determine the linear order of floors
        and assigns an integer level to each floor.
        Returns a dictionary mapping floor name to level (integer).
        """
        above_map = {} # Maps floor -> floor_above
        below_map = {} # Maps floor -> floor_below
        all_floors_mentioned = set()

        # Collect all floor names mentioned in relevant predicates
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "above":
                if len(parts) == 3:
                    f_above, f_below = parts[1:]
                    above_map[f_below] = f_above
                    below_map[f_above] = f_below
                    all_floors_mentioned.add(f_above)
                    all_floors_mentioned.add(f_below)
            elif predicate in ["lift-at", "origin", "destin"]:
                 if len(parts) >= 2: # lift-at has 2 parts, origin/destin have 3
                     floor = parts[-1] # Floor is the last argument
                     all_floors_mentioned.add(floor)

        if not all_floors_mentioned:
             # No floors mentioned at all? Empty problem?
             return {}

        # Find potential bottom floors: floors that are mentioned but nothing is below them
        # A floor 'f' is a bottom candidate if it's in all_floors_mentioned
        # AND it is NOT a key in below_map (i.e., no floor is immediately below it).
        potential_bottoms = all_floors_mentioned - set(below_map.keys())

        bottom_floor = None
        if len(potential_bottoms) == 1:
            bottom_floor = list(potential_bottoms)[0]
        elif len(potential_bottoms) > 1:
            # Multiple potential bottoms - implies disconnected floor structures or error.
            # Sort alphabetically and pick the first as a deterministic fallback.
            sorted_potential_bottoms = sorted(list(potential_bottoms))
            bottom_floor = sorted_potential_bottoms[0]
            # print(f"Warning: Multiple potential bottom floors found: {potential_bottoms}. Assuming '{bottom_floor}' is the lowest.")
        elif not potential_bottoms and all_floors_mentioned:
             # No potential bottom found, but there are floors. This could indicate a cycle
             # in 'above' facts or a single floor mentioned only in 'above' as f_above.
             # If there's only one floor total, it's the bottom.
             if len(all_floors_mentioned) == 1:
                 bottom_floor = list(all_floors_mentioned)[0]
             else:
                 # Still can't find a bottom. Fallback: assign level 0 to all.
                 # print("Warning: Could not determine unique bottom floor. Assigning level 0 to all floors.")
                 return {floor: 0 for floor in all_floors_mentioned}


        # Build the ordered list of floors starting from the bottom
        ordered_floors = []
        current_floor = bottom_floor
        seen_floors = set() # Prevent infinite loops in case of cycles (though PDDL above should be acyclic)
        while current_floor is not None and current_floor not in seen_floors:
            ordered_floors.append(current_floor)
            seen_floors.add(current_floor)
            current_floor = above_map.get(current_floor) # Get the floor immediately above

        # Create the floor_levels map
        floor_levels = {floor: level for level, floor in enumerate(ordered_floors)}

        # Assign level 0 to any floors mentioned but not part of the main ordered chain
        # This handles single floors or disconnected structures.
        for floor in all_floors_mentioned:
            if floor not in floor_levels:
                floor_levels[floor] = 0
                # print(f"Warning: Floor '{floor}' not part of main floor chain. Assigning level 0.")


        return floor_levels


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is already reached
        if self.goals <= state:
             return 0

        # 1. Identify the current floor of the lift.
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is not found and goal is not reached, this is an invalid state for this heuristic.
        if current_lift_floor is None:
             return float('inf') # Return infinity or a large value


        # 2. Identify unserved goal passengers and their state (waiting or boarded).
        waiting_passengers_info = {} # {p: (f_origin, f_destin)}
        boarded_passengers_info = {} # {p: f_destin}
        served_passengers_in_state = set()

        # First, find all served passengers in the current state
        for fact in state:
            if match(fact, "served", "*"):
                served_passengers_in_state.add(get_parts(fact)[1])

        # Identify unserved goal passengers
        unserved_goal_passengers = self.goal_served_passengers - served_passengers_in_state

        # If no unserved goal passengers, but goal check at start failed, something is wrong.
        # This shouldn't happen if goal_served_passengers is correctly derived from task.goals
        # and the initial goal check is correct. But let's be safe.
        if not unserved_goal_passengers:
             # This implies all goal passengers are served, but the initial goal check failed.
             # This shouldn't be reachable if the initial check is correct.
             # Return 0 as a fallback, assuming the goal check is the primary authority.
             return 0


        # Now, find the state of these unserved goal passengers
        # We need to iterate through the state again or pre-process state facts.
        # Let's pre-process state facts into lookups for efficiency.
        state_facts_dict = {}
        for fact in state:
             parts = get_parts(fact)
             predicate = parts[0]
             if predicate == "origin" and len(parts) == 3:
                 p, f = parts[1:]
                 state_facts_dict[("origin", p)] = f
             elif predicate == "boarded" and len(parts) == 2:
                 p = parts[1]
                 state_facts_dict[("boarded", p)] = True # Just mark presence
             # served facts already processed

        for p in unserved_goal_passengers:
             if ("origin", p) in state_facts_dict:
                 f_origin = state_facts_dict[("origin", p)]
                 # Need destination, which is stored in self.passenger_destinations
                 if p in self.passenger_destinations:
                     f_destin = self.passenger_destinations[p]
                     waiting_passengers_info[p] = (f_origin, f_destin)
             elif ("boarded", p) in state_facts_dict:
                 # Need destination
                 if p in self.passenger_destinations:
                     f_destin = self.passenger_destinations[p]
                     boarded_passengers_info[p] = f_destin
             # Passengers who are neither waiting nor boarded but unserved are not handled by this heuristic.
             # This assumes unserved passengers are always either waiting or boarded.


        total_heuristic = 0

        # 3. Calculate action cost component.
        total_heuristic += 2 * len(waiting_passengers_info) # board + depart for each waiting
        total_heuristic += 1 * len(boarded_passengers_info) # depart for each boarded

        # 4. Identify required floors for lift travel.
        required_floors = set()
        for f_origin, f_destin in waiting_passengers_info.values():
            required_floors.add(f_origin)
            required_floors.add(f_destin)
        for f_destin in boarded_passengers_info.values():
            required_floors.add(f_destin)

        # 5. Calculate lift movement cost component.
        movement_cost = 0
        if required_floors:
            current_level = self.floor_levels.get(current_lift_floor, 0) # Default 0 if floor not found
            required_levels = [self.floor_levels.get(f, 0) for f in required_floors] # Default 0 if floor not found

            min_req_level = min(required_levels)
            max_req_level = max(required_levels)

            # Movement cost estimate: travel to nearest extreme, then sweep to the other.
            movement_cost = min(abs(current_level - min_req_level), abs(current_level - max_req_level)) + (max_req_level - min_req_level)

        # 6. Combine costs (already done in step 3).
        total_heuristic += movement_cost

        # 7. Return total.
        return total_heuristic
