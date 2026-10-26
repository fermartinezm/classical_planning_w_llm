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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort to serve all passengers.
    It counts the number of board and depart actions still needed for unserved
    passengers and adds an estimate of the lift movement required to reach
    the necessary floors.

    # Assumptions
    - Each unserved passenger needs a 'board' action (unless already boarded)
      and a 'depart' action.
    - The lift must visit the origin floor for waiting passengers and the
      destination floor for boarded passengers.
    - The cost of movement is estimated by the distance required to travel
      from the current floor to the range of floors that need visiting,
      plus the distance to traverse that range.

    # Heuristic Initialization
    - Parses static facts to determine the floor order and create a mapping
      from floor names to numerical indices.
    - Stores passenger origin and destination floors from static facts.
    - Stores the set of all passengers.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Current Lift Location:** Find the floor where the lift is currently located.
    2.  **Identify Unserved Passengers:** Determine which passengers have not yet reached their destination (i.e., the `(served ?p)` predicate is not true for them).
    3.  **Count Required Actions:**
        -   For each unserved passenger `p`:
            -   If `p` is currently waiting at their origin floor (`(origin p f)` is true in the state), they need both a 'board' and a 'depart' action. Add 2 to the action count.
            -   If `p` is currently boarded (`(boarded p)` is true in the state), they only need a 'depart' action. Add 1 to the action count.
    4.  **Identify Floors to Visit:** Determine the set of floors the lift *must* visit to perform the required actions:
        -   Include the origin floor for every unserved passenger who is currently waiting.
        -   Include the destination floor for every unserved passenger who is currently boarded.
    5.  **Estimate Movement Cost:**
        -   If there are no floors to visit, the movement cost is 0.
        -   If there are floors to visit, find the minimum and maximum floor indices among these floors.
        -   Calculate the distance from the current lift floor index to the minimum relevant floor index and to the maximum relevant floor index.
        -   The movement cost is the minimum of these two distances plus the total distance between the minimum and maximum relevant floors (the range width). This estimates the cost to reach the "cluster" of relevant floors and traverse it.
    6.  **Calculate Total Heuristic:** Sum the required action counts and the estimated movement cost.
    7.  **Goal State Check:** If there are no unserved passengers, the state is a goal state, and the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger origins,
        and destinations from static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build floor order and index mapping from 'above' predicates
        self.floor_order = []
        above_map = {} # floor_lower -> floor_higher
        below_map = {} # floor_higher -> floor_lower
        all_floors = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_lower, f_higher = get_parts(fact)
                above_map[f_lower] = f_higher
                below_map[f_higher] = f_lower
                all_floors.add(f_lower)
                all_floors.add(f_higher)

        # Find the bottom floor (a floor that is not the higher floor in any 'above' fact)
        bottom_floor = None
        for floor in all_floors:
            if floor not in below_map:
                bottom_floor = floor
                break

        # Build the ordered list of floors starting from the bottom
        current = bottom_floor
        while current is not None:
            self.floor_order.append(current)
            current = above_map.get(current)

        # Create a mapping from floor name to its index (0-based)
        self.floor_to_index = {floor: i for i, floor in enumerate(self.floor_order)}

        # Store passenger origins and destinations
        self.passenger_origins = {} # passenger -> origin_floor
        self.passenger_destins = {} # passenger -> destin_floor
        self.all_passengers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'origin':
                p, f = parts[1:]
                self.passenger_origins[p] = f
                self.all_passengers.add(p)
            elif parts[0] == 'destin':
                p, f = parts[1:]
                self.passenger_destins[p] = f
                self.all_passengers.add(p) # Ensure passengers with only destin in static are included

        # In some domains, all passengers might be listed in objects but not init/static origin/destin
        # We assume passengers are defined by origin/destin facts in static/init
        # If task.facts contained passenger types, we could iterate through objects of type passenger

    def get_floor_index(self, floor_name):
        """Returns the numerical index for a given floor name."""
        return self.floor_to_index.get(floor_name, -1) # Should not happen in valid problems

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # This should not happen in a valid miconic state, but handle defensively
             return float('inf') # Or a large value indicating an invalid state

        current_floor_index = self.get_floor_index(current_lift_floor)

        # Identify served, boarded, and waiting passengers in the current state
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        
        # Identify passengers waiting at their origin floor in the current state
        waiting_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "origin", "*", "*")}

        # Identify unserved passengers
        unserved_passengers = self.all_passengers - served_passengers

        # If all passengers are served, the heuristic is 0
        if not unserved_passengers:
            return 0

        # --- Calculate Action Costs ---
        # Count board actions needed: for unserved passengers who are waiting
        board_actions_needed = len(waiting_passengers_in_state.intersection(unserved_passengers))

        # Count depart actions needed: for unserved passengers who are boarded
        depart_actions_needed = len(boarded_passengers.intersection(unserved_passengers))

        # --- Identify Floors to Visit ---
        floors_to_visit = set()

        # Add origin floors for unserved passengers who are waiting
        for p in waiting_passengers_in_state.intersection(unserved_passengers):
             floors_to_visit.add(waiting_passengers_in_state[p]) # Get origin floor from state

        # Add destination floors for unserved passengers who are boarded
        for p in boarded_passengers.intersection(unserved_passengers):
             # Get destination floor from static facts (destin predicate)
             if p in self.passenger_destins:
                 floors_to_visit.add(self.passenger_destins[p])

        # --- Estimate Movement Cost ---
        movement_cost = 0
        if floors_to_visit:
            relevant_indices = sorted([self.get_floor_index(f) for f in floors_to_visit])
            min_relevant_index = relevant_indices[0]
            max_relevant_index = relevant_indices[-1]

            # Cost to reach the range of relevant floors + cost to traverse the range
            dist_to_min = abs(current_floor_index - min_relevant_index)
            dist_to_max = abs(current_floor_index - max_relevant_index)
            range_width = max_relevant_index - min_relevant_index

            movement_cost = min(dist_to_min, dist_to_max) + range_width

        # --- Total Heuristic Value ---
        # Each waiting unserved passenger needs board (1) + depart (1) = 2 actions
        # Each boarded unserved passenger needs depart (1) = 1 action
        # Note: A passenger is either waiting OR boarded OR served.
        # So, the sum of waiting unserved and boarded unserved is the total number of unserved passengers.
        # Let's count actions per passenger state:
        actions_cost = 0
        for p in unserved_passengers:
            if p in waiting_passengers_in_state:
                actions_cost += 2 # Needs board and depart
            elif p in boarded_passengers:
                actions_cost += 1 # Needs depart
            # else: passenger is unserved but not waiting or boarded? (e.g., just got off at wrong floor?)
            # The domain doesn't seem to allow this state. Assume unserved means waiting or boarded.

        total_cost = actions_cost + movement_cost

        return total_cost

