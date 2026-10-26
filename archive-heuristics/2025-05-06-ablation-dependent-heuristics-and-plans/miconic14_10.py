# Assuming heuristic_base.py exists and defines a Heuristic base class
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the required board and depart actions for each unserved passenger
    and adds an estimate of the lift movement cost needed to visit all relevant floors
    (origins of unboarded passengers and destinations of boarded passengers).

    # Assumptions
    - Passengers need to be picked up at their origin and dropped off at their destination.
    - The lift can carry multiple passengers.
    - The 'above' predicate defines a total order on floors, and floor names like 'fN'
      correspond to this order (f1 < f2 < ...).

    # Heuristic Initialization
    - Extracts the origin and destination floors for each passenger from static facts.
    - Identifies all floor names and creates a mapping from floor name to its numerical index
      based on the sorted order of floors.
    - Stores the list of all passengers that need to be served according to the goal.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Identify the current floor of the lift and its corresponding numerical index.
    3. Initialize sets `floors_to_visit` to store floors the lift must visit.
    4. Identify the set of passengers that are currently served and currently boarded.
    5. Iterate through all passengers that need to be served (identified during initialization from goals).
    6. For each unserved passenger `p`:
       - Add 1 to `h` for the eventual `depart` action.
       - Check if `p` is currently `boarded`.
       - If `p` is `boarded`, add `p`'s destination floor to `floors_to_visit`.
       - If `p` is not `boarded`, add 1 to `h` for the eventual `board` action, and add `p`'s origin floor to `floors_to_visit`.
    7. If the current lift floor is known and there are floors in `floors_to_visit`:
       - Find the minimum and maximum floor indices among the floors in `floors_to_visit`.
       - Calculate the estimated movement cost: `min(abs(current_floor_index - min_needed_index), abs(current_floor_index - max_needed_index)) + (max_needed_index - min_needed_index)`. This estimates the minimum travel distance to cover the range of needed floors starting from the current floor.
       - Add the calculated movement cost to `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about
        passenger origins/destinations and floor ordering.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destins = {}
        all_floor_names = set()
        self.passengers_to_serve = set() # Store passengers mentioned in goals

        # Extract passengers to serve from goals
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served" and len(parts) == 2:
                 self.passengers_to_serve.add(parts[1])

        # Extract static information: origins, destinations, and floor names
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "origin" and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.passenger_origins[p] = f
                all_floor_names.add(f)
            elif predicate == "destin" and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.passenger_destins[p] = f
                all_floor_names.add(f)
            elif predicate == "above" and len(parts) == 3:
                 f1, f2 = parts[1], parts[2]
                 all_floor_names.add(f1)
                 all_floor_names.add(f2)

        # Create a sorted list of floor names and a mapping to indices
        # Assumes floor names are like 'f1', 'f2', etc., and the number indicates order.
        # This assumption is based on the structure of example instance 2.
        sorted_floor_names = sorted(list(all_floor_names), key=lambda f: int(f[1:]))
        self.floor_name_to_index = {f: i + 1 for i, f in enumerate(sorted_floor_names)}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        h = 0 # Initialize heuristic value

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        # If lift location is unknown or not in our floor mapping (shouldn't happen in valid states),
        # fallback to just counting board/depart actions needed.
        if current_lift_floor is None or current_lift_floor not in self.floor_name_to_index:
             served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == "served"}
             boarded_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == "boarded"}

             for p in self.passengers_to_serve:
                 if p not in served_passengers:
                     h += 1 # for depart
                     if p not in boarded_passengers:
                         h += 1 # for board
             return h

        current_lift_floor_idx = self.floor_name_to_index[current_lift_floor]

        # Sets to track floors the lift must visit
        floors_to_visit = set()

        # Check each passenger's status
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == "served"}
        boarded_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == "boarded"}

        # Only consider passengers that need to be served according to the goal
        unserved_passengers = [p for p in self.passengers_to_serve if p not in served_passengers]

        for p in unserved_passengers:
            # Each unserved passenger needs a 'depart' action
            h += 1

            if p in boarded_passengers:
                # Passenger is boarded, needs to go to destination
                dest_floor = self.passenger_destins.get(p)
                if dest_floor: # Ensure destination is known
                    floors_to_visit.add(dest_floor)
            else:
                # Passenger is not boarded, needs 'board' action and trip from origin to destination
                h += 1 # Cost for 'board' action
                origin_floor = self.passenger_origins.get(p)
                if origin_floor: # Ensure origin is known
                    floors_to_visit.add(origin_floor)


        # Calculate movement cost if there are floors to visit
        if floors_to_visit:
            # Filter out any floors not in our mapping (shouldn't happen if init is correct)
            needed_indices = [self.floor_name_to_index[f] for f in floors_to_visit if f in self.floor_name_to_index]

            if needed_indices: # Ensure needed_indices is not empty after filtering
                min_needed_idx = min(needed_indices)
                max_needed_idx = max(needed_indices)

                # Movement cost calculation
                # min distance to reach either end of the needed range, plus the range span
                movement_cost = min(abs(current_lift_floor_idx - min_needed_idx), abs(current_lift_floor_idx - max_needed_idx)) + (max_needed_idx - min_needed_idx)

                h += movement_cost

        return h
