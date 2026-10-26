import math
from fnmatch import fnmatch

# Try to import the base class Heuristic from the expected location.
# If it's not found (e.g., when running standalone or in a different environment),
# define a dummy base class to allow the code to be parsed and potentially tested.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic base class if the actual one cannot be imported.
    class Heuristic:
        def __init__(self, task):
            """Dummy initializer."""
            pass
        def __call__(self, node):
            """Dummy call method."""
            raise NotImplementedError("Heuristic base class not found.")

def get_parts(fact):
    """Extracts the predicate and arguments from a PDDL fact string.
    Removes the surrounding parentheses and splits the string by spaces.
    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    """
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a given PDDL fact string matches a specified pattern.
    The pattern is provided as arguments (*args), where '*' can be used as a wildcard.
    Uses fnmatch for wildcard matching capability.
    Example: match("(at ball1 roomA)", "at", "*", "roomA") -> True
             match("(at ball1 roomA)", "at", "ball?", "room*") -> True
             match("(at ball1 roomA)", "on", "*", "*") -> False
    """
    parts = get_parts(fact)
    # The number of parts in the fact must match the number of pattern arguments.
    if len(parts) != len(args):
        return False
    # Check if each part of the fact matches the corresponding pattern argument.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic function for the PDDL domain 'miconic', which models
    an elevator transporting passengers between floors.

    # Summary
    This heuristic estimates the minimum number of actions required to reach a goal state
    where all specified passengers have been served (i.e., transported to their
    destination floors). The estimate is calculated as the sum of two components:
    1. Action Cost: The total count of 'board' and 'depart' actions required for
       all passengers who are not yet served. Each unserved passenger waiting at
       their origin requires one 'board' and one 'depart' (cost 2). Each unserved
       passenger already boarded requires one 'depart' (cost 1).
    2. Lift Movement Cost: An estimate of the minimum number of 'up'/'down' actions
       needed for the lift to travel between the required floors. This is estimated
       as the vertical distance (number of floors) between the highest and lowest
       floors that the lift needs to visit. These floors include the lift's current
       location, the origin floors of waiting passengers, and the destination floors
       of all unserved passengers (both waiting and boarded).

    # Assumptions
    - Floors are arranged in a linear sequence (e.g., f1, f2, f3, ...).
    - Floor names allow for sorting to determine this linear order. The heuristic
      attempts to sort numerically based on the number in the name (e.g., 'f1', 'f10')
      and falls back to alphabetical sorting if numerical parsing fails.
    - The cost of moving the lift between adjacent floors ('up' or 'down') is 1.
    - The cost of 'board' and 'depart' actions is 1 each.
    - The PDDL task and states provided are consistent with the domain definition.
      Specifically, an unserved passenger is expected to be either at their origin
      location (represented by an `(origin p f)` fact) or currently inside the
      lift (represented by a `(boarded p)` fact).

    # Heuristic Initialization (`__init__`)
    - The constructor receives the planning task object (`task`).
    - It extracts the set of all possible passenger objects and floor objects by
      scanning all potential facts defined in `task.facts`.
    - It parses the static facts (`task.static`) to build a dictionary `self.destin`
      mapping each passenger to their destination floor.
    - It determines the linear order of floors by sorting the extracted floor names.
      Based on this order, it creates `self.floor_index`, a dictionary mapping each
      floor name to a unique integer index (starting from 0 for the lowest floor).
    - It identifies the set of passengers that must be served to satisfy the goal
      conditions (`self.passenger_goals`).

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Receive the current state node (`node`). Extract the state's facts (`state`).
    2. Determine which passengers required by the goal are already served by checking
       for `(served p)` facts in the state.
    3. If all goal passengers are served, the goal is reached, return 0.
    4. Find the lift's current floor from the `(lift-at ?f)` fact. If the lift's
       location is unknown or the floor is invalid, return infinity (indicating an
       unreachable or invalid state).
    5. Initialize `h_actions = 0` to accumulate the cost of 'board'/'depart' actions.
    6. Create a set `required_floors_indices` and add the index of the lift's current floor.
       This set will store the indices of all floors the lift must potentially visit.
    7. Identify passengers currently boarded (`(boarded p)` facts).
    8. Identify the origin floors for passengers who are waiting (unserved, not boarded,
       and have an `(origin p f)` fact present in the state). Store this in `passenger_origins`.
    9. Iterate through each passenger `p` that needs to be served (`active_passengers`):
       a. Get the destination floor `d` for `p` and its index `dest_floor_idx`. Handle cases
          where the destination is missing or invalid.
       b. If `p` is currently boarded: Add 1 to `h_actions`. Add `dest_floor_idx` to `required_floors_indices`.
       c. If `p` is waiting at an origin floor `o`: Add 2 to `h_actions`. Add the index of `o`
          and `dest_floor_idx` to `required_floors_indices`.
       d. If `p` is unserved but neither boarded nor at an origin (unexpected state): Print a
          warning and potentially add a penalty or estimate cost based on destination.
    10. Calculate the lift movement estimate `h_lift`:
        a. If `required_floors_indices` contains 1 or fewer unique indices (meaning the lift
           doesn't need to move or only needs to service passengers at its current location),
           set `h_lift = 0`.
        b. Otherwise, find the minimum (`min_idx`) and maximum (`max_idx`) floor indices in the set.
        c. Calculate `h_lift = max_idx - min_idx`. This represents the minimum vertical span
           the lift must cover.
    11. The final heuristic value is the sum `h_actions + h_lift`.
    """

    def __init__(self, task):
        """
        Initializes the Miconic heuristic.
        - Extracts passengers, floors, destinations, and goal passengers.
        - Establishes floor ordering and indices.
        """
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static
        all_facts = task.facts # Set of all possible ground atoms in the domain

        # Extract passenger and floor objects by scanning all possible facts
        self.passengers = set()
        self.floors = set()
        for fact in all_facts:
            parts = get_parts(fact)
            pred = parts[0]
            # Identify passengers (usually the first argument in relevant predicates)
            if pred in ["origin", "destin", "boarded", "served"] and len(parts) > 1:
                self.passengers.add(parts[1])
            # Identify floors (appear as arguments in various predicates)
            if pred == "above":
                if len(parts) > 1: self.floors.add(parts[1])
                if len(parts) > 2: self.floors.add(parts[2])
            elif pred == "lift-at" and len(parts) > 1:
                self.floors.add(parts[1])
            elif pred in ["origin", "destin"] and len(parts) > 2:
                # Floor is usually the second argument for origin/destin
                self.floors.add(parts[2])

        # Store passenger destinations from static facts
        self.destin = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, p, f = get_parts(fact)
                # Ensure the passenger and floor are recognized objects before storing
                if p in self.passengers and f in self.floors:
                    self.destin[p] = f

        # Determine floor order and assign indices (0 = lowest floor)
        self.floor_index = {}
        ordered_floors = []
        if self.floors:
            try:
                # Attempt to sort floors numerically based on their names (e.g., f1, f2, f10)
                ordered_floors = sorted(list(self.floors), key=lambda x: int(x[1:]))
            except (ValueError, IndexError):
                # If numerical sort fails (e.g., names are not f<number>), fallback to alphabetical sort
                print(f"Info: Floor names may not be in 'f<number>' format. Using alphabetical sort for floor order.")
                ordered_floors = sorted(list(self.floors))

            if ordered_floors:
                # Assign indices based on the sorted order (0 for the first/lowest floor)
                self.floor_index = {floor: i for i, floor in enumerate(ordered_floors)}
            else:
                 print("Warning: Could not establish floor order.")
        else:
            print("Warning: No floor objects found during initialization.")

        # Raise an error if floor indices could not be determined, as they are crucial
        if not self.floor_index and self.floors:
             raise ValueError("Failed to initialize floor indices. Heuristic cannot operate.")

        # Store the set of passengers that are required to be served by the goal conditions
        self.passenger_goals = {get_parts(g)[1] for g in self.goals if match(g, "served", "*")}


    def get_lift_floor(self, state):
        """Finds the current floor of the lift from the state facts."""
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) > 1:
                    # Return the floor name (second part of the fact)
                    return parts[1]
        # This should ideally not happen in a valid state for this domain
        print("Warning: Lift location ('lift-at') predicate not found in the current state.")
        return None

    def __call__(self, node):
        """
        Calculates the heuristic value (estimated cost to goal) for the state
        represented by the given search node.
        """
        state = node.state
        h_actions = 0  # Accumulator for board/depart action costs

        # Identify which goal passengers are currently served
        current_served = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # If all passengers required by the goal are served, we are at the goal state.
        if self.passenger_goals.issubset(current_served):
            return 0

        # Find the lift's current floor
        lift_floor = self.get_lift_floor(state)
        # If lift location is unknown or the floor name isn't in our index, state is invalid/unreachable.
        if lift_floor is None or lift_floor not in self.floor_index:
             return float('inf')

        # Get the index of the current lift floor
        current_lift_idx = self.floor_index[lift_floor]
        # Initialize the set of required floor indices with the current lift floor's index
        required_floors_indices = {current_lift_idx}

        # Identify which passengers are currently boarded
        passengers_boarded = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        # Identify the origin floors of passengers who are waiting (unserved and not boarded)
        passenger_origins = {}
        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, p, f = get_parts(fact)
                # Consider only passengers relevant to the goal and not yet served
                if p in self.passenger_goals and p not in current_served:
                     # Ensure the origin floor is valid before storing
                     if f in self.floor_index:
                         passenger_origins[p] = f
                     else:
                         # Log a warning if an origin floor is unrecognized
                         print(f"Warning: Origin floor '{f}' for passenger '{p}' not found in floor index.")


        # Calculate costs for passengers who still need to be served
        active_passengers = self.passenger_goals - current_served
        for p in active_passengers:
            dest_floor = self.destin.get(p)
            # Check if the destination floor is valid and known
            if dest_floor is None or dest_floor not in self.floor_index:
                 print(f"Warning: Destination floor for passenger '{p}' is invalid or missing in static info.")
                 # Add a penalty as some actions will be needed, but exact cost is unclear
                 h_actions += 2 # Arbitrary penalty (e.g., assume board+depart needed)
                 continue

            dest_floor_idx = self.floor_index[dest_floor]

            if p in passengers_boarded:
                # Passenger is boarded: needs 1 'depart' action.
                h_actions += 1
                # Add destination floor index to required floors
                required_floors_indices.add(dest_floor_idx)
            elif p in passenger_origins:
                # Passenger is waiting at origin: needs 'board' + 'depart' (2 actions).
                origin_floor = passenger_origins[p]
                origin_floor_idx = self.floor_index[origin_floor]
                h_actions += 2
                # Add both origin and destination floor indices to required floors
                required_floors_indices.add(origin_floor_idx)
                required_floors_indices.add(dest_floor_idx)
            else:
                # Passenger 'p' is unserved, not boarded, and not at origin.
                # This state configuration should not occur if the domain model and
                # state transitions are correct. It implies the passenger is "lost".
                print(f"Warning: Inconsistent state detected for passenger '{p}'. "
                      f"Not served, not boarded, and not at origin.")
                # Apply a penalty or estimate? Let's estimate based on destination.
                h_actions += 1 # Assume at least 'depart' is needed if they somehow reach destination
                required_floors_indices.add(dest_floor_idx)


        # Calculate lift movement estimate based on the range of required floors
        h_lift = 0
        # Movement is only needed if there's more than one required floor index
        if len(required_floors_indices) > 1:
            min_idx = min(required_floors_indices)
            max_idx = max(required_floors_indices)
            # Estimate movement cost as the difference between highest and lowest required floor indices
            h_lift = max_idx - min_idx

        # The final heuristic value is the sum of action counts and estimated lift travel distance
        return h_actions + h_lift
