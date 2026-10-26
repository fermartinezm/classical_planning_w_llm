import sys
from fnmatch import fnmatch

# Try to import the Heuristic base class from the expected location.
# If running standalone or in a different environment, provide a dummy class.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Could not import Heuristic from heuristics.heuristic_base. Using dummy base class.", file=sys.stderr)
    class Heuristic:
        """Dummy base class for Heuristic if the standard import fails."""
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not implemented.")

# Helper function to parse PDDL fact strings
def get_parts(fact: str) -> list[str]:
    """Extract the components of a PDDL fact string (e.g., "(pred obj1 obj2)")."""
    # Removes the surrounding parentheses and splits the string by spaces.
    return fact[1:-1].split()

# Helper function to match a fact against a pattern with wildcards
def match(fact: str, *args: str) -> bool:
    """
    Check if a PDDL fact matches a given pattern using fnmatch for wildcard support.

    Args:
        fact: The fact string (e.g., "(at obj loc)").
        *args: The pattern elements (e.g., "at", "*", "loc*"). '*' allows wildcard matching.

    Returns:
        True if the fact matches the pattern (both predicate and arguments), False otherwise.
    """
    parts = get_parts(fact)
    # The number of parts in the fact must match the number of elements in the pattern.
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern element using fnmatch.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'miconic' (elevator control).

    # Summary
    This heuristic estimates the remaining cost (number of actions) to reach a goal state
    where all specified passengers have been served (transported to their destination).
    It calculates the cost by summing two components:
    1.  The number of 'board' and 'depart' actions required for unserved passengers.
    2.  An estimate of the lift's 'up'/'down' movement actions needed to visit the
        required origin and destination floors.

    # Assumptions
    - The domain involves a single lift serving multiple passengers.
    - Floors are linearly ordered. The `(above f1 f2)` predicate implies `f1` is strictly
      above `f2` and defines a transitive relationship.
    - Floor names allow mapping to unique numerical levels (floor numbers). This heuristic
      calculates this mapping based on the count of floors strictly below each floor,
      derived from the `(above ...)` static facts.
    - Moving between adjacent floors costs 1 action ('up' or 'down').
    - The primary goal is always to achieve `(served p)` for a specific set of passengers.
    - The heuristic is designed for Greedy Best-First Search and prioritizes informativeness
      over admissibility.

    # Heuristic Initialization
    - The constructor (`__init__`) pre-processes static information from the planning task:
        - It identifies all unique floor objects from static facts, initial state, and goals.
        - It parses `(above f1 f2)` facts to build a `floor_to_num` mapping, assigning a
          numerical level to each floor based on how many floors are below it.
        - It parses `(destin p f)` facts to store each passenger's destination (`destin_map`).
        - It identifies the set of passengers required to be served in the goal (`goal_passengers`).

    # Step-By-Step Thinking for Computing Heuristic
    The `__call__` method computes the heuristic value for a given state node:
    1.  **Parse Current State:** Extracts the current state information:
        - The lift's current floor (`current_lift_floor`).
        - Passengers waiting at their origin (`waiting_passengers`: map p -> origin_floor).
        - Passengers currently inside the lift (`boarded_passengers`: set).
        - Passengers already served (`served_passengers`: set).
    2.  **Check Goal:** If the set of `served_passengers` matches the set of `goal_passengers`,
        the goal is reached, and the heuristic returns 0.
    3.  **Calculate Passenger Action Cost (`h_actions`):**
        - Each passenger waiting at their origin requires 1 `board` and 1 `depart` action (cost = 2).
        - Each passenger currently boarded requires 1 `depart` action (cost = 1).
        - `h_actions` = (2 * number of waiting passengers) + (1 * number of boarded passengers).
    4.  **Calculate Lift Movement Cost (`h_movement`):**
        - Identify the set of "target floors" the lift must visit for current requests:
            - The origin floor for each waiting passenger.
            - The destination floor for each boarded passenger.
        - Convert these floor names to numerical levels using `floor_to_num`. Let this set be `TargetFloorNums`.
        - Get the numerical level of the lift's current floor (`current_lift_num`).
        - If `TargetFloorNums` is empty, no immediate movement is required for passengers, so `h_movement = 0`.
        - Otherwise:
            - Find the minimum (`min_req`) and maximum (`max_req`) floor numbers in `TargetFloorNums`.
            - Find the target floor (`nearest_target_num`) closest to the lift's current position.
            - Estimate movement cost as: distance to the nearest target + the total vertical span of required floors.
              `h_movement = abs(current_lift_num - nearest_target_num) + (max_req - min_req)`.
    5.  **Total Heuristic Value:** The final estimate is the sum of the action and movement costs:
        `h = h_actions + h_movement`. The value is ensured to be non-negative.
    """

    def __init__(self, task):
        # Initialize the base Heuristic class
        super().__init__(task)
        # Store task components for easier access
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Identify all unique floor objects from the problem definition
        self.floors = set()
        # Collect floors mentioned in 'above' static facts
        for fact in static_facts:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == 'above':
                 self.floors.add(parts[1])
                 self.floors.add(parts[2])
             elif pred == 'destin': # Also capture floors mentioned as destinations
                 self.floors.add(parts[2])
        # Collect floors mentioned in the initial state
        for fact in initial_state:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == 'lift-at':
                 self.floors.add(parts[1])
             elif pred == 'origin':
                 self.floors.add(parts[2])
        # Collect floors mentioned in goals (less common, but possible)
        for goal in self.goals:
             parts = get_parts(goal)
             # Example: if a goal involved the lift being at a specific floor
             if parts[0] == 'lift-at':
                 self.floors.add(parts[1])

        if not self.floors:
             print("Warning: No floor objects identified from problem definition. Heuristic may fail.", file=sys.stderr)

        # 2. Build the floor-to-number mapping using 'above' facts
        self.floor_to_num = {}
        # Create an adjacency list representing the 'below' relationship (f_above -> set of f_below)
        below_map = {f: set() for f in self.floors}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:]
                # Ensure both floors are recognized before adding the relationship
                if f_above in self.floors and f_below in self.floors:
                     below_map[f_above].add(f_below)

        # Calculate the floor number for each floor by counting distinct floors below it (transitive closure)
        for f in self.floors:
            all_below = set()
            queue = list(below_map.get(f, set())) # Start BFS with floors directly below f
            visited_below = set(queue) # Keep track of visited floors to avoid cycles/redundancy
            while queue:
                current_f = queue.pop(0)
                all_below.add(current_f)
                # Explore floors below the current floor
                for next_below in below_map.get(current_f, set()):
                    if next_below not in visited_below:
                        visited_below.add(next_below)
                        queue.append(next_below)
            # The floor number is the count of unique floors below it
            self.floor_to_num[f] = len(all_below)

        # Sanity check: Verify that floor numbers are unique, indicating a linear order
        if len(self.floors) > 0 and len(self.floor_to_num) != len(set(self.floor_to_num.values())):
             print(f"Warning: Floor numbering conflict detected (duplicate numbers assigned). "
                   f"This suggests the 'above' predicates might not define a strict linear order. "
                   f"Heuristic calculations might be affected. Map: {self.floor_to_num}", file=sys.stderr)

        # 3. Store passenger destination floors from static facts
        self.destin_map = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, p, f = get_parts(fact)
                # Check if the destination floor is a known floor
                if f not in self.floors:
                     print(f"Warning: Destination floor '{f}' for passenger '{p}' is not among the identified floors.", file=sys.stderr)
                self.destin_map[p] = f

        # 4. Identify the set of passengers that need to be served according to the goal
        self.goal_passengers = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 self.goal_passengers.add(get_parts(goal)[1])

        if not self.goal_passengers:
             print("Warning: No '(served p)' goals found. The heuristic assumes this is the primary goal structure.", file=sys.stderr)


    def __call__(self, node):
        """Calculates the heuristic value for the given state node."""
        state = node.state

        # 1. Parse the current state to understand passenger and lift status
        current_lift_floor = None
        waiting_passengers = {} # map: passenger -> origin_floor
        boarded_passengers = set()
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "lift-at":
                # Record lift position, handling potential (though unlikely) multiple lift-at facts
                if current_lift_floor is not None and current_lift_floor != parts[1]:
                     print(f"State Error: Multiple lift positions detected ('{current_lift_floor}', '{parts[1]}'). Using the first one found.", file=sys.stderr)
                if current_lift_floor is None:
                    current_lift_floor = parts[1]
            elif pred == "origin":
                p, f = parts[1], parts[2]
                # Track waiting passengers only if they are part of the goal
                if p in self.goal_passengers:
                    waiting_passengers[p] = f
            elif pred == "boarded":
                p = parts[1]
                # Track boarded passengers relevant to the goal
                if p in self.goal_passengers:
                    boarded_passengers.add(p)
            elif pred == "served":
                 p = parts[1]
                 # Track served passengers relevant to the goal
                 if p in self.goal_passengers:
                     served_passengers.add(p)

        # 2. Check if the goal state has been reached
        # The goal is met if all passengers required to be served are indeed served.
        if served_passengers == self.goal_passengers:
             return 0 # Goal state has heuristic value 0

        # --- Pre-computation Checks ---
        # Ensure the lift's position was found in the state
        if current_lift_floor is None:
            raise ValueError("State Error: Lift position ('lift-at') predicate not found in the current state.")
        # Ensure the lift's current floor has a valid number mapping
        if current_lift_floor not in self.floor_to_num:
             raise ValueError(f"State Error or Configuration Issue: Lift's current floor '{current_lift_floor}' "
                              f"could not be mapped to a number. Check floor definitions and 'above' facts. "
                              f"Current map: {self.floor_to_num}")
        current_lift_num = self.floor_to_num[current_lift_floor]

        # 3. Calculate the cost associated with passenger actions (boarding/departing)
        num_waiting = len(waiting_passengers)
        num_boarded = len(boarded_passengers)
        # Cost = 2 actions (board + depart) for each waiting passenger
        # Cost = 1 action (depart) for each boarded passenger
        h_actions = (2 * num_waiting) + (1 * num_boarded)

        # 4. Calculate the estimated cost of lift movement
        pickup_floors_nums = set()
        for p, f in waiting_passengers.items():
            if f not in self.floor_to_num:
                 # This indicates an inconsistency between the state and the precomputed map
                 raise ValueError(f"State/Configuration Error: Origin floor '{f}' for passenger '{p}' "
                                  f"is not in the floor number map: {self.floor_to_num}")
            pickup_floors_nums.add(self.floor_to_num[f])

        dropoff_floors_nums = set()
        for p in boarded_passengers:
            dest_f = self.destin_map.get(p)
            if dest_f is None:
                 # This implies a passenger is boarded but their destination isn't known from static facts
                 raise ValueError(f"Configuration Error: Destination floor for boarded passenger '{p}' is unknown.")
            if dest_f not in self.floor_to_num:
                 # Destination floor from static facts doesn't match known floors
                 raise ValueError(f"Configuration Error: Destination floor '{dest_f}' for passenger '{p}' "
                                  f"is not in the floor number map: {self.floor_to_num}")
            dropoff_floors_nums.add(self.floor_to_num[dest_f])

        # Combine the sets of floor numbers the lift needs to visit for current requests
        target_floor_nums = pickup_floors_nums | dropoff_floors_nums

        h_movement = 0
        if target_floor_nums:
            # Find the minimum and maximum floor numbers among the targets
            min_req_floor = min(target_floor_nums)
            max_req_floor = max(target_floor_nums)

            # Find the target floor that is numerically closest to the lift's current floor
            # Uses min() with a lambda function to find the element minimizing the distance calculation
            nearest_target_num = min(target_floor_nums, key=lambda f_num: abs(current_lift_num - f_num))
            min_dist_to_target = abs(current_lift_num - nearest_target_num)

            # Estimate movement cost: distance to the nearest target + the span of the target range
            # This approximates the travel needed to start service and cover the required vertical distance.
            h_movement = min_dist_to_target + (max_req_floor - min_req_floor)
        # If target_floor_nums is empty, h_movement remains 0

        # 5. Calculate the total heuristic value
        heuristic_value = h_actions + h_movement
        # Ensure the heuristic value is never negative
        return max(0, heuristic_value)

