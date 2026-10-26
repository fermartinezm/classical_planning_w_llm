import math
from heuristics.heuristic_base import Heuristic
# Assuming the Task class is available from this path based on the provided example code.
# Adjust the import path if your project structure is different.
from planning.strips.representation import Task

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts the components of a PDDL fact string by removing the surrounding
    parentheses and splitting the string by spaces.

    Args:
        fact (str): A PDDL fact string, e.g., '(predicate obj1 obj2)'.

    Returns:
        list[str]: A list of strings representing the parts of the fact,
                   e.g., ['predicate', 'obj1', 'obj2'].
    """
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'miconic' (elevator control).

    # Summary
    This heuristic estimates the minimum number of actions required to reach a
    goal state where all specified passengers have been served (transported to
    their destination). The estimate is calculated by summing:
    1. The number of 'board' actions still required for unserved passengers waiting at their origin.
    2. The number of 'depart' actions still required for all unserved passengers.
    3. An estimate of the lift's travel cost, calculated as the vertical distance
       (number of floors) between the highest and lowest floors the lift needs to
       visit (including its current location, pickup floors, and drop-off floors).

    This heuristic is designed for use with Greedy Best-First Search and does not
    guarantee admissibility (i.e., it might overestimate the true cost).

    # Assumptions
    - The PDDL domain uses `(above f1 f2)` predicates to define a strict total
      ordering of floors, where `f1` is somewhere above `f2`.
    - Floor levels are assigned starting from 1 for the bottom-most floor,
      increasing upwards. The level is determined by counting how many floors
      are below a given floor based on the `above` predicates.
    - All actions ('up', 'down', 'board', 'depart') have a uniform cost of 1.
    - The problem instance is well-formed (e.g., passengers in the goal have
      defined destinations, floor structure is consistent).

    # Heuristic Initialization
    - The constructor (`__init__`) processes the static information provided
      in the `task` object.
    - It parses `(destin p f)` facts to store the destination floor for each passenger.
    - It parses `(above f1 f2)` facts to determine the level of each floor. This
      involves identifying all unique floors and calculating their level based
      on how many other floors are below them.
    - It identifies the set of passengers (`goal_passengers`) that must have the
      `(served p)` predicate true in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Get Current State:** Extract relevant information from the current state node:
        - Find the lift's current floor (`lift_floor`) from the `(lift-at f)` fact.
        - Identify all passengers currently served (`served_passengers`) from `(served p)` facts.
        - Identify passengers currently waiting at their origin floor (`current_origins`) from `(origin p f)` facts.
    2.  **Identify Unserved Passengers:** Determine the set of passengers (`unserved_passengers`)
        that are required by the goal but are not yet in the `served_passengers` set.
    3.  **Goal Check:** If `unserved_passengers` is empty, the current state is a goal state
        (or satisfies the goal conditions w.r.t passengers), so the heuristic estimate is 0.
    4.  **Calculate Interaction Costs:**
        - Count how many `unserved_passengers` are currently waiting at their origin
          (`num_waiting_unserved`). This estimates the required 'board' actions.
        - The total number of `unserved_passengers` (`num_depart_needed`) estimates the
          required 'depart' actions.
        - Sum these counts: `interaction_cost = num_waiting_unserved + num_depart_needed`.
    5.  **Identify Required Floors:** Determine the set of all floors (`all_stops`) the lift
        must visit to serve the remaining passengers:
        - Include the current `lift_floor`.
        - Include the origin floors (`pickup_floors`) for all `unserved_passengers` currently waiting.
        - Include the destination floors (`dropoff_floors`) for all `unserved_passengers`.
    6.  **Ensure Floor Levels Known:** Verify that the pre-calculated level is available for
        every floor in `all_stops`. Handle the edge case where only a single floor exists
        in the problem and might not have been defined by static `above` facts.
    7.  **Calculate Movement Cost:**
        - If `all_stops` contains floors at more than one level, calculate the range:
          `movement_cost = max_level - min_level`, where `max_level` and `min_level`
          are the highest and lowest floor levels among `all_stops`.
        - If all stops are on the same level (or there's only one stop), `movement_cost` is 0.
    8.  **Compute Final Heuristic Value:** Sum the interaction cost and the movement cost:
        `h = interaction_cost + movement_cost`. Ensure the result is non-negative.
    """

    def __init__(self, task: Task):
        """
        Initializes the heuristic by processing static information from the task.

        Args:
            task (Task): The planning task object containing static facts, goals, etc.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse passenger destinations from static facts
        self.destin = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                # Format: (destin ?person - passenger ?floor - floor)
                _, person, floor = parts
                self.destin[person] = floor

        # 2. Determine floor levels using 'above' predicates from static facts
        floors = set()
        above_pairs = []
        # Collect all unique floors mentioned in relevant static predicates
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'above':
                # Format: (above ?floor1 - floor ?floor2 - floor)
                _, f1, f2 = parts
                floors.add(f1)
                floors.add(f2)
                above_pairs.append((f1, f2))
            elif predicate == 'destin':
                 # Format: (destin ?person ?floor)
                 floors.add(parts[2])
            # Add checks for other static predicates defining floors if needed

        self.floor_level = {}
        if floors:
            # Calculate level for each floor based on how many floors are below it
            for f in floors:
                # Count f_below such that (above f f_below) is true
                num_below = sum(1 for f_above, f_below in above_pairs if f_above == f)
                # Assign level (bottom floor = level 1)
                self.floor_level[f] = num_below + 1

            # Optional: Sanity check for level consistency (e.g., unique levels)
            if len(self.floor_level) != len(set(self.floor_level.values())):
                 print(f"Warning: Duplicate floor levels detected. Levels: {self.floor_level}. "
                       "This might indicate a non-total order or PDDL issue.")
        # If 'floors' is empty (e.g., single-floor problem not using 'above'),
        # levels might be determined dynamically in __call__.

        # 3. Store the set of passengers required by the goal conditions
        self.goal_passengers = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'served':
                # Format: (served ?person - passenger)
                self.goal_passengers.add(parts[1])

    def _ensure_floor_level(self, floor):
        """
        Ensures the level of a given floor is known.

        If the level is not already stored, it attempts to assign level 1,
        assuming this is the first and only floor encountered (single-floor case).
        Raises a ValueError if the floor level is unknown in a multi-floor scenario,
        indicating an issue with the PDDL definition or static fact processing.

        Args:
            floor (str): The name of the floor to check.
        """
        if floor not in self.floor_level:
            # Handle the edge case: only one floor in the problem, not defined by 'above'.
            if not self.floor_level:
                 # print(f"Info: Dynamically assigning level 1 to the first encountered floor: {floor}")
                 self.floor_level[floor] = 1
            else:
                 # If other floors already have levels, this one should too.
                 raise ValueError(f"Floor level unknown for '{floor}'. This floor was likely not "
                                  f"defined in the static 'above' or 'destin' predicates. "
                                  f"Known levels: {self.floor_level}")

    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.

        Args:
            node: The state node in the search graph, containing the current state.

        Returns:
            int: The estimated cost (number of actions) to reach the goal state.
                 Returns 0 if the goal conditions related to passengers are met.
        """
        state = node.state

        # 1. Find the lift's current floor
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                lift_floor = parts[1]
                break
        if lift_floor is None:
            raise ValueError("Lift location ('lift-at') not found in the current state.")
        self._ensure_floor_level(lift_floor) # Ensure the lift's floor level is known

        # 2. Identify current states of passengers
        served_passengers = set()
        current_origins = {} # Map: passenger -> origin_floor for those at origin
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'served':
                served_passengers.add(parts[1])
            elif predicate == 'origin':
                passenger, floor = parts[1], parts[2]
                current_origins[passenger] = floor
            # Note: 'boarded' facts are implicitly handled later by checking if
            # an unserved passenger is at their origin or not.

        # 3. Determine which goal passengers are still unserved
        unserved_passengers = self.goal_passengers - served_passengers

        # Goal Check: If all required passengers are served, heuristic cost is 0
        if not unserved_passengers:
            return 0

        # 4. Calculate interaction costs (estimated board + depart actions)
        num_waiting_unserved = 0
        pickup_floors = set()
        for p in unserved_passengers:
            if p in current_origins:
                # This unserved passenger is waiting at their origin
                num_waiting_unserved += 1
                origin_floor = current_origins[p]
                pickup_floors.add(origin_floor)
                self._ensure_floor_level(origin_floor) # Ensure origin floor level is known

        # All unserved passengers will eventually need a 'depart' action
        num_depart_needed = len(unserved_passengers)
        # Total estimated interaction actions
        interaction_cost = num_waiting_unserved + num_depart_needed

        # 5. Determine all floors the lift must potentially visit
        dropoff_floors = set()
        for p in unserved_passengers:
            if p in self.destin:
                dest_floor = self.destin[p]
                dropoff_floors.add(dest_floor)
                self._ensure_floor_level(dest_floor) # Ensure destination floor level is known
            else:
                # This indicates an invalid problem setup if a goal passenger has no destination
                raise ValueError(f"Unserved passenger '{p}' is required by the goal "
                                 f"but has no destination ('destin') defined in static facts.")

        # Set of all floors the lift might need to stop at
        all_stops = {lift_floor} | pickup_floors | dropoff_floors

        # 6. Calculate estimated movement cost based on vertical span
        movement_cost = 0
        if len(all_stops) > 1: # Movement is only relevant if multiple floors are involved
            # Collect the levels of all required stops
            stop_levels = {self.floor_level[f] for f in all_stops}

            if len(stop_levels) > 1: # If stops span more than one level
                min_level = min(stop_levels)
                max_level = max(stop_levels)
                # Estimate movement as the difference between highest and lowest required levels
                movement_cost = max_level - min_level

        # 7. Final heuristic value is sum of interactions and movement
        heuristic_value = interaction_cost + movement_cost

        # Return the non-negative heuristic value
        return max(0, heuristic_value)

