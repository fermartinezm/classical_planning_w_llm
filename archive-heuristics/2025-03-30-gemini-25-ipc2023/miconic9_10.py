import collections
from fnmatch import fnmatch
import math # Import math for infinity

# Try to import the base class provided by the planner environment.
# If it's not available, define a placeholder base class.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class not found. Using a placeholder.")
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError("Heuristic calculation not implemented.")

# Helper functions (can be module-level or static methods)
def get_parts(fact):
    """
    Extract the components of a PDDL fact string.
    Example: "(predicate obj1 obj2)" -> ["predicate", "obj1", "obj2"]
    """
    # Remove parentheses and split by space
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern using fnmatch for wildcards.

    Args:
        fact (str): The PDDL fact string (e.g., "(at obj room)").
        *args: A sequence of strings representing the pattern elements
               (e.g., "at", "*", "room"). Wildcards (*) are supported.

    Returns:
        bool: True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the pattern length
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern element using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'miconic'.

    # Summary
    This heuristic estimates the number of actions required to reach a goal state
    where all passengers are served `(served p)`. It calculates the cost as a sum
    of two components:
    1.  `h_actions`: The minimum number of `board` and `depart` actions required
        for passengers based on their current state (waiting or boarded).
    2.  `h_move`: An estimate of the lift movement actions (`up`/`down`) needed.
        This considers the distance from the lift's current position to the
        nearest required floor (an origin or destination) and the total vertical
        span of floors the lift needs to cover to serve all passengers.

    The heuristic is designed for Greedy Best-First Search and is not necessarily
    admissible.

    # Assumptions
    - The `(above fA fB)` predicate indicates that floor `fA` is directly one
      level above floor `fB`, implying a linear ordering of floors.
    - Floor levels (numerical heights) can be consistently derived from the
      `above` predicates. The implementation calculates these levels, assuming
      the lowest floor is at level 0.
    - The lift has infinite capacity (a standard PDDL assumption unless capacity
      constraints are modeled).
    - The goal is always to achieve `(served p)` for all relevant passengers `p`.

    # Heuristic Initialization (`__init__`)
    - Parses static facts (`task.static`) and facts from the initial state
      (`task.initial_state`) to identify all floors and passengers.
    - Builds a mapping `floor_levels` from floor names (str) to numerical levels
      (int) based on the `(above fA fB)` relations. It finds the lowest floor(s)
      and assigns level 0, then works upwards. Includes warnings for potential
      inconsistencies (e.g., disconnected floors, cycles).
    - Parses static `(destin p f)` facts to store the destination floor for each
      passenger in the `destinations` dictionary.
    - Compiles a set `passengers` containing all passenger names found.
    - Stores the goal conditions (`task.goals`) for reference (though typically
      goals are just `(served p)` for all `p`).

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Get Current State:** Obtain the set of facts `state` from the input `node`.
    2.  **Find Lift Location:** Identify the current floor `f_lift` from the
        `(lift-at f_lift)` fact in the state. If not found, return infinity (error).
    3.  **Identify Passenger States:** Iterate through the state facts to determine:
        - `waiting_passengers`: A dictionary `{passenger: origin_floor}` for those
          with an `(origin p f)` fact.
        - `boarded_passengers`: A set of passengers with a `(boarded p)` fact.
        - `served_passengers`: A set of passengers with a `(served p)` fact.
    4.  **Identify Unserved Passengers:** Calculate the set `unserved_passengers`
        by taking the difference between all known passengers (`self.passengers`)
        and the `served_passengers`.
    5.  **Check for Goal State:** If `unserved_passengers` is empty, the goal is
        reached, return 0.
    6.  **Calculate Action Costs (`h_actions`):**
        - Each passenger in `waiting_passengers` needs one `board` and one `depart`
          action eventually (cost contribution: 2).
        - Each passenger in `boarded_passengers` needs one `depart` action
          (cost contribution: 1).
        - `h_actions = (2 * len(waiting_passengers)) + len(boarded_passengers)`.
    7.  **Determine Required Floors for Movement:**
        - `O`: Set of origin floors for waiting passengers.
        - `D_U`: Set of destination floors for all *unserved* passengers (using
          `self.destinations`).
        - `OriginsDestinations = O U D_U`: The set of floors the lift must visit
          for pickups or dropoffs related to currently unserved passengers.
    8.  **Calculate Movement Costs (`h_move`):**
        - Handle the edge case where `OriginsDestinations` is empty (unlikely if
          `h_actions > 0`, but possible with definition errors). If empty, `h_move = 0`.
        - **Cost to Nearest Stop (`min_dist`):** Find the minimum distance from
          `f_lift` to any floor `f` in `OriginsDestinations`. Distance is calculated
          as `abs(level(f_lift) - level(f))`. Uses helper `_get_distance` which
          handles unknown levels (returning 1 as a minimum estimate).
        - **Travel Span Cost (`span`):** Determine the set of all floors potentially
          relevant to the current plan: `RequiredFloors = {f_lift} U OriginsDestinations`.
          Find the minimum (`min_req_level`) and maximum (`max_req_level`) known
          levels among these floors. `span = max_req_level - min_req_level`. If no
          valid levels are found, `span = 0`.
        - Combine movement estimates: `h_move = min_dist + span`.
    9.  **Combine Costs:** The final heuristic value is `h = h_actions + h_move`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static info and goals."""
        super().__init__(task) # Store task if base class requires it
        self.goals = task.goals
        static_facts = task.static
        # Combine static facts and initial state facts for comprehensive setup
        all_init_facts = task.initial_state.union(static_facts)

        # 1. Build floor levels map
        self.floor_levels = self._build_floor_levels(all_init_facts)
        if not self.floor_levels:
             print("Warning: Heuristic init: No floor levels could be determined.")

        # 2. Store passenger destinations and identify all passengers
        self.destinations = {}
        self.passengers = set()
        # Discover passengers and destinations from relevant facts
        for fact in all_init_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "destin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor
                self.passengers.add(passenger)
            elif predicate == "origin" and len(parts) == 3:
                 # Discover passengers from origin facts too
                 self.passengers.add(parts[1])
            elif predicate == "passenger" and len(parts) == 2: # If type declarations exist
                 self.passengers.add(parts[1])
            elif predicate == "boarded" and len(parts) == 2: # Passenger might only appear boarded initially
                 self.passengers.add(parts[1])
            elif predicate == "served" and len(parts) == 2: # Passenger might only appear served initially
                 self.passengers.add(parts[1])


        # Ensure passengers mentioned in goals are included
        for goal in self.goals:
             if match(goal, "served", "*"):
                 self.passengers.add(get_parts(goal)[1])

        if not self.passengers:
             print("Warning: Heuristic init: No passengers found in the problem definition.")


    def _build_floor_levels(self, facts):
        """
        Builds a mapping from floor name to its level (integer height).
        Assumes floors are linearly ordered and (above fA fB) means fA is
        directly above fB. Level 0 is assigned to the lowest floor.
        Handles discovery of floors from various predicates.
        """
        above_map = {}  # Stores f_below -> f_above (direct successor going up)
        all_floors = set()

        # Discover floors and 'above' relations
        for fact in facts:
            parts = get_parts(fact)
            predicate = parts[0]
            # Check for 'above' relation
            if predicate == 'above' and len(parts) == 3:
                f_above, f_below = parts[1], parts[2]
                above_map[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)
            # Check for floor type declaration if present
            elif predicate == 'floor' and len(parts) == 2:
                 all_floors.add(parts[1])
            # Infer floors from usage in other relevant predicates
            elif predicate in ['lift-at', 'origin', 'destin']:
                floor_arg_indices = {'lift-at': [1], 'origin': [2], 'destin': [2]}
                if predicate in floor_arg_indices:
                    for idx in floor_arg_indices[predicate]:
                         if len(parts) > idx:
                             potential_floor = parts[idx]
                             # Basic check: does it look like a floor name?
                             # Assumes floors start with 'f' - adjust if needed.
                             if isinstance(potential_floor, str) and potential_floor.startswith('f'):
                                 all_floors.add(potential_floor)

        if not all_floors:
            return {} # No floors found

        # Find bottom floors: those that are never 'f_above'
        floors_that_are_above_others = set(above_map.values())
        bottom_floors = all_floors - floors_that_are_above_others

        if not bottom_floors:
            if len(all_floors) == 1:
                return {list(all_floors)[0]: 0} # Single floor case
            else:
                # Cyclic or incomplete 'above' definitions, or no 'above' defined
                print(f"Warning: Heuristic init: Could not determine the bottom floor among {all_floors}. Assigning arbitrary levels.")
                # Fallback: assign levels alphabetically for determinism
                sorted_floors = sorted(list(all_floors))
                return {f: i for i, f in enumerate(sorted_floors)}

        # Assume a single linear structure, pick one bottom floor deterministically
        # Sorting ensures deterministic choice if multiple bottom floors exist (unusual)
        bottom_floor = sorted(list(bottom_floors))[0]

        floor_levels = {bottom_floor: 0}
        queue = collections.deque([bottom_floor])
        visited = {bottom_floor}

        while queue:
            current_floor = queue.popleft()
            current_level = floor_levels[current_floor]

            # Find the floor directly above the current one using the pre-parsed map
            if current_floor in above_map:
                next_floor = above_map[current_floor]
                if next_floor not in visited:
                     floor_levels[next_floor] = current_level + 1
                     visited.add(next_floor)
                     queue.append(next_floor)
                # Check for cycles/inconsistencies (optional but good practice)
                elif floor_levels.get(next_floor) != current_level + 1:
                     print(f"Warning: Heuristic init: Re-visiting floor {next_floor} with inconsistent level. Check 'above' relations for cycles.")

        # Assign -1 to any floors that were declared but not reached (disconnected)
        for f in all_floors:
            if f not in floor_levels:
                print(f"Warning: Heuristic init: Floor {f} seems disconnected. Assigning level -1.")
                floor_levels[f] = -1

        return floor_levels

    def _get_level(self, floor):
        """Safely get the level of a floor, returning -1 if unknown."""
        level = self.floor_levels.get(floor, -1)
        # Avoid repeated warnings if level was already determined to be -1 (disconnected)
        if level == -1 and floor not in self.floor_levels:
             print(f"Warning: Heuristic call: Level requested for unknown floor {floor}. Returning -1.")
        return level

    def _get_distance(self, floor1, floor2):
        """
        Calculate distance between floors based on levels.
        Returns 1 if either level is unknown, as a minimum move estimate.
        """
        level1 = self._get_level(floor1)
        level2 = self._get_level(floor2)
        if level1 == -1 or level2 == -1:
            # Cannot compute distance accurately. Assume minimum move cost of 1.
            return 1
        return abs(level1 - level2)

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given node's state."""
        state = node.state # The state is a frozenset of PDDL fact strings

        # Find lift location
        f_lift = None
        for fact in state:
            # Use match for potentially faster checking if state is large
            if match(fact, "lift-at", "*"):
                f_lift = get_parts(fact)[1]
                break
        if f_lift is None:
            print("Error: Heuristic call: Lift location not found in state.")
            return float('inf') # Indicate an invalid or error state

        # Identify passenger states
        waiting_passengers = {} # p -> origin_floor
        boarded_passengers = set()
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "origin" and len(parts) == 3:
                waiting_passengers[parts[1]] = parts[2]
            elif predicate == "boarded" and len(parts) == 2:
                boarded_passengers.add(parts[1])
            elif predicate == "served" and len(parts) == 2:
                served_passengers.add(parts[1])

        # Identify all unserved passengers using the precomputed set
        unserved_passengers = self.passengers - served_passengers

        # If all passengers are served, heuristic is 0
        if not unserved_passengers:
             # Sanity check: if no unserved passengers, waiting and boarded should be empty
             if waiting_passengers or boarded_passengers:
                  print("Warning: Heuristic call: Inconsistency - no unserved passengers, but some are waiting or boarded.")
             return 0

        # Calculate base action costs (board/depart)
        # Each waiting passenger needs 1 board + 1 depart = 2 actions
        # Each boarded passenger needs 1 depart = 1 action
        h_actions = (2 * len(waiting_passengers)) + len(boarded_passengers)

        # --- Calculate movement costs ---
        h_move = 0
        origins = set(waiting_passengers.values())
        # Get destinations for *all* unserved passengers that have a known destination
        destinations_unserved = set()
        for p in unserved_passengers:
             if p in self.destinations:
                 destinations_unserved.add(self.destinations[p])
             else:
                 # This might happen if a passenger is added dynamically or problem is ill-defined
                 print(f"Warning: Heuristic call: Unserved passenger {p} has no destination defined in static facts.")

        # Floors the lift needs to visit (excluding current position initially)
        origins_destinations = origins.union(destinations_unserved)

        if not origins_destinations:
            # No specific floors to visit. This could happen if all unserved passengers
            # are boarded but have no known destinations. Movement cost is 0.
            min_dist = 0
            span = 0
        else:
            # Calculate distance to nearest required stop
            current_lift_level = self._get_level(f_lift)
            min_dist = float('inf')
            computable_dist_found = False # Track if we found any floor with known level

            for f in origins_destinations:
                dist = self._get_distance(f_lift, f)
                min_dist = min(min_dist, dist)
                # If distance is not 1 (the default for unknown levels), it's computable
                if self._get_level(f_lift) != -1 and self._get_level(f) != -1:
                     computable_dist_found = True

            # If min_dist is still inf, something went wrong. Default to 1 if stops exist.
            if min_dist == float('inf'):
                 min_dist = 1

            # Calculate travel span
            required_floors_for_span = {f_lift}.union(origins_destinations)
            min_req_level = float('inf')
            max_req_level = float('-inf')
            valid_level_found_for_span = False
            for f in required_floors_for_span:
                level = self._get_level(f)
                if level != -1: # Only consider floors with known levels for span
                    min_req_level = min(min_req_level, level)
                    max_req_level = max(max_req_level, level)
                    valid_level_found_for_span = True

            if not valid_level_found_for_span:
                 span = 0 # No valid levels found to compute span
            else:
                 # Span is the difference between highest and lowest required levels
                 span = max_req_level - min_req_level

        # Combine movement costs: distance to nearest stop + range of travel
        h_move = min_dist + span

        # Final heuristic value: actions needed + estimated movement
        return h_actions + h_move
