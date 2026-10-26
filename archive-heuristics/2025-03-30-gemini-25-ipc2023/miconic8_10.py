import itertools
import math
from heuristics.heuristic_base import Heuristic
# Make sure to place this file in a folder structure like:
# your_planner_base_directory/heuristics/miconic_heuristic.py
# And update the import path for Heuristic if necessary based on your project structure.

# Helper function to parse PDDL facts represented as strings
def get_parts(fact_string):
    """Removes parentheses and splits the fact string into parts."""
    return fact_string[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all passengers
    who have not yet reached their destination. It calculates an estimated cost for each
    unserved passenger individually and sums these costs. The cost for a passenger includes
    the estimated lift movements needed to pick them up (if they are waiting at their origin)
    and drop them off at their destination, plus the mandatory board (if waiting) and
    depart actions.

    # Assumptions
    - The `(above f1 f2)` predicate means floor `f1` is directly above floor `f2`.
    - Floors form one or more linear sequences (stacks). The heuristic computes a numerical
      level for each floor based on these 'above' relationships.
    - The cost of moving the lift between two floors is the absolute difference in their
      assigned numerical levels. This represents the number of single-floor moves (`up`/`down` actions).
    - The heuristic sums the estimated costs for each passenger independently. This approach
      is computationally efficient but may overcount lift movements if multiple passengers
      can share parts of the lift's trip (e.g., picked up or dropped off at the same floor
      or along the same path). This overestimation is acceptable for a non-admissible
      heuristic aimed at guiding a greedy search.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the static facts provided in the task description.
    - It identifies all floors and passengers defined in the problem.
    - It builds a map (`destinations`) storing the destination floor for each passenger based
      on `(destin p f)` facts.
    - It processes the `(above f1 f2)` facts to determine the vertical relationship between floors.
    - It computes a numerical level for each floor (`floor_levels` map), where lower floors
      have lower levels (e.g., level 0 for the bottom-most floor). This is done using a
      graph traversal (BFS) starting from the bottom floors.
    - It stores the goal conditions, which are typically a set of `(served p)` facts.

    # Step-By-Step Thinking for Computing Heuristic
    The `__call__` method computes the heuristic value for a given state (`node.state`):
    1. Check if the current state is a goal state. If `task.goals <= state`, the heuristic value is 0.
    2. Identify the current floor of the lift by finding the `(lift-at f)` fact in the state.
    3. Determine the set of passengers (`p`) that are required to be served in the goal state but are not yet served in the current state.
    4. Initialize `total_heuristic_value` to 0.
    5. For each unserved passenger `p`:
        a. Retrieve the passenger's destination floor `dest_floor` from the precomputed `destinations` map.
        b. Check if the passenger is currently boarded by looking for `(boarded p)` in the state.
            i. If `p` is boarded:
               - The remaining actions are moving the lift to `dest_floor` and departing.
               - Calculate lift movement cost: `dist(current_lift_floor, dest_floor)`.
               - Add the depart action cost: `1`.
               - The cost for this passenger is `dist(current_lift_floor, dest_floor) + 1`.
            ii. If `p` is not boarded (i.e., waiting at their origin):
               - Find the passenger's origin floor `origin_floor` by looking for `(origin p origin_floor)` in the state.
               - The required sequence is: move lift to `origin_floor`, board, move lift to `dest_floor`, depart.
               - Calculate cost to pick up: `dist(current_lift_floor, origin_floor) + 1` (move + board).
               - Calculate cost to drop off: `dist(origin_floor, dest_floor) + 1` (move + depart).
               - The cost for this passenger is `dist(current_lift_floor, origin_floor) + 1 + dist(origin_floor, dest_floor) + 1`.
        c. Add the calculated cost for passenger `p` to the `total_heuristic_value`.
    6. After iterating through all unserved passengers, if the `total_heuristic_value` is 0 but the state is not a goal state (e.g., due to missing floor levels or other edge cases), return 1 to ensure non-zero cost for non-goal states.
    7. Return the `total_heuristic_value`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        - Extracts goals, floors, passengers, destinations.
        - Calculates floor levels based on 'above' predicates.
        """
        self.goals = task.goals
        static_facts = task.static

        # --- Precompute data structures ---

        # 1. Find all floors, passengers, destinations, and direct above relationships
        self.all_floors = set()
        self.all_passengers = set()
        self.destinations = {}
        direct_above_pairs = [] # Stores (floor_above, floor_below) tuples

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'destin':
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor
                self.all_passengers.add(passenger)
                self.all_floors.add(floor) # Destination floor is a floor
            elif predicate == 'above':
                f_above, f_below = parts[1], parts[2]
                self.all_floors.add(f_above)
                self.all_floors.add(f_below)
                direct_above_pairs.append((f_above, f_below))

        # Ensure all passengers mentioned in goals are included
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'served':
                 self.all_passengers.add(parts[1])
        
        # Also add floors mentioned in origin/lift-at from initial state if missed
        # (though static facts should ideally cover all objects)
        for fact in task.initial_state:
             parts = get_parts(fact)
             predicate = parts[0]
             if predicate == 'origin':
                 self.all_floors.add(parts[2])
                 self.all_passengers.add(parts[1])
             elif predicate == 'lift-at':
                 self.all_floors.add(parts[1])


        # 2. Build adjacency lists for floor level calculation
        # adj[f] = list of floors directly below f
        # rev_adj[f] = list of floors directly above f
        adj = {f: [] for f in self.all_floors}
        rev_adj = {f: [] for f in self.all_floors}
        for f_above, f_below in direct_above_pairs:
            # Ensure floors exist in case they were only mentioned in 'above'
            if f_above in adj and f_below in rev_adj:
                 adj[f_above].append(f_below)
                 rev_adj[f_below].append(f_above)
            else:
                 print(f"Warning: Floor mentioned in 'above' ({f_above}, {f_below}) not found elsewhere. Check PDDL consistency.")


        # 3. Calculate floor levels using BFS from bottom floors
        self.floor_levels = {} # Map: floor_name -> level (int)
        queue = [] # Queue for BFS
        num_floors = len(self.all_floors)

        # Find bottom floors (those with no floors below them) and initialize level to 0
        for f in self.all_floors:
            if not adj.get(f): # Use .get() for safety if a floor somehow wasn't added to adj
                self.floor_levels[f] = 0
                queue.append(f)

        # Perform BFS upwards from bottom floors
        processed_count = 0
        head = 0
        while head < len(queue):
            current_floor = queue[head]
            head += 1
            processed_count += 1
            current_level = self.floor_levels[current_floor]

            # Look at floors directly above the current one
            for upper_floor in rev_adj.get(current_floor, []):
                if upper_floor not in self.floor_levels:
                     # Assign level based on the floor below it
                     self.floor_levels[upper_floor] = current_level + 1
                     queue.append(upper_floor)
                else:
                    # If already visited (e.g., multiple paths in non-linear structure),
                    # ensure consistency or take max. Assuming linear stacks, the first assignment is correct.
                    # For robustness in case of odd structures, we could check:
                    # assert self.floor_levels[upper_floor] == current_level + 1, "Inconsistent floor levels found!"
                    pass # Stick with the level assigned first in BFS

        # Check if all floors were reached (handles disconnected components)
        if processed_count != num_floors:
            print(f"Warning: Processed {processed_count} floors via BFS, but expected {num_floors}. "
                  f"May indicate disconnected floor stacks or issue in 'above' predicates.")
            # Assign a default level (e.g., -1 or 0) to unreached floors to avoid errors,
            # although this indicates a potential problem with the PDDL instance.
            for f in self.all_floors:
                if f not in self.floor_levels:
                    print(f"Assigning default level 0 to unreached floor: {f}")
                    self.floor_levels[f] = 0 # Assign default level

    def _get_lift_floor(self, state):
        """Finds the current floor of the lift from the state facts."""
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                return parts[1]
        # This should ideally not happen in a valid state representation
        raise ValueError("Lift location ('lift-at' predicate) not found in state.")

    def _get_passenger_origin(self, state, passenger):
        """Finds the origin floor of a waiting passenger."""
        fact_prefix = f'(origin {passenger}'
        for fact in state:
            if fact.startswith(fact_prefix):
                return get_parts(fact)[2]
        return None # Passenger might be boarded, served, or state is inconsistent

    def _is_boarded(self, state, passenger):
        """Checks if a passenger is currently boarded."""
        return f'(boarded {passenger})' in state

    def _dist(self, floor1, floor2):
        """
        Calculates the distance (number of moves) between two floors
        based on their precomputed levels.
        """
        if floor1 == floor2:
            return 0
        if floor1 not in self.floor_levels or floor2 not in self.floor_levels:
             # Handle case where floor level might be missing due to parsing issues or disconnected floors
             print(f"Warning: Level missing for floor '{floor1}' or '{floor2}'. Returning default distance 1.")
             return 1 # Return a default distance to allow heuristic calculation, but signals an issue.
        level1 = self.floor_levels[floor1]
        level2 = self.floor_levels[floor2]
        return abs(level1 - level2)

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state

        # Check if the goal is already satisfied
        if self.goals <= state:
            return 0

        try:
            lift_floor = self._get_lift_floor(state)
        except ValueError as e:
            print(f"Error: {e}. Returning infinity as heuristic value.")
            return float('inf') # Cannot calculate heuristic without lift position

        total_cost = 0

        # Identify passengers that need to be served according to the goal
        goal_passengers = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'served'}
        
        # Find passengers in the goal set that are not yet served in the current state
        served_passengers_in_state = {get_parts(f)[1] for f in state if get_parts(f)[0] == 'served'}
        unserved_goal_passengers = goal_passengers - served_passengers_in_state

        if not unserved_goal_passengers:
             # This case should be caught by the `self.goals <= state` check,
             # but serves as a safeguard. If no goal passengers remain unserved, cost is 0.
             return 0

        # Calculate cost for each unserved passenger required by the goal
        for p in unserved_goal_passengers:
            passenger_cost = 0
            dest_floor = self.destinations.get(p)

            if not dest_floor:
                 # This indicates an inconsistency between goals and static facts
                 print(f"Warning: Destination for goal passenger {p} not found in static facts. Skipping.")
                 continue # Cannot calculate cost without destination

            if self._is_boarded(state, p):
                # Passenger is in the lift: needs move to destination + depart
                cost_move_to_dest = self._dist(lift_floor, dest_floor)
                cost_depart = 1
                passenger_cost = cost_move_to_dest + cost_depart
            else:
                # Passenger is waiting: needs move to origin + board + move to dest + depart
                origin_floor = self._get_passenger_origin(state, p)
                if origin_floor:
                    cost_move_to_origin = self._dist(lift_floor, origin_floor)
                    cost_board = 1
                    cost_move_to_dest = self._dist(origin_floor, dest_floor)
                    cost_depart = 1
                    passenger_cost = cost_move_to_origin + cost_board + cost_move_to_dest + cost_depart
                else:
                    # Unserved goal passenger is not boarded and not at origin.
                    # This implies the passenger might have been dropped off already but the
                    # '(served p)' fact is missing, or the state is somehow inconsistent.
                    # For heuristic purposes, estimate minimum remaining actions: move lift to dest + depart.
                    print(f"Warning: Unserved goal passenger {p} is not boarded and not at origin. Estimating cost from lift location.")
                    cost_move_to_dest = self._dist(lift_floor, dest_floor)
                    cost_depart = 1
                    passenger_cost = cost_move_to_dest + cost_depart # Minimum cost estimate

            total_cost += passenger_cost

        # Ensure the heuristic returns a non-zero value for non-goal states,
        # even if the calculation resulted in 0 (e.g., if all passengers are
        # somehow at their destination but not 'served').
        if total_cost == 0 and not (self.goals <= state):
            return 1 # Return a minimum cost of 1 for non-goal states

        return total_cost

