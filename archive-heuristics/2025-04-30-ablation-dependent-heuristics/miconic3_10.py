# Import necessary modules
import math # For abs, min, max (though built-in are fine)
# The Task class definition is assumed to be available in the environment.

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        This heuristic estimates the remaining cost to reach the goal state
        (all passengers served) by summing two components:
        1. An estimate of the minimum number of lift movement actions (up/down)
           required to visit all necessary floors.
        2. The total number of board and depart actions still needed for
           all unserved passengers.

    Assumptions:
        - The floor structure is linear and ordered according to the 'above'
          predicates in the static facts. The heuristic parses these facts
          to establish a numerical index for each floor.
        - The cost of 'up' and 'down' actions between adjacent floors is 1.
        - The cost of 'board' and 'depart' actions is 1 each.
        - The lift can carry any number of passengers.
        - The heuristic assumes a simplified strategy where the lift must
          visit all floors where unserved passengers are waiting or need to
          be dropped off. The estimated travel cost is the minimum moves
          to cover the range of these necessary floors starting from the
          current lift position.

    Heuristic Initialization:
        The constructor pre-processes the static information from the task:
        - It identifies all floor objects and establishes a mapping from
          floor names (e.g., 'f1', 'f2') to numerical indices based on the
          '(above f_i f_j)' facts. The index represents the floor's rank
          in the vertical order (lowest floor gets index 1).
        - It identifies all passenger objects and stores their destination
          floor based on the '(destin p f)' facts.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Identify the current floor of the lift from the state facts.
        2. Identify all passengers who are not yet served.
        3. For each unserved passenger:
           - If the passenger is currently boarded (checked via '(boarded p)' fact),
             their destination floor is added to a set of floors the lift must visit.
             Increment a counter for needed 'depart' actions.
           - If the passenger is not boarded (and not served), they must be
             waiting at their origin floor (checked via '(origin p f)' fact).
             Their current origin floor is added to the set of floors the lift
             must visit. Increment a counter for needed 'board' actions.
        4. If the set of floors to visit is empty, it means all passengers are
           served, and the heuristic value is 0.
        5. If the set of floors to visit is not empty:
           - Map the current lift floor and all floors in the visit set to their
             numerical indices using the precomputed mapping.
           - Find the minimum and maximum floor indices among the floors to visit.
           - Calculate the estimated travel cost: This is the minimum number of
             'up' or 'down' actions required to move the lift from its current
             floor to cover the entire range of floors that need visiting.
             This is calculated as the distance from the current floor index
             to the nearest end of the needed floor range (min or max index),
             plus the distance spanning the entire needed floor range
             (max index - min index).
           - The total heuristic value is the sum of the estimated travel cost,
             the total number of needed 'board' actions, and the total number
             of needed 'depart' actions.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-processing static task information.

        Args:
            task: The planning task object (instance of the Task class).
        """
        self.task = task
        self.floor_to_index = {}
        self.index_to_floor = {}
        self.passenger_destin = {}

        # 1. Collect all floor names
        all_floors = set()
        # Look in static and initial state for facts involving floors
        for fact_str in task.static | task.initial_state:
            fact = self._parse_fact(fact_str)
            if fact and fact[0] == 'above':
                all_floors.add(fact[1])
                all_floors.add(fact[2])
            elif fact and fact[0] == 'lift-at':
                all_floors.add(fact[1])
            elif fact and fact[0] == 'origin':
                all_floors.add(fact[2])
            elif fact and fact[0] == 'destin':
                all_floors.add(fact[2])

        # 2. Build floor_to_index mapping based on 'above' facts
        # Count how many floors are "below" each floor
        floor_below_counts = {}
        for f1 in all_floors:
            count = 0
            for f2 in all_floors:
                # (above f1 f2) means f1 is higher than f2, so f2 is below f1
                if '(above {} {})'.format(f1, f2) in task.static:
                    count += 1
            floor_below_counts[f1] = count

        # Sort floors by the number of floors below them to get ranks (indices)
        # The floor with 0 floors below is the lowest (index 1), etc.
        sorted_floors = sorted(floor_below_counts.keys(), key=lambda f: floor_below_counts[f])
        self.floor_to_index = {f: i + 1 for i, f in enumerate(sorted_floors)}
        self.index_to_floor = {i + 1: f for i, f in enumerate(sorted_floors)} # Keep for potential debugging/lookup

        # 3. Build passenger_destin mapping
        all_passengers = set()
        # Look in static and initial state for facts involving passengers
        for fact_str in task.static | task.initial_state:
             fact = self._parse_fact(fact_str)
             if fact and fact[0] in ('origin', 'destin', 'boarded', 'served'):
                 # Passenger name is the second element in these facts
                 all_passengers.add(fact[1])

        for p in all_passengers:
            # Find destin in static facts (destin is usually static)
            destin_found = False
            for fact_str in task.static:
                 fact = self._parse_fact(fact_str)
                 if fact and fact[0] == 'destin' and fact[1] == p:
                     self.passenger_destin[p] = fact[2]
                     destin_found = True
                     break # Assuming one destin per passenger
            # If destin not found in static, check initial state (less common for destin)
            if not destin_found:
                 for fact_str in task.initial_state:
                      fact = self._parse_fact(fact_str)
                      if fact and fact[0] == 'destin' and fact[1] == p:
                          self.passenger_destin[p] = fact[2]
                          break


    @staticmethod
    def _parse_fact(fact_str):
        """Helper to parse a fact string into a tuple."""
        # Removes leading/trailing brackets and splits by space
        # Handles cases like '(predicate)' with no arguments
        parts = fact_str[1:-1].split()
        return tuple(parts)

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            The estimated number of actions to reach the goal.
        """
        # 1. Identify current lift floor
        current_lift_floor = None
        for fact_str in state:
            fact = self._parse_fact(fact_str)
            if fact and fact[0] == 'lift-at':
                current_lift_floor = fact[1]
                break

        # If lift location is unknown, the state is likely invalid or unsolvable from here.
        # Returning infinity guides the search away from such states.
        if current_lift_floor is None:
             return float('inf')

        # 2. Identify floors to visit and actions needed
        floors_to_visit = set()
        num_board_needed = 0
        num_depart_needed = 0

        # Iterate through all passengers identified during initialization
        for passenger in self.passenger_destin.keys():
            served_fact = '(served {})'.format(passenger)
            boarded_fact = '(boarded {})'.format(passenger)

            if served_fact in state:
                # Passenger is already served, no more actions needed for this passenger
                continue

            # Passenger is not served. Check if boarded or waiting.
            if boarded_fact in state:
                # Passenger is boarded but not served. Needs to depart at destin.
                destin_floor = self.passenger_destin[passenger]
                floors_to_visit.add(destin_floor)
                num_depart_needed += 1
            else:
                # Passenger is not served and not boarded. Must be waiting at origin.
                # Find current origin from state facts
                current_origin_floor = None
                # We need to iterate through the current state to find the origin fact
                for fact_str in state:
                    fact = self._parse_fact(fact_str)
                    if fact and fact[0] == 'origin' and fact[1] == passenger:
                        current_origin_floor = fact[2]
                        break

                if current_origin_floor:
                    floors_to_visit.add(current_origin_floor)
                    num_board_needed += 1
                # else: This passenger is unserved, unboarded, and not at an origin fact
                # in the current state. This implies an invalid state or an unreachable
                # passenger. Assuming valid states, this case should not occur for
                # any passenger not yet served or boarded. If it occurs, this path
                # is likely bad, so we could return infinity or just ignore this passenger
                # (which might lead to underestimation if they are truly needed).
                # Ignoring for now, assuming valid state transitions.


        # 4. If no floors need visiting, all active passengers are served (goal state)
        if not floors_to_visit:
            return 0

        # 5. Calculate travel cost
        # Map floor names to indices
        current_lift_index = self.floor_to_index.get(current_lift_floor)
        # If current_lift_floor wasn't found in the initial floor parsing, something is wrong.
        # This check is redundant if current_lift_floor is guaranteed to be in all_floors.
        if current_lift_index is None:
             return float('inf') # Should not happen in valid states

        visit_indices = set()
        for f in floors_to_visit:
             idx = self.floor_to_index.get(f)
             if idx is None:
                  # A floor to visit wasn't in our initial floor list. Invalid state?
                  return float('inf') # Should not happen in valid states
             visit_indices.add(idx)


        min_visit_index = min(visit_indices)
        max_visit_index = max(visit_indices)

        # Estimated travel cost: minimum moves to cover the range [min_visit_index, max_visit_index]
        # starting from current_lift_index.
        # This is the distance from the current floor index to the nearest end of the range + the size of the range.
        dist_to_min = abs(current_lift_index - min_visit_index)
        dist_to_max = abs(current_lift_index - max_visit_index)
        range_dist = max_visit_index - min_visit_index

        travel_cost = min(dist_to_min, dist_to_max) + range_dist

        # 6. Total heuristic = Travel cost + Board cost + Depart cost
        total_heuristic = travel_cost + num_board_needed + num_depart_needed

        return total_heuristic
