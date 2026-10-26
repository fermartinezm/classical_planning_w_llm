# Required imports
import re

# Assuming the Task class is available in the environment where this heuristic is used.
# from task import Task # Example import if Task is in a separate file

class MiconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the remaining cost by summing the number of board actions needed,
    the number of depart actions needed, and the estimated number of move actions
    the lift must perform.
    - Board actions needed: Number of unserved passengers not currently boarded.
    - Depart actions needed: Number of unserved passengers.
    - Estimated move actions: Minimum number of floor movements required for the
      lift to visit all origin floors of waiting passengers and all destination
      floors of boarded passengers, starting from the current lift floor. This
      is calculated based on the range of required floors and the lift's current
      position relative to that range.

    Assumptions:
    - Floor names are consistently formatted (e.g., 'f1', 'f2', 'f10').
    - The 'above' predicates define a linear ordering of floors, allowing them
      to be mapped to sequential integer indices. The heuristic assumes floors
      can be sorted numerically based on the number in their name.
    - Passenger origins and destinations are static and provided in the static facts.
    - The problem is solvable (i.e., all origins and destinations are valid floors).

    Heuristic Initialization:
    The constructor processes the static facts to:
    1. Determine the ordered list of floors and create a mapping from floor names
       to integer indices and vice versa. This is done by parsing the number
       from floor names like 'f<number>' and sorting them numerically.
    2. Store the origin and destination floor for each passenger in dictionaries.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the state is a goal state (all passengers served). If yes, return 0.
    2. Identify the current floor of the lift from the state. Map it to its index.
    3. Identify all unserved passengers. An unserved passenger is one for whom
       '(served <passenger>)' is not in the state.
    4. Count the number of unserved passengers who are currently waiting at their
       origin floor (i.e., not boarded and not served, and their origin fact is true).
       Each of these passengers requires a 'board' action. This count contributes
       directly to the heuristic.
    5. Count the total number of unserved passengers. Each of these passengers
       requires a 'depart' action at some point. This count contributes directly
       to the heuristic.
    6. Determine the set of 'required' floors that the lift must visit. This set
       includes:
       - The origin floor of any unserved passenger who is not boarded (waiting).
       - The destination floor of any unserved passenger who is boarded.
    7. Map these required floor names to their integer indices.
    8. If the set of required floor indices is empty, the estimated moves is 0.
    9. If the set of required floor indices is not empty, find the minimum and
       maximum indices in this set (`min_v_idx`, `max_v_idx`).
    10. Calculate the estimated number of move actions:
        - If the current lift floor index (`f_current_idx`) is less than `min_v_idx`,
          the lift must travel up at least to `min_v_idx` and then cover the range
          up to `max_v_idx`. Estimated moves = `(min_v_idx - f_current_idx) + (max_v_idx - min_v_idx)`.
        - If `f_current_idx` is greater than `max_v_idx`, the lift must travel down
          at least to `max_v_idx` and then cover the range down to `min_v_idx`.
          Estimated moves = `(f_current_idx - max_v_idx) + (max_v_idx - min_v_idx)`.
        - If `f_current_idx` is within the range `[min_v_idx, max_v_idx]`, the lift
          must travel to the furthest required floor from its current position.
          Estimated moves = `max(f_current_idx - min_v_idx, max_v_idx - current_lift_floor_idx)`.
    11. The total heuristic value is the sum of the counts from steps 4 and 5,
        plus the estimated moves from step 10.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        @param task: The planning task object (instance of Task class).
        """
        self.task = task
        self.passenger_origin = {}
        self.passenger_destin = {}
        self.floor_to_idx = {}
        self.idx_to_floor = {}
        self._process_static_facts()

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into a tuple."""
        # Remove surrounding brackets and split by space
        # Use regex to handle potential multiple spaces or weird formatting
        parts = re.findall(r'\S+', fact_str.strip("()"))
        return tuple(parts)

    def _process_static_facts(self):
        """Processes static facts to build floor map and passenger info."""
        floors = set()
        # Collect all floors and passenger origins/destinations
        for fact_str in self.task.static:
            fact = self._parse_fact(fact_str)
            if fact[0] == 'origin':
                # fact is ('origin', passenger, floor)
                passenger, floor = fact[1], fact[2]
                self.passenger_origin[passenger] = floor
                floors.add(floor)
            elif fact[0] == 'destin':
                # fact is ('destin', passenger, floor)
                passenger, floor = fact[1], fact[2]
                self.passenger_destin[passenger] = floor
                floors.add(floor)
            elif fact[0] == 'above':
                # fact is ('above', floor1, floor2)
                floors.add(fact[1])
                floors.add(fact[2])

        # Sort floors numerically based on the number in the name
        # Assumes floor names are like 'f<number>'
        # This might fail if floor names are not in this format.
        # A more robust approach would build the order from 'above' facts.
        # Given the examples, numerical sort seems intended.
        sorted_floors = sorted(list(floors), key=lambda f: int(f[1:]))

        # Create floor name to index mapping and vice versa
        for i, floor in enumerate(sorted_floors):
            # Use 1-based indexing for floors to match intuition f1 -> index 1
            self.floor_to_idx[floor] = i + 1
            self.idx_to_floor[i + 1] = floor

    def _get_floor_index(self, floor_name):
        """Maps a floor name to its integer index."""
        # This will raise KeyError if floor_name is not in self.floor_to_idx,
        # which indicates an issue with the input problem definition.
        return self.floor_to_idx[floor_name]

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated number of actions to reach the goal.
        """
        # 1. Check if goal is reached
        # The goal is having all passengers served.
        # We can check this by seeing if the number of unserved passengers is 0.
        # Let's calculate unserved passengers first.

        all_passengers = set(self.passenger_origin.keys()) # Get all passenger names
        served_passengers = set()
        boarded_passengers = set()
        waiting_at_origin_passengers = set() # Passengers for whom (origin p o) is true in state

        current_lift_floor = None

        # Iterate through state facts once to populate sets and find lift location
        for fact_str in state:
            fact = self._parse_fact(fact_str)
            if fact[0] == 'served':
                served_passengers.add(fact[1])
            elif fact[0] == 'boarded':
                boarded_passengers.add(fact[1])
            elif fact[0] == 'origin':
                 # fact is ('origin', passenger, floor)
                 passenger = fact[1]
                 waiting_at_origin_passengers.add(passenger)
            elif fact[0] == 'lift-at':
                current_lift_floor = fact[1]

        unserved_passengers = all_passengers - served_passengers

        # If no unserved passengers, goal is reached
        if not unserved_passengers:
             return 0

        # 2. Get current lift floor index
        # We assume current_lift_floor was found in the state iteration
        current_lift_floor_idx = self._get_floor_index(current_lift_floor)

        # 4. Count board actions needed
        # These are unserved passengers who are currently waiting at their origin.
        # A passenger is waiting at origin if '(origin p o)' is true in the state.
        # If '(origin p o)' is true, '(boarded p)' must be false (due to action effects).
        # So, unserved waiting passengers are those in waiting_at_origin_passengers
        # who are also in unserved_passengers.
        unserved_waiting_passengers = waiting_at_origin_passengers & unserved_passengers
        num_board_actions_needed = len(unserved_waiting_passengers)

        # 5. Count depart actions needed
        # These are all unserved passengers.
        num_depart_actions_needed = len(unserved_passengers)

        # 6. Determine required floors
        required_floor_indices = set()

        # Origins of unserved waiting passengers
        for passenger in unserved_waiting_passengers:
             origin_floor = self.passenger_origin[passenger] # Use [] as key should exist
             required_floor_indices.add(self._get_floor_index(origin_floor))

        # Destinations of unserved boarded passengers
        # These are passengers in boarded_passengers but not in served_passengers
        unserved_boarded_passengers = boarded_passengers - served_passengers
        for passenger in unserved_boarded_passengers:
             destin_floor = self.passenger_destin[passenger] # Use [] as key should exist
             required_floor_indices.add(self._get_floor_index(destin_floor))

        # 8. Calculate estimated moves
        estimated_moves = 0
        if required_floor_indices:
            min_v_idx = min(required_floor_indices)
            max_v_idx = max(required_floor_indices)

            # 10. Calculate estimated move actions
            if current_lift_floor_idx < min_v_idx:
                # Must go up at least to min_v_idx, then cover range up to max_v_idx
                estimated_moves = (min_v_idx - current_lift_floor_idx) + (max_v_idx - min_v_idx)
            elif current_lift_floor_idx > max_v_idx:
                # Must go down at least to max_v_idx, then cover range down to min_v_idx
                 estimated_moves = (current_lift_floor_idx - max_v_idx) + (max_v_idx - min_v_idx)
            else: # min_v_idx <= current_lift_floor_idx <= max_v_idx
                # Must go to the furthest required floor within the range
                estimated_moves = max(current_lift_floor_idx - min_v_idx, max_v_idx - current_lift_floor_idx)

        # 11. Total heuristic value
        heuristic_value = num_board_actions_needed + num_depart_actions_needed + estimated_moves

        return heuristic_value
