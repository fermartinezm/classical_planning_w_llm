# Add necessary imports
from heuristics.heuristic_base import Heuristic
from task import Operator, Task # Assuming Task class is available

# Define the heuristic class
class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the number of actions required to reach the goal state.
        The heuristic is calculated as the sum of:
        1. The minimum number of floor moves required to visit all floors
           where passengers need to be picked up or dropped off.
        2. The number of passengers waiting at their origin floors (requiring a board action).
        3. The number of passengers who are boarded and need to be dropped off
           at their destination floors (requiring a depart action).

    Assumptions:
        - The PDDL domain follows the structure described, particularly the
          predicates (origin, destin, above, boarded, served, lift-at) and
          actions (board, depart, up, down).
        - The 'above' predicate defines the immediate adjacency relation between floors,
          such that (above f_higher f_lower) means f_higher is immediately above f_lower.
        - The static facts include all 'destin' facts and all 'above' facts defining
          the complete floor structure as a single linear sequence of floors.
        - Floor names can be mapped to integer levels based on the 'above' relation.

    Heuristic Initialization:
        - Parses static facts to build a mapping from passenger to destination floor.
        - Parses static facts to build the immediate 'above' relationship between floors.
        - Determines the lowest floor and constructs mappings between floor names and integer levels.
        - Identifies all passengers from the goal state.

    Step-By-Step Thinking for Computing Heuristic:
        1. Get the current state from the search node.
        2. Extract the current lift location from the state. If not found, return infinity (invalid state).
        3. Identify all passengers who are currently served.
        4. Determine the set of unserved passengers by subtracting served passengers from the total set of passengers.
        5. If there are no unserved passengers, the state is the goal state, return 0.
        6. Identify passengers waiting at their origin floors (those with '(origin p f)' fact in state and p is unserved). Store their origin floors.
        7. Identify passengers who are boarded (those with '(boarded p)' fact in state and p is unserved). Find their destination floors using the pre-calculated destinations map.
        8. Collect the set of unique floors that the lift must visit: the origin floors of waiting unserved passengers and the destination floors of boarded unserved passengers.
        9. Calculate floor moves:
           a. If there are no required stops, floor moves is 0.
           b. If there are required stops:
              i. Ensure all required stops and the current floor have level mappings. If not, return infinity (invalid floor structure).
              ii. Find the integer level of the current lift floor.
              iii. Find the minimum and maximum integer levels among the required stop floors.
              iv. Calculate the minimum floor moves required to visit the range of required floors starting from the current floor. This is max(current_level, max_req_level) - min(current_level, min_req_level).
        10. Count the number of waiting unserved passengers (each needs a board action).
        11. Count the number of boarded unserved passengers (each needs a depart action).
        12. The heuristic value is the sum of floor moves, waiting passenger count, and boarded passenger count.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Parse static facts for destinations
        self.destinations = {}
        # 2. Parse static facts for immediate 'above' relations
        self.immediately_above = {}
        # Store all floors mentioned in 'above' facts to find the lowest later
        all_floors_set = set()

        for fact_str in self.static_facts:
            # Example fact_str: '(destin p1 f10)' or '(above f2 f1)'
            # Remove surrounding brackets and split by space
            parts = fact_str[1:-1].split()
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'destin' and len(args) == 2:
                # args = [passenger, floor]
                self.destinations[args[0]] = args[1]
            elif predicate == 'above' and len(args) == 2:
                # args = [floor_higher, floor_lower]
                f_higher, f_lower = args
                # Assuming (above f_higher f_lower) means f_higher is immediately above f_lower
                self.immediately_above[f_lower] = f_higher
                all_floors_set.add(f_higher)
                all_floors_set.add(f_lower)

        # 3. Determine floor levels based on immediate 'above' relations
        self.floor_to_level = {}
        self.level_to_floor = {}

        # Find the lowest floor (a floor that is in all_floors_set but is not a value in immediately_above)
        floors_that_are_above_something = set(self.immediately_above.values())
        lowest_floor = None
        potential_lowest_floors = [f for f in all_floors_set if f not in floors_that_are_above_something]

        if len(potential_lowest_floors) == 1:
             lowest_floor = potential_lowest_floors[0]
        elif len(all_floors_set) == 1: # Case with only one floor
             if all_floors_set:
                 lowest_floor = list(all_floors_set)[0]
        # else: # Multiple potential lowest floors or no floors mentioned in 'above'
             # The floor mapping will be incomplete. Heuristic will return infinity later if needed floors are not mapped.
             # print(f"Warning: Could not determine a unique lowest floor from static facts. Found {len(potential_lowest_floors)} candidates.")


        if lowest_floor:
            current_level = 1
            current_floor = lowest_floor
            while current_floor is not None:
                if current_floor in self.floor_to_level:
                    # Cycle detected or already processed - break to prevent infinite loop
                    # This indicates an issue with the static 'above' facts structure
                    print(f"Warning: Cycle or duplicate floor detected in floor structure at {current_floor}. Floor mapping incomplete.")
                    self.floor_to_level = {} # Clear potentially incomplete map
                    self.level_to_floor = {}
                    break
                self.floor_to_level[current_floor] = current_level
                self.level_to_floor[current_level] = current_floor
                current_floor = self.immediately_above.get(current_floor) # Get floor immediately above
                current_level += 1

        # 4. Identify all passengers from the goal state
        # Goal facts are like '(served p1)'
        self.all_passengers = {goal.split()[1][:-1] for goal in self.goals if goal.startswith('(served ')}


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Extract current lift location
        current_floor = None
        for fact_str in state:
            if fact_str.startswith('(lift-at '):
                current_floor = fact_str.split()[1][:-1] # Remove trailing ')'
                break

        # If lift location is unknown, this is likely an invalid state representation
        if current_floor is None:
             # Check if it's the goal state (where lift-at might not matter)
             all_served = True
             for p in self.all_passengers:
                 if f"'(served {p})'" not in state:
                     all_served = False
                     break
             if all_served:
                 return 0
             else:
                 # Not goal state and lift location unknown -> invalid state for miconic
                 # print("Error: Lift location unknown in non-goal state.")
                 return float('inf')

        # Ensure current floor is mapped to a level
        if current_floor not in self.floor_to_level:
             # This should not happen if static facts were parsed correctly and state is valid
             print(f"Error: Current floor '{current_floor}' not found in floor level mapping derived from static facts.")
             return float('inf')


        # 2. Identify served passengers in the current state
        served_passengers = {fact_str.split()[1][:-1] for fact_str in state if fact_str.startswith('(served ')}

        # 3. Determine unserved passengers
        unserved_passengers = self.all_passengers - served_passengers

        # 4. If no unserved passengers, goal is reached
        if not unserved_passengers:
            return 0

        # 5. Identify waiting and boarded unserved passengers
        waiting_passengers_info = {} # {passenger: origin_floor}
        boarded_passengers_set = set() # {passenger}

        for fact_str in state:
            # Remove surrounding brackets and split by space
            parts = fact_str[1:-1].split()
            if not parts: # Skip empty facts
                continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'origin' and len(args) == 2 and args[0] in unserved_passengers:
                # args = [passenger, floor]
                waiting_passengers_info[args[0]] = args[1]
            elif predicate == 'boarded' and len(args) == 1 and args[0] in unserved_passengers:
                # args = [passenger]
                boarded_passengers_set.add(args[0])

        # 6. Collect required stop floors
        required_stops = set()
        for p, f_origin in waiting_passengers_info.items():
            # Ensure origin floor is mapped
            if f_origin in self.floor_to_level:
                required_stops.add(f_origin)
            else:
                 print(f"Error: Origin floor '{f_origin}' for passenger '{p}' not found in floor level mapping.")
                 return float('inf') # Invalid state/domain setup

        for p in boarded_passengers_set:
            # Need destination for boarded passengers
            if p in self.destinations:
                 f_destin = self.destinations[p]
                 # Ensure destination floor is mapped
                 if f_destin in self.floor_to_level:
                     required_stops.add(f_destin)
                 else:
                     print(f"Error: Destination floor '{f_destin}' for passenger '{p}' not found in floor level mapping.")
                     return float('inf') # Invalid state/domain setup
            else:
                 # This passenger is boarded but has no destination? Invalid state.
                 print(f"Error: Boarded passenger '{p}' has no destination defined.")
                 return float('inf')


        # 7. Calculate floor moves
        floor_moves = 0
        if required_stops:
            # All required stops and current floor are guaranteed to be in self.floor_to_level by checks above
            current_level = self.floor_to_level[current_floor]
            min_req_level = min(self.floor_to_level[f] for f in required_stops)
            max_req_level = max(self.floor_to_level[f] for f in required_stops)
            # Minimum moves to cover the range [min_req_level, max_req_level] starting from current_level
            floor_moves = max(current_level, max_req_level) - min(current_level, min_req_level)
            # Note: This is a lower bound on travel. It assumes the lift travels directly
            # to one end of the required range and then to the other, visiting all
            # required floors in between. It doesn't account for needing to visit
            # floors outside this range or non-monotonic travel.

        # 8. Count board and depart actions needed for unserved passengers
        board_actions = len(waiting_passengers_info)
        depart_actions = len(boarded_passengers_set)

        # 9. Calculate total heuristic value
        h_value = floor_moves + board_actions + depart_actions

        return h_value
