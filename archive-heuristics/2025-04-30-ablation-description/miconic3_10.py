# Need to import fnmatch
import fnmatch

# Assume Heuristic base class is available from the environment
# from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check: number of parts must match number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


# Assuming Heuristic base class is provided in the environment like this:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#         self.initial_state = task.initial_state
#
#     def __call__(self, node):
#         raise NotImplementedError


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers who are not yet served. It sums the estimated cost for each
    unserved passenger independently, ignoring interactions and lift capacity.
    The estimated cost for a passenger depends on their current state (at origin
    or boarded) and the lift's current location relative to their origin and
    destination floors.

    # Assumptions
    - Each move action (up/down) costs 1.
    - Each board action costs 1.
    - Each depart action costs 1.
    - The cost for each unserved passenger can be calculated independently and summed.
    - The floor structure is a linear sequence defined by 'above' predicates.
    - Unserved, unboarded passengers are always at their origin floor.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - Parses the 'above' facts from the static information to build a mapping
      from floor names to integer levels, representing their vertical position.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the goal state is reached. If yes, the heuristic is 0.
    2. Find the current floor of the lift.
    3. Identify all passengers who are listed in the goal (need to be served) but are not yet marked as 'served' in the current state.
    4. Initialize the total heuristic cost to 0.
    5. For each unserved passenger:
       a. Retrieve their destination floor (pre-calculated during initialization).
       b. Check if the passenger is currently 'boarded' in the lift.
       c. If the passenger is 'boarded':
          i. Calculate the vertical distance between the current lift floor and the passenger's destination floor using the pre-calculated floor levels: `abs(level(lift_floor) - level(destin_floor))`.
          ii. Add this distance to the passenger's estimated cost (represents move actions).
          iii. Add 1 to the passenger's estimated cost for the 'depart' action.
       d. If the passenger is NOT 'boarded':
          i. Find the passenger's current origin floor from the state facts.
          ii. Calculate the vertical distance between the current lift floor and the passenger's origin floor: `abs(level(lift_floor) - level(origin_floor))`.
          iii. Add this distance to the passenger's estimated cost (represents move actions to pick them up).
          iv. Add 1 to the passenger's estimated cost for the 'board' action.
          v. Calculate the vertical distance between the passenger's origin floor and their destination floor: `abs(level(origin_floor) - level(destin_floor))`.
          vi. Add this distance to the passenger's estimated cost (represents move actions to transport them).
          vii. Add 1 to the passenger's estimated cost for the 'depart' action.
       e. Add the passenger's estimated cost to the total heuristic cost.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal destinations and floor levels.
        """
        super().__init__(task)

        # Extract destination floor for each passenger from the static facts.
        # The goal is a set of (served ?p) facts. We need the destination
        # for each passenger mentioned in the goal.
        # The destination facts are in the static set.
        self.passenger_destinations = {}
        for fact in self.static:
             if match(fact, "destin", "*", "*"):
                 _, passenger, floor = get_parts(fact)
                 self.passenger_destinations[passenger] = floor

        # Build the floor level mapping from 'above' facts.
        self.floor_levels = self._build_floor_levels(self.static)

    def _build_floor_levels(self, static_facts):
        """
        Parses 'above' facts to create a mapping from floor name to integer level.
        Assumes a linear sequence of floors.
        """
        above_relations = []
        all_floors = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                above_relations.append((f_lower, f_higher)) # Store as (lower, higher)
                all_floors.add(f_lower)
                all_floors.add(f_higher)

        if not all_floors:
             return {} # Handle case with no floors

        # Find the lowest floor: it's a floor that is 'lower' than another,
        # but no floor is 'lower' than it (i.e., it's not the second argument
        # in any 'above' predicate).
        floors_that_are_second_arg = {f_lower for f_lower, f_higher in above_relations}
        potential_lowest = all_floors - floors_that_are_second_arg

        if len(potential_lowest) != 1:
             # This indicates the 'above' facts don't form a single linear chain
             # with a unique lowest floor (e.g., multiple lowest floors, cycle, single floor).
             # For miconic, we expect a unique lowest floor.
             # If there's only one floor, above_relations is empty, all sets are empty.
             # potential_lowest becomes all_floors. If len(all_floors) == 1, this works.
             if len(all_floors) == 1 and not above_relations:
                 lowest_floor = all_floors.pop()
             else:
                 # Problem with floor structure definition
                 # print(f"Error: Could not determine a unique lowest floor from 'above' facts. Found {potential_lowest}")
                 # Return empty map or raise error? Returning empty map might lead to errors later.
                 # Raising error is safer for debugging problem definition.
                 raise ValueError(f"Could not determine a unique lowest floor from 'above' facts. Found {potential_lowest}")
        else:
             lowest_floor = potential_lowest.pop() # Get the unique lowest floor

        floor_levels = {}
        current_floor = lowest_floor
        level = 1

        # Build the mapping by following the 'above' chain
        # Create a dictionary for quick lookup of which floor is above another
        above_map = {f_lower: f_higher for f_lower, f_higher in above_relations}

        while current_floor is not None:
            if current_floor in floor_levels:
                 # Cycle detected or already processed - should not happen in linear structure
                 # print(f"Warning: Cycle or duplicate floor {current_floor} detected in floor levels.")
                 break # Stop processing
            floor_levels[current_floor] = level
            level += 1
            current_floor = above_map.get(current_floor) # Get the floor directly above

        # Basic check: ensure all floors found were assigned a level
        if len(floor_levels) != len(all_floors):
             # This indicates the 'above' facts don't form a single linear chain
             # covering all floors, or there's a cycle that wasn't fully traversed.
             # print("Warning: Not all floors included in the linear 'above' chain.")
             pass # Proceed with the levels found, might be incomplete but best effort

        return floor_levels


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all unserved passengers.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Check if the goal state is reached.
        # The goal is defined by task.goals, which is a frozenset of (served ?p) facts.
        if self.goals <= state:
             return 0

        total_cost = 0  # Initialize heuristic cost.

        # Find the current lift location.
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, lift_floor = get_parts(fact)
                break
        # Assuming lift_floor is always present in a valid state.
        if lift_floor is None:
             # This state is likely invalid or terminal (e.g., unsolvable)
             # A large heuristic value is appropriate.
             return float('inf') # Or some large number

        # Identify unserved passengers.
        # Passengers mentioned in goals are the ones that need to be served.
        # A passenger is unserved if (served ?p) is NOT in the state.
        goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}
        unserved_passengers = {p for p in goal_passengers if f"(served {p})" not in state}

        for passenger in unserved_passengers:
            # Get destination floor (pre-calculated).
            # Assuming all goal passengers have a destination defined in static facts.
            destin_floor = self.passenger_destinations.get(passenger)
            if destin_floor is None:
                 # Passenger in goal has no destination defined? Invalid problem.
                 # Assign large cost.
                 # print(f"Error: Destination not found for passenger {passenger} in static facts.")
                 return float('inf')

            # Check if the passenger is currently 'boarded'.
            is_boarded = f"(boarded {passenger})" in state

            if is_boarded:
                # Passenger is boarded, needs to go to destination and depart.
                # Cost = moves to destination + depart action
                # Ensure destination floor is in floor_levels (should be if problem is valid)
                if destin_floor not in self.floor_levels or lift_floor not in self.floor_levels:
                     # Invalid state or problem definition
                     # print(f"Error: Floor level not found for lift_floor ({lift_floor}) or destin_floor ({destin_floor}).")
                     return float('inf')
                cost_moves = abs(self.floor_levels[lift_floor] - self.floor_levels[destin_floor])
                cost_depart = 1
                total_cost += cost_moves + cost_depart
            else: # Passenger is not boarded
                # Passenger is at origin, needs pickup, transport, and depart.
                # Find the origin floor from the state.
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        _, _, origin_floor = get_parts(fact)
                        break

                if origin_floor is None:
                    # Passenger is unserved, not boarded, and not at origin.
                    # This state should not be reachable in a valid plan if the passenger
                    # started at an origin and hasn't been served.
                    # Assign a large penalty, as this state is likely invalid or unsolvable.
                    # print(f"Warning: Passenger {passenger} is unserved, not boarded, and not at origin.")
                    return float('inf') # Indicate unsolvable path

                # Ensure origin and destination floors are in floor_levels
                if origin_floor not in self.floor_levels or destin_floor not in self.floor_levels or lift_floor not in self.floor_levels:
                     # Invalid state or problem definition
                     # print(f"Error: Floor level not found for lift_floor ({lift_floor}), origin_floor ({origin_floor}), or destin_floor ({destin_floor}).")
                     return float('inf')

                # Cost = moves to origin + board action + moves origin->destin + depart action
                cost_to_origin = abs(self.floor_levels[lift_floor] - self.floor_levels[origin_floor])
                cost_board = 1
                cost_origin_to_destin = abs(self.floor_levels[origin_floor] - self.floor_levels[destin_floor])
                cost_depart = 1
                total_cost += cost_to_origin + cost_board + cost_origin_to_destin + cost_depart

        return total_cost
