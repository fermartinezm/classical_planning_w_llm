# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It calculates the cost for each unserved passenger individually and sums them up.
    The individual cost includes the lift movement to the passenger's location
    (origin if waiting, current lift floor if boarded), the board/depart actions,
    and the lift movement from origin to destination (if waiting). This approach
    is non-admissible as it sums costs for passengers independently, ignoring
    potential batching of movements and actions.

    # Assumptions
    - All actions have a unit cost.
    - The floor structure is linear, defined by the 'above' predicate, allowing
      floors to be ordered and indexed.
    - Passengers are always in one of three states: waiting at their origin,
      boarded in the lift, or served at their destination.
    - The state representation is complete and consistent with the domain rules.

    # Heuristic Initialization
    - Parses 'above' facts from static information to create a mapping from floor
      objects to integer indices, representing their vertical order from lowest (0)
      to highest.
    - Parses 'destin' facts from the initial state to map each passenger to their
      destination floor.
    - Extracts the set of all passengers whose 'served' predicate is part of the
      goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state and identify the lift's current floor by finding the
       fact '(lift-at ?f)'.
    2. Identify the set of all passengers that have not yet been served by checking
       if '(served ?p)' is present in the current state for each passenger in the goal.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each unserved passenger:
       a. Retrieve the passenger's destination floor using the pre-calculated mapping
          from initialization.
       b. Check if the passenger is currently boarded by looking for the fact
          '(boarded ?p)' in the current state.
       c. If the passenger is boarded:
          - Add 1 to the total cost, representing the 'depart' action needed.
          - Calculate the vertical distance (number of floors) between the lift's
            current floor and the passenger's destination floor using the floor indices.
            Add this distance to the total cost, representing the movement needed
            to reach the drop-off floor.
       d. If the passenger is not boarded (implying they are waiting at their origin):
          - Find the passenger's origin floor by searching for the fact
            '(origin ?p ?f)' in the current state.
          - Add 2 to the total cost, representing the 'board' and 'depart' actions needed.
          - Calculate the vertical distance between the lift's current floor and the
            passenger's origin floor. Add this distance to the total cost, representing
            the movement needed to reach the pick-up floor.
          - Calculate the vertical distance between the passenger's origin floor and
            their destination floor. Add this distance to the total cost, representing
            the movement needed while the passenger is boarded.
    5. Return the total accumulated cost as the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the set of all passengers from the task definition.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state # Need destin facts from here
        static_facts = task.static

        # 1. Build floor_to_index mapping from 'above' facts
        above_facts_parts = [get_parts(fact) for fact in static_facts if match(fact, "above", "*", "*")]

        # Collect all floor objects mentioned in above facts
        all_floors_set = set()
        for _, f_higher, f_lower in above_facts_parts:
            all_floors_set.add(f_higher)
            all_floors_set.add(f_lower)

        ordered_floors = []
        self.floor_to_index = {}

        if all_floors_set:
            # Find the lowest floor: the one that is never the 'higher' floor in any 'above' fact
            higher_floors_in_above = {f_higher for _, f_higher, f_lower in above_facts_parts}
            potential_lowest = all_floors_set - higher_floors_in_above

            if len(potential_lowest) == 1:
                lowest_floor = potential_lowest.pop()
                # Build the ordered list starting from the lowest floor
                ordered_floors.append(lowest_floor)
                current_f = lowest_floor
                # Build a map from lower to higher floor for direct adjacency
                lower_to_higher = {f_lower: f_higher for f_higher, f_lower in [(h, l) for _, h, l in above_facts_parts]}
                while current_f in lower_to_higher:
                    current_f = lower_to_higher[current_f]
                    ordered_floors.append(current_f)
            elif len(all_floors_set) == 1:
                 # Case with only one floor
                 ordered_floors = list(all_floors_set)
            else:
                 # Fallback for complex or insufficient 'above' structure.
                 # Assume alphabetical order represents vertical order (f1 < f2 < ...)
                 # This is a strong assumption but common in simple benchmarks.
                 ordered_floors = sorted(list(all_floors_set))
                 # print(f"Warning: Complex or insufficient 'above' facts. Assuming alphabetical floor order: {ordered_floors}")


            self.floor_to_index = {f: i for i, f in enumerate(ordered_floors)}

        # 2. Build passenger_to_destin mapping from 'destin' facts in initial state
        self.passenger_to_destin = {}
        # Destin facts are in the initial state, not static facts.
        for fact in task.initial_state:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_to_destin[passenger] = floor

        # 3. Extract all passengers from goal conditions
        self.all_passengers = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 self.all_passengers.add(passenger)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find the current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # If lift location is not found, state is likely invalid or goal is reached (h=0)
        # A goal state should have all passengers served, which is checked below.
        # If lift-at is truly missing in a non-goal state, something is wrong.
        # Returning a large value indicates an unreachable or invalid state for search.
        # Check if there are any unserved passengers before returning inf for missing lift
        if current_lift_floor is None and any(f"(served {p})" not in state for p in self.all_passengers):
             # print("Warning: Lift location not found in non-goal state.")
             return float('inf') # Should not happen in valid states

        # If current_lift_floor is None here, it means all passengers are served (goal state)
        # or there are no passengers in the problem. In this case, the loop below
        # will correctly result in total_cost = 0.

        current_floor_index = None
        if current_lift_floor is not None:
            current_floor_index = self.floor_to_index.get(current_lift_floor)
            # If lift is at an unknown floor, state is invalid.
            if current_floor_index is None:
                 # print(f"Warning: Lift at unknown floor {current_lift_floor}")
                 return float('inf') # Should not happen in valid states


        total_cost = 0

        # Check each passenger
        for passenger in self.all_passengers:
            # Check if the passenger is already served
            if f"(served {passenger})" in state:
                continue # Passenger is served, contributes 0 to heuristic

            # Passenger is unserved. Calculate their individual contribution.
            destin_floor = self.passenger_to_destin.get(passenger)
            if destin_floor is None:
                 # Passenger destination not found. Should not happen based on problem definition.
                 # print(f"Warning: Destination not found for passenger {passenger}")
                 return float('inf') # Should not happen in valid problems

            destin_floor_index = self.floor_to_index.get(destin_floor)
            if destin_floor_index is None:
                 # Destination floor is unknown. Should not happen.
                 # print(f"Warning: Destination floor {destin_floor} unknown for passenger {passenger}")
                 return float('inf') # Should not happen in valid problems


            # Check if the passenger is boarded
            if f"(boarded {passenger})" in state:
                # Passenger is boarded, needs to travel to destination and depart
                total_cost += 1 # Depart action
                # Add travel cost from current lift floor to destination
                # current_floor_index must be valid here because we checked for missing lift above
                total_cost += abs(current_floor_index - destin_floor_index)
            else:
                # Passenger is waiting at origin, needs board, travel, depart
                # Find origin floor
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        _, p, origin_floor = get_parts(fact)
                        break

                if origin_floor is None:
                    # Passenger is unserved, not boarded, and not at origin? Invalid state.
                    # This check assumes the state is always consistent.
                    # print(f"Warning: Unserved, unboarded passenger {passenger} not found at any origin.")
                    return float('inf') # Should not happen in valid states

                origin_floor_index = self.floor_to_index.get(origin_floor)
                if origin_floor_index is None:
                     # Origin floor is unknown. Should not happen.
                     # print(f"Warning: Origin floor {origin_floor} unknown for passenger {passenger}")
                     return float('inf') # Should not happen in valid states


                total_cost += 2 # Board and Depart actions
                # Add travel cost from current lift floor to origin
                # current_floor_index must be valid here because we checked for missing lift above
                total_cost += abs(current_floor_index - origin_floor_index)
                # Add travel cost from origin to destination
                total_cost += abs(origin_floor_index - destin_floor_index)

        return total_cost
