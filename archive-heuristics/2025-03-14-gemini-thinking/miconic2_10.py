from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(origin p1 f6)' it returns ['p1', 'f6'].
    """
    fact_content = fact_str[1:-1].split()
    return fact_content[1:]  # Return objects, skip predicate name

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    by considering the necessary steps for each passenger: moving the lift to the
    origin floor, boarding, moving to the destination floor, and departing.

    # Assumptions:
    - The heuristic assumes that for each passenger, the lift will first go to their
      origin floor, then to their destination floor.
    - It simplifies the floor movement cost by counting each move between adjacent floors as one action.
    - It does not consider optimizing lift movements for multiple passengers simultaneously.

    # Heuristic Initialization
    - Extracts static information about passenger origins, destinations, and floor order
      from the task's static facts.
    - Precomputes a mapping of floor names to their 'index' based on the 'above' predicates
      to estimate the cost of moving between floors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:
    1. Initialize the heuristic value to 0.
    2. Identify all passengers from the static facts.
    3. For each passenger:
       a. Check if the passenger is already served in the current state. If yes, no cost is added.
       b. If not served, check if the passenger is boarded.
       c. If not boarded:
          i.  Determine the passenger's origin and destination floors from static facts.
          ii. Determine the current lift floor from the state.
          iii.Estimate the cost to move the lift from its current floor to the passenger's origin floor.
          iv. Add 1 action for 'board'.
          v. Estimate the cost to move the lift from the origin floor to the destination floor.
          vi.Add 1 action for 'depart'.
       d. If boarded:
          i.  Determine the passenger's destination floor from static facts.
          ii. Determine the current lift floor from the state.
          iii.Estimate the cost to move the lift from the current floor to the passenger's destination floor.
          iv. Add 1 action for 'depart'.
    4. Sum up the costs for all passengers to get the total heuristic value.
    5. Return the total heuristic value.

    The cost of moving between floors is estimated by the absolute difference in their
    'indices', which are derived from the 'above' predicates.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic.
        Extracts passenger origins, destinations, and floor order from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.floor_above_relations = {}
        self.floors_list = []

        # Extract origin and destination for each passenger
        for fact in static_facts:
            if fact.startswith('(origin'):
                person, floor = get_objects_from_fact(fact)
                self.passenger_origins[person] = floor
            elif fact.startswith('(destin'):
                person, floor = get_objects_from_fact(fact)
                self.passenger_destinations[person] = floor
            elif fact.startswith('(above'):
                f1, f2 = get_objects_from_fact(fact)
                self.floor_above_relations[f1] = f2
                if f1 not in self.floors_list:
                    self.floors_list.append(f1)
                if f2 not in self.floors_list:
                    self.floors_list.append(f2)

        # Determine floor order based on 'above' relations. Simple ordering by name for now.
        self.floors_list = sorted(list(set(self.floors_list)), key=lambda f: int(f[1:]))
        self.floor_indices = {floor: index for index, floor in enumerate(self.floors_list)}


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        current_lift_floor = None
        for fact in state:
            if fact.startswith('(lift-at'):
                current_lift_floor = get_objects_from_fact(fact)[0]
                break
        if current_lift_floor is None:
            return float('inf') # Should not happen in valid states, but handle for robustness

        passengers = set(self.passenger_origins.keys())

        for passenger in passengers:
            if f'(served {passenger})' in state:
                continue # Passenger already served, no cost

            if f'(boarded {passenger})' not in state:
                origin_floor = self.passenger_origins[passenger]
                destination_floor = self.passenger_destinations[passenger]

                origin_floor_index = self.floor_indices.get(origin_floor, -1)
                destination_floor_index = self.floor_indices.get(destination_floor, -1)
                current_lift_floor_index = self.floor_indices.get(current_lift_floor, -1)

                if origin_floor_index != -1 and destination_floor_index != -1 and current_lift_floor_index != -1:
                    heuristic_value += abs(current_lift_floor_index - origin_floor_index) # Move to origin
                    heuristic_value += 1 # Board
                    heuristic_value += abs(origin_floor_index - destination_floor_index) # Move to destin
                    heuristic_value += 1 # Depart
                else:
                    return float('inf') # Handle cases where floor indices are not found (error case)

            else: # Passenger is boarded
                destination_floor = self.passenger_destinations[passenger]
                destination_floor_index = self.floor_indices.get(destination_floor, -1)
                current_lift_floor_index = self.floor_indices.get(current_lift_floor, -1)

                if destination_floor_index != -1 and current_lift_floor_index != -1:
                    heuristic_value += abs(current_lift_floor_index - destination_floor_index) # Move to destin
                    heuristic_value += 1 # Depart
                else:
                    return float('inf') # Handle cases where floor indices are not found (error case)

        return heuristic_value
