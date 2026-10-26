from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace
    return fact.strip()[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to serve all unserved passengers.
    For each passenger, it estimates the cost to move the lift from its current
    location to the passenger's origin (if waiting), board the passenger,
    move the lift to the passenger's destination, and depart the passenger.
    The total heuristic is the sum of these individual passenger costs.

    # Assumptions
    - Floors are linearly ordered and named like 'f1', 'f2', 'f3', etc., where
      'f<i>' is above 'f<j>' if i > j.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic sums the estimated costs for each unserved passenger
      independently, ignoring potential synergies (e.g., picking up/dropping
      off multiple passengers on the same trip or at the same floor). This
      makes it potentially inadmissible but suitable for greedy best-first search.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the goal state
      and initial/static facts.
    - Builds a mapping from floor names (e.g., 'f1') to integer levels (e.g., 1)
      based on the 'above' predicates in the static facts, assuming the 'f<number>'
      naming convention implies the floor order.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. For each passenger that needs to be served (i.e., is in the goal state
       but not yet served in the current state):
       a. Check if the passenger is waiting at their origin floor (`(origin p f_origin)`).
          - If yes, calculate the estimated cost:
            - Move from current lift floor to `f_origin`: `abs(level(current_floor) - level(f_origin))`
            - Board action: +1
            - Move from `f_origin` to `f_destin`: `abs(level(f_origin) - level(f_destin))`
            - Depart action: +1
          - Add this total to the heuristic sum.
       b. Check if the passenger is boarded (`(boarded p)`).
          - If yes, calculate the estimated cost:
            - Move from current lift floor to `f_destin`: `abs(level(current_floor) - level(f_destin))`
            - Depart action: +1
          - Add this total to the heuristic sum.
    3. The total heuristic value is the sum of the estimated costs for all
       unserved passengers.
    4. If all passengers are served, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and floor information.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Store goal destinations for each passenger who needs to be served.
        self.goal_destinations = {}
        passengers_to_serve = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'}

        # Find the destination for each passenger who needs to be served
        # Destination facts are typically in the initial state.
        all_initial_and_static_facts = initial_state | static_facts
        for passenger_name in passengers_to_serve:
             destin_fact = next((fact for fact in all_initial_and_static_facts if get_parts(fact)[:2] == ['destin', passenger_name]), None)
             if destin_fact:
                 self.goal_destinations[passenger_name] = get_parts(destin_fact)[2]
             # Note: If a passenger needs serving but no destination is found,
             # they won't be added to goal_destinations. This means they won't
             # contribute to the heuristic, which is a reasonable behavior for
             # potentially ill-defined problems, assuming valid problems provide destinations.


        # Build floor mapping: floor_name -> level (integer)
        self.floor_to_level = {}
        self.level_to_floor = {}
        floor_names = set()
        # Collect all floor names from 'above' predicates
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                floor_names.add(parts[1])
                floor_names.add(parts[2])

        # Also collect floor names from initial state facts like (lift-at f) or (origin p f) or (destin p f)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] in ['lift-at', 'origin', 'destin']:
                  if len(parts) > 1: # Ensure there's an argument
                      # The floor is usually the last argument for these predicates
                      floor_names.add(parts[-1])


        # Assuming floor names are 'f<number>' and order is by number
        # Extract numbers and sort floors
        try:
            sorted_floors = sorted(list(floor_names), key=lambda f: int(re.search(r'\d+', f).group()))
            for level, floor_name in enumerate(sorted_floors):
                self.floor_to_level[floor_name] = level
                self.level_to_floor[level] = floor_name
        except (AttributeError, ValueError):
             # Handle cases where floor names might not match 'f<number>' pattern or number extraction fails
             print("Warning: Floor names do not match 'f<number>' pattern or number extraction failed. Floor mapping may be incomplete or incorrect.")
             # If floor mapping fails, distance calculation will fail.
             # A robust heuristic would need a proper graph traversal here.
             # For this problem, we proceed assuming the pattern holds for solvable instances.
             # If it fails, the heuristic will likely return errors or inf later.
             self.floor_to_level = {} # Clear potentially partial mapping
             self.level_to_floor = {}


    def __call__(self, node):
        """
        Estimate the minimum cost to serve all remaining passengers.
        """
        state = node.state  # Current world state.

        # If the goal is already reached, heuristic is 0.
        # This check also handles the case where goal_destinations is empty (no passengers to serve)
        if self.goals <= state:
             return 0

        # If floor mapping failed during init, we cannot calculate distances.
        if not self.floor_to_level:
             # Cannot compute heuristic without floor levels
             return float('inf') # Indicate unsolvable state if floor structure is unknown

        total_cost = 0  # Initialize action cost counter.

        # Find the current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # Should not happen in a valid state
             return float('inf') # Indicate invalid state

        current_level = self.floor_to_level.get(current_lift_floor)
        if current_level is None:
             # Should not happen if floor mapping is correct and lift is at a known floor
             return float('inf') # Indicate unsolvable or invalid state


        # Iterate through passengers who need to be served according to the goal
        for passenger, goal_dest_floor in self.goal_destinations.items():
            # Check if the passenger is already served
            if f'(served {passenger})' in state:
                continue # This passenger is done

            # Passenger is not served. Check their current status.
            is_waiting = False
            waiting_floor = None
            is_boarded = False

            # Find the passenger's current status and location
            # We only need to find one of origin or boarded facts for a non-served passenger
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'origin' and parts[1] == passenger:
                    waiting_floor = parts[2]
                    is_waiting = True
                    break # Found origin
                elif parts[0] == 'boarded' and parts[1] == passenger:
                    is_boarded = True
                    break # Found boarded

            # Calculate cost based on passenger status
            if is_waiting:
                # Passenger is waiting at origin
                origin_level = self.floor_to_level.get(waiting_floor)
                dest_level = self.floor_to_level.get(goal_dest_floor)

                if origin_level is None or dest_level is None:
                     # Should not happen if floor mapping is correct
                     return float('inf')

                # Cost to pick up: move to origin + board action
                cost_to_pickup = abs(current_level - origin_level) + 1

                # Cost to deliver: move from origin to destination + depart action
                # Movement cost is from the floor where the passenger was boarded (origin_level)
                cost_to_deliver = abs(origin_level - dest_level) + 1

                total_cost += cost_to_pickup + cost_to_deliver

            elif is_boarded:
                # Passenger is boarded
                dest_level = self.floor_to_level.get(goal_dest_floor)

                if dest_level is None:
                     # Should not happen
                     return float('inf')

                # Cost to deliver: move to destination + depart action
                cost_to_deliver = abs(current_level - dest_level) + 1

                total_cost += cost_to_deliver

            # If passenger is neither waiting nor boarded, and not served,
            # this indicates an inconsistency in the state relative to the goal.
            # We assume valid states where non-served passengers are either waiting or boarded.
            # If somehow a passenger is not served, not waiting, and not boarded,
            # they won't contribute to the heuristic, which is arguably incorrect
            # but reflects an invalid state representation.

        return total_cost
