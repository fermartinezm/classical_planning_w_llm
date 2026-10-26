from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(predicate arg1 arg2)" -> ["predicate", "arg1", "arg2"]
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort by summing the number of
    pending board actions, pending depart actions, and an estimated cost
    for lift movement based on the range of floors that need visiting.

    # Assumptions
    - Each unserved passenger requires one board action (if waiting at origin)
      and one depart action (if boarded).
    - The lift must visit the origin floor for waiting passengers and the
      destination floor for boarded passengers.
    - The cost of lift movement is estimated by the distance the lift needs
      to travel to cover the range of required floors.
    - Floor names are in the format 'f<number>', where number indicates the level.
    - Passenger destinations are static and determined from the initial state.
    - States are well-formed (e.g., lift location is always present, passengers
      are always in one of (origin), (boarded), or (served) states until served).

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the initial state
      facts ('destin' predicate).
    - Extracts all floor names from static facts ('above' predicate) and initial
      state facts ('lift-at', 'origin', 'destin') and creates a mapping from
      floor name to its numerical level. This mapping is used to calculate
      floor distances.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Build temporary lookups for passengers waiting at origin (`origin_facts`)
       and passengers currently boarded (`boarded_passengers`) by iterating
       through the state facts. Also find the current lift floor.
    2. Initialize counters: `num_board_needed = 0`, `num_depart_needed = 0`.
    3. Initialize a set to store required floor stops (by floor name):
       `required_stop_floors = set()`.
    4. Iterate through all known passengers (from initialization).
    5. For each passenger `p`:
       a. Check if `(served p)` is NOT in the state.
       b. If unserved:
          i. Retrieve `F_dest` for `p`.
          ii. If `p` is in `origin_facts`: Increment `num_board_needed`, add
              `origin_facts[p]` and `F_dest` to `required_stop_floors`.
          iii. Elif `p` is in `boarded_passengers`: Increment `num_depart_needed`,
               add `F_dest` to `required_stop_floors`.
    6. Calculate estimated movement cost:
       a. If `required_stop_floors` is empty, movement cost is 0.
       c. Else:
          i. Get the numerical level for each floor in `required_stop_floors`
             using the floor-to-level mapping.
          ii. Find the minimum (`min_level`) and maximum (`max_level`) levels
              among the required stops.
          iii. Get the numerical level for the current lift floor (`current_level`).
          iv. The movement cost is estimated as the distance to reach the closest
              end of the required floor range plus the distance to traverse the
              entire range: `min(abs(current_level - min_level), abs(current_level - max_level)) + (max_level - min_level)`.
    7. The heuristic value is `num_board_needed` + `num_depart_needed` + movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and floor levels.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract passenger destinations from the initial state
        self.passenger_destinations = {}
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "destin" and len(parts) == 3:
                self.passenger_destinations[parts[1]] = parts[2]

        # Extract floor names and map them to numerical levels
        self.floor_to_level = {}
        floor_names = set()
        # Collect all floor names mentioned in 'above' facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "above" and len(parts) == 3:
                floor_names.add(parts[1])
                floor_names.add(parts[2])

        # Also collect floors mentioned in initial state facts
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Look for floors in predicates that involve locations
             if parts[0] in ["lift-at", "origin", "destin"] and len(parts) > 1:
                  # The floor is typically the last argument for these predicates
                  floor_names.add(parts[-1])


        # Extract level number from floor name (assuming 'f<number>')
        # and create the mapping
        for floor_name in floor_names:
             match = re.match(r'f(\d+)', floor_name)
             if match:
                 self.floor_to_level[floor_name] = int(match.group(1))
             else:
                 # This case should ideally not happen in valid miconic instances
                 # print(f"Warning: Unexpected floor name format: {floor_name}")
                 pass # Assume format is always f<number>

        # Add any floors from destinations that weren't in the collected set
        # (e.g., if 'above' facts don't list all floors or initial state is minimal)
        for dest_floor in self.passenger_destinations.values():
             if dest_floor not in self.floor_to_level:
                  match = re.match(r'f(\d+)', dest_floor)
                  if match:
                      self.floor_to_level[dest_floor] = int(match.group(1))
                  else:
                       # This would indicate a problem with the instance definition
                       # print(f"Error: Destination floor {dest_floor} not mapped to level.")
                       pass # Assume valid PDDL


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            An integer estimate of the remaining cost to reach the goal.
        """
        state = node.state

        # Build quick lookups for origin and boarded facts
        origin_facts = {} # {passenger: floor}
        boarded_passengers = set()
        current_lift_floor = None

        # Use a set for faster membership checks if state is large, though frozenset is usually fine
        state_set = set(state)

        for fact in state_set:
            parts = get_parts(fact)
            if parts[0] == "origin" and len(parts) == 3:
                origin_facts[parts[1]] = parts[2]
            elif parts[0] == "boarded" and len(parts) == 2:
                boarded_passengers.add(parts[1])
            elif parts[0] == "lift-at" and len(parts) == 2:
                 current_lift_floor = parts[1]

        # Ensure lift location is found (should always be in a valid state)
        if current_lift_floor is None:
             # If the goal is reached, h=0. Otherwise, state is likely invalid.
             if self.goals.issubset(state_set):
                  return 0
             else:
                  # print("Error: Lift location not found in state.")
                  return float('inf') # Return infinity for invalid states


        num_board_needed = 0
        num_depart_needed = 0
        required_stop_floors = set()

        # Iterate through all passengers we know about
        for passenger, destination_floor in self.passenger_destinations.items():
            # Check if the passenger is already served
            if f"(served {passenger})" not in state_set:
                # Passenger is unserved

                if passenger in origin_facts:
                    # Passenger is waiting at origin
                    origin_floor = origin_facts[passenger]
                    num_board_needed += 1
                    required_stop_floors.add(origin_floor)
                    required_stop_floors.add(destination_floor)
                elif passenger in boarded_passengers:
                    # Passenger is boarded
                    num_depart_needed += 1
                    required_stop_floors.add(destination_floor)
                # else: Unserved but neither origin nor boarded - implies passenger is lost
                # or state is inconsistent. Assuming valid states, this shouldn't happen.


        # Calculate estimated movement cost
        movement_cost = 0
        if required_stop_floors:
            # Ensure all required floors have a mapped level
            required_stop_levels = set()
            for f in required_stop_floors:
                 if f in self.floor_to_level:
                      required_stop_levels.add(self.floor_to_level[f])
                 else:
                      # This indicates a floor in state wasn't mapped during init
                      # print(f"Error: Required stop floor {f} not mapped to level.")
                      # This state is likely unsolvable or heuristic is invalid for it.
                      return float('inf') # Cannot compute heuristic

            if required_stop_levels: # Should not be empty if required_stop_floors was not empty
                min_level = min(required_stop_levels)
                max_level = max(required_stop_levels)

                if current_lift_floor in self.floor_to_level:
                    current_level = self.floor_to_level[current_lift_floor]
                    # Estimate moves to cover the range of required floors
                    # Go from current level to the closest end of the range, then traverse the range
                    movement_cost = min(abs(current_level - min_level), abs(current_level - max_level)) + (max_level - min_level)
                else:
                     # Current lift floor not mapped - indicates error
                     # print(f"Error: Current lift floor {current_lift_floor} not mapped to level.")
                     return float('inf')


        heuristic_value = num_board_needed + num_depart_needed + movement_cost

        # The heuristic value is 0 iff num_board_needed=0, num_depart_needed=0, and movement_cost=0.
        # This happens iff there are no unserved passengers (num_board_needed=0, num_depart_needed=0)
        # and thus no required stops (required_stop_floors is empty, movement_cost=0).
        # No unserved passengers means all passengers are served, which is the goal state.
        # So, h=0 iff goal state.

        return heuristic_value
