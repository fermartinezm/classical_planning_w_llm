from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The heuristic class
# class miconicHeuristic(Heuristic): # Uncomment this line in the actual environment
# Replace with the actual class definition expected by the framework
# For standalone testing, keep the dummy Heuristic class or remove inheritance
class miconicHeuristic: # Using this for generating the final code block
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers. It calculates the cost for each unserved passenger independently
    by summing the minimum moves needed for the lift to reach their origin,
    the board action, the minimum moves to reach their destination, and the
    depart action. If a passenger is already boarded, it only counts the moves
    to the destination and the depart action. The total heuristic is the sum
    of these individual costs.

    # Assumptions
    - The floors form a single linear sequence ordered by the `above` predicate,
      where `(above f_higher f_lower)` means `f_higher` is immediately above `f_lower`.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic relaxes the constraint that the lift can only be at one place
      at a time and can only carry passengers currently boarded. It effectively
      calculates the cost for each passenger as if they were the only one,
      ignoring potential synergies (picking up/dropping off multiple passengers
      at the same floor). This makes it non-admissible but potentially good
      for greedy search.

    # Heuristic Initialization
    - Parses the static facts to determine the linear order of floors based on
      the `above` predicate and creates a mapping from floor name to its index
      in the ordered list (starting from the bottom floor at index 0).
    - Parses the goal conditions to store the destination floor for each passenger.
    - Identifies all passengers involved in the problem (those needing to be served).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift from the state.
    2. Identify which passengers are already served from the state.
    3. Identify which passengers are currently boarded in the lift from the state.
    4. Identify the origin floor for passengers who are neither served nor boarded
       from the state.
    5. Initialize the total heuristic cost to 0.
    6. Iterate through all passengers known in the problem (those in the goal):
       a. If the passenger is already served, add 0 to the total cost.
       b. If the passenger is currently boarded:
          i. Get the passenger's destination floor (pre-calculated in init).
          ii. Calculate the number of floor moves required from the lift's current
              floor to the passenger's destination floor using the floor index map.
          iii. Add this number of moves + 1 (for the depart action) to the total cost.
       c. If the passenger is waiting at their origin floor:
          i. Get the passenger's origin floor from the state.
          ii. Get the passenger's destination floor (pre-calculated in init).
          iii. Calculate the number of floor moves required from the lift's current
               floor to the passenger's origin floor.
          iv. Calculate the number of floor moves required from the passenger's
              origin floor to their destination floor.
          v. Add the first move cost + 1 (for board) + the second move cost + 1
             (for depart) to the total cost.
    7. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        self.static = task.static
        # Assuming task.objects contains all objects including passengers and floors
        # self.objects = task.objects # Not strictly needed for this heuristic logic

        # 1. Build floor order and index map from static facts
        above_map = {} # Maps floor_lower -> floor_higher
        all_floors = set()
        floors_on_left_of_above = set() # Floors that are 'above' something
        floors_on_right_of_above = set() # Floors that are 'below' something

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                f_above, f_below = parts[1], parts[2]
                above_map[f_below] = f_above # f_above is immediately above f_below
                floors_on_left_of_above.add(f_above)
                floors_on_right_of_above.add(f_below)
                all_floors.add(f_above)
                all_floors.add(f_below)

        self.floor_order = []
        self.floor_to_index = {}

        if all_floors:
            # Find the bottom floor: appears on the right of 'above' but never on the left
            potential_bottoms = floors_on_right_of_above - floors_on_left_of_above
            bottom_floor = None

            if len(potential_bottoms) == 1:
                 bottom_floor = list(potential_bottoms)[0]
            elif len(all_floors) == 1:
                 # Case with only one floor and no 'above' facts
                 bottom_floor = list(all_floors)[0]
            # Note: If len(all_floors) > 1 and len(potential_bottoms) != 1,
            # the 'above' facts don't form a simple linear chain.
            # The heuristic relies on this linear structure. If not found,
            # floor_order and floor_to_index will remain empty, and the fallback
            # heuristic in __call__ will be used.

            if bottom_floor:
                current = bottom_floor
                index = 0
                while current is not None:
                    self.floor_order.append(current)
                    self.floor_to_index[current] = index
                    index += 1
                    current = above_map.get(current)

        # 2. Store destination floors for each passenger and identify all passengers
        self.destin_map = {}
        self.all_passengers = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served':
                passenger = parts[1]
                self.all_passengers.add(passenger)
                # Find the destination for this passenger from static facts
                for static_fact in self.static:
                    static_parts = get_parts(static_fact)
                    if static_parts and static_parts[0] == 'destin' and static_parts[1] == passenger:
                        self.destin_map[passenger] = static_parts[2]
                        break # Found destination for this passenger

        # Ensure all passengers in destin_map are also in all_passengers
        # (redundant if goal is always served, but safe)
        self.all_passengers.update(self.destin_map.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Fallback heuristic if floor order could not be determined (e.g., complex 'above' relations)
        # In a valid miconic problem, floor order should be linear.
        # If not, this heuristic is ill-suited. A simple fallback is number of unserved passengers.
        if not self.floor_order:
             # Check if all passengers are served in the current state
             all_served = True
             for p in self.all_passengers:
                 if f'(served {p})' not in state:
                     all_served = False
                     break
             if all_served: return 0 # Goal state reached

             # If floor order is missing and not goal state, return count of unserved passengers
             # This is a very weak heuristic but prevents errors.
             return sum(1 for p in self.all_passengers if f'(served {p})' not in state)


        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # This indicates an invalid state (lift must be somewhere)
             # Return infinity as it's likely an unsolvable path from here
             return float('inf') # Should not be reachable in valid problems

        # 2. Get sets of served and boarded passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        # 3. Get origin floors for passengers waiting at origin
        origin_map = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'origin':
                passenger, floor = parts[1], parts[2]
                origin_map[passenger] = floor

        total_cost = 0

        # 6. Iterate through all passengers that need to be served (from goals)
        for passenger in self.all_passengers:
            # a. If served, cost is 0 for this passenger
            if passenger in served_passengers:
                continue

            # Ensure passenger has a known destination (should be true if in self.all_passengers)
            destin_floor = self.destin_map.get(passenger)
            if destin_floor is None:
                 # This passenger is in the goal but has no destination in static facts?
                 # Indicates a problem definition error. Cannot calculate heuristic.
                 # Return infinity or skip? Skipping might mask the error. Infinity is safer.
                 return float('inf') # Indicates problem with static/goal definition

            # b. If boarded
            if passenger in boarded_passengers:
                # i. Get destination floor (already have it)

                # ii. Calculate floor moves from current lift floor to destination
                current_floor_index = self.floor_to_index.get(current_lift_floor)
                destin_floor_index = self.floor_to_index.get(destin_floor)

                if current_floor_index is None or destin_floor_index is None:
                     # Indicates a floor name in state/destin not found in the derived floor order.
                     # Problem definition error or parsing error.
                     return float('inf') # Indicates problem with floor names

                dist_to_destin = abs(current_floor_index - destin_floor_index)

                # iii. Add cost: moves + depart action
                total_cost += dist_to_destin + 1

            # c. If waiting at origin floor
            elif passenger in origin_map:
                origin_floor = origin_map[passenger]

                # iii. Calculate moves from current lift floor to origin
                current_floor_index = self.floor_to_index.get(current_lift_floor)
                origin_floor_index = self.floor_to_index.get(origin_floor)
                destin_floor_index = self.floor_to_index.get(destin_floor) # Already got destin_floor

                if current_floor_index is None or origin_floor_index is None or destin_floor_index is None:
                     # Indicates a floor name in state/origin/destin not found in the derived floor order.
                     # Problem definition error or parsing error.
                     return float('inf') # Indicates problem with floor names

                dist_to_origin = abs(current_floor_index - origin_floor_index)

                # iv. Calculate moves from origin floor to destination
                dist_origin_to_destin = abs(origin_floor_index - destin_floor_index)

                # v. Add cost: moves to origin + board + moves to destin + depart
                total_cost += dist_to_origin + 1 + dist_origin_to_destin + 1

            # else: Passenger is not served, not boarded, and not at origin?
            # This state should not be reachable in a valid miconic problem progression.
            # A passenger is either at origin, boarded, or served.
            # If this state occurs, something is wrong. Return infinity.
            # This check might be too strict if the state representation can be sparse,
            # but PDDL states usually explicitly list facts that are true.
            # Let's assume valid states adhere to the domain predicates.
            # If a passenger is in self.all_passengers but not served, boarded, or in origin_map,
            # it implies an invalid state representation.
            # For robustness, we could add a check here, but it might hide valid sparse states
            # if the framework doesn't explicitly list all origins/boarded facts for *all* passengers.
            # Assuming the state lists all relevant facts for unserved passengers.

        # 7. Return total cost
        return total_cost
