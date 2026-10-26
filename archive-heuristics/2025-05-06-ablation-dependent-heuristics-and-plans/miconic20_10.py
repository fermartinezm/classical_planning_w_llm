from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all passengers.
    It calculates the cost for each passenger independently, summing the estimated
    lift movement actions and the board/depart actions needed for that passenger.
    It assumes the lift travels directly from its current location to the passenger's
    origin, then to their destination, without considering optimizing paths for multiple
    passengers. This makes it a non-admissible heuristic suitable for greedy best-first search.

    # Assumptions
    - Floors are ordered based on the `above` predicates, where `(above f_lower f_higher)`
      means `f_higher` is immediately above `f_lower`.
    - Each passenger requires a board action (if not already boarded) and a depart action
      (if not already served).
    - Lift movement cost between adjacent floors is 1.
    - The heuristic sums the costs for each passenger as if they were transported
      individually, ignoring potential efficiencies from batching passengers.
    - All passengers mentioned in the goal state have a corresponding `destin` fact
      in the static information or initial state.
    - If a passenger is not served and not boarded, they are assumed to be at their
      origin floor as specified in the current state's `origin` fact.

    # Heuristic Initialization
    - Parses `above` predicates from static facts to establish the floor order and
      create mappings between floor names and numerical indices (1-based).
    - Parses `destin` predicates from static facts to store the destination floor
      for each passenger.
    - Collects the set of all passengers that need to be served based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Identify the current floor of the lift from the state facts.
    3. Identify the set of passengers who are already served in the current state.
    4. Identify the set of passengers who are currently boarded in the current state.
    5. Identify the set of `origin` facts in the current state to find waiting passengers' locations.
    6. Initialize the total heuristic cost to 0.
    7. Iterate through each passenger that needs to be served (identified during initialization from the goal state).
    8. For the current passenger:
       a. If the passenger is in the set of served passengers, they contribute 0 to the heuristic. Continue to the next passenger.
       b. If the passenger is not served:
          i. Retrieve the passenger's destination floor from the pre-calculated destinations map. Get its numerical index.
          ii. Check if the passenger is in the set of boarded passengers.
          iii. If the passenger is boarded:
              - Add 1 to the total heuristic (representing the necessary `depart` action).
              - Calculate the estimated lift movement cost from the current lift floor to the passenger's destination floor (absolute difference in floor indices). Add this cost to the total heuristic.
          iv. If the passenger is not boarded (implying they are waiting at their origin):
              - Find the passenger's origin floor by searching the `origin` facts in the current state. Get its numerical index.
              - Add 2 to the total heuristic (representing the necessary `board` and `depart` actions).
              - Calculate the estimated lift movement cost from the current lift floor to the passenger's origin floor (absolute difference in floor indices). Add this cost to the total heuristic.
              - Calculate the estimated lift movement cost from the passenger's origin floor to their destination floor (absolute difference in floor indices). Add this cost to the total heuristic.
    9. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, floor mappings,
        passenger destinations, and the set of all passengers that need serving.
        """
        self.goals = task.goals  # Goal conditions (used to identify all passengers to serve)
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Build floor order and mappings
        self.floor_to_idx = {}
        self.idx_to_floor = {}
        above_relations = set()
        all_floors_in_above = set()
        is_lower_in_above = set()
        is_higher_in_above = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_lower, f_higher = parts[1], parts[2]
                above_relations.add((f_lower, f_higher))
                all_floors_in_above.add(f_lower)
                all_floors_in_above.add(f_higher)
                is_lower_in_above.add(f_lower)
                is_higher_in_above.add(f_higher)

        # Collect all floors mentioned in the initial state and goals as well
        all_floors_in_task = set(all_floors_in_above)
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] in ["lift-at", "origin", "destin"]:
                if len(parts) > 1:
                    all_floors_in_task.add(parts[-1]) # Last arg is floor

        # Find the bottom floor: a floor that is a 'lower' floor in some 'above' relation,
        # but never a 'higher' floor in any 'above' relation.
        # Or, if no 'above' relations, just pick one floor if any exist.
        bottom_floor = None
        if all_floors_in_task:
            potential_bottom_floors = all_floors_in_task - is_higher_in_above
            if potential_bottom_floors:
                 # Prefer a floor that is also a 'lower' floor if possible
                 candidates = sorted(list(potential_bottom_floors & is_lower_in_above))
                 if not candidates:
                      # If no such floor, take any potential bottom floor
                      candidates = sorted(list(potential_bottom_floors))
                 if candidates:
                      bottom_floor = candidates[0]
                 else:
                      # Fallback: If no potential bottom floor found by this logic,
                      # pick the alphabetically first floor from all floors.
                      bottom_floor = sorted(list(all_floors_in_task))[0]
            else:
                 # Fallback: If no potential bottom floors at all, pick the alphabetically first floor.
                 bottom_floor = sorted(list(all_floors_in_task))[0]


        if bottom_floor:
            ordered_floors = [bottom_floor]
            current_floor = bottom_floor
            # Build the chain upwards using immediate 'above' relations
            # Assuming (above f_lower f_higher) means f_higher is immediately above f_lower.
            immediate_above_map = {}
            for f_lower, f_higher in above_relations:
                 immediate_above_map[f_lower] = f_higher # Assumes unique immediate above

            while current_floor in immediate_above_map:
                 next_floor = immediate_above_map[current_floor]
                 if next_floor not in ordered_floors:
                      ordered_floors.append(next_floor)
                      current_floor = next_floor
                 else:
                      # Cycle detected or already added - stop
                      break

            # Add any remaining floors not part of the main chain (e.g., disconnected)
            remaining_floors = sorted(list(all_floors_in_task - set(ordered_floors)))
            ordered_floors.extend(remaining_floors)

            for i, floor in enumerate(ordered_floors):
                self.floor_to_idx[floor] = i + 1 # Use 1-based indexing
                self.idx_to_floor[i + 1] = floor

        # 2. Store passenger destinations and collect passengers to serve
        self.passenger_destinations = {}
        self.passengers_to_serve = set() # Passengers mentioned in goal (served ?p)

        # Collect passengers to serve from goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                passenger = parts[1]
                self.passengers_to_serve.add(passenger)

        # Collect destinations from static facts (destin ?p ?f)
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "destin":
                 passenger, floor = parts[1], parts[2]
                 self.passenger_destinations[passenger] = floor

        # Note: Passengers might be in goals but not in destin facts (malformed problem),
        # or in destin facts but not goals (already served initially?).
        # We will only calculate heuristic for passengers in self.passengers_to_serve
        # that also have a destination in self.passenger_destinations.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        # If lift location is unknown, the state is likely invalid or unreachable
        if current_lift_floor is None or current_lift_floor not in self.floor_to_idx:
             # print(f"Heuristic Warning: Lift location unknown or invalid floor: {current_lift_floor}")
             return float('inf') # Cannot compute heuristic without lift location

        current_lift_floor_idx = self.floor_to_idx[current_lift_floor]

        total_heuristic = 0

        # Identify served and boarded passengers in the current state
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_facts_in_state = {fact for fact in state if match(fact, "origin", "*", "*")}

        # Iterate through all passengers that need to be served (those in the goal)
        for passenger in self.passengers_to_serve:
            if passenger in served_passengers:
                continue # This passenger is already served

            # Passenger is not served. Calculate cost for this passenger.
            destin_floor = self.passenger_destinations.get(passenger)
            if destin_floor is None or destin_floor not in self.floor_to_idx:
                 # Should not happen in a valid problem, but handle gracefully
                 # print(f"Heuristic Warning: Passenger {passenger} needs serving but has no valid destination.")
                 # Assign a penalty or skip? Skipping might underestimate. Let's skip for now.
                 continue

            destin_floor_idx = self.floor_to_idx[destin_floor]

            if passenger in boarded_passengers:
                # Passenger is boarded but not served
                total_heuristic += 1 # Cost for depart action
                # Cost for lift movement from current floor to destination floor
                total_heuristic += abs(destin_floor_idx - current_lift_floor_idx)
            else:
                # Passenger is not boarded and not served, must be at origin
                origin_floor = None
                # Find origin floor from the current state's origin facts
                for fact in origin_facts_in_state:
                    p, f = get_parts(fact)[1], get_parts(fact)[2]
                    if p == passenger:
                        origin_floor = f
                        break

                if origin_floor is None or origin_floor not in self.floor_to_idx:
                    # This state is inconsistent: passenger not served, not boarded, and not at origin.
                    # Or maybe the origin fact was removed by an action not shown?
                    # Assuming valid states, this shouldn't happen for passengers that need serving.
                    # print(f"Heuristic Warning: Passenger {passenger} not served, not boarded, and no origin fact found in state.")
                    # Assign a penalty or skip? Skipping might underestimate. Let's skip for now.
                    continue

                origin_floor_idx = self.floor_to_idx[origin_floor]

                total_heuristic += 2 # Cost for board + depart actions
                # Cost for lift movement from current floor to origin floor
                total_heuristic += abs(origin_floor_idx - current_lift_floor_idx)
                # Cost for lift movement from origin floor to destination floor
                total_heuristic += abs(destin_floor_idx - origin_floor_idx)

        return total_heuristic
