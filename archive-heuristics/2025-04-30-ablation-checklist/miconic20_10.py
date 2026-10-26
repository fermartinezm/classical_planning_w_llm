from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Use regex to handle potential spaces within arguments if needed, but simple split is fine for miconic
    # return re.findall(r'\S+', fact[1:-1])
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It counts the number of boarding and departing actions needed
    and adds an estimate of the lift movement cost.

    # Assumptions
    - Floors are linearly ordered based on the `above` predicate.
    - Each passenger needs one board action and one depart action.
    - Lift movement cost is estimated based on the range of floors that
      need to be visited to pick up waiting passengers and drop off
      boarded passengers.

    # Heuristic Initialization
    - Parses the `above` facts to create a mapping from floor names to
      integer indices, assuming a linear floor structure (e.g., f1 above f2,
      f2 above f3 implies f1 > f2 > f3). The lowest floor is assigned index 1.
    - Stores the destination floor for each passenger.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Unserved Passengers:** Determine which passengers still need
        to be served (i.e., do not have the `(served ?p)` fact).
    2.  **Count Boarding Actions:** For each unserved passenger currently at
        their origin floor (`(origin ?p ?f)`), they need one `board` action.
        Add this count to the heuristic.
    3.  **Count Departing Actions:** For each unserved passenger currently
        boarded (`(boarded ?p)`), they need one `depart` action. Add this
        count to the heuristic.
    4.  **Identify Required Floors:** Collect the set of floors the lift must
        visit:
        - The origin floor for every waiting passenger (`(origin ?p ?f)`).
        - The destination floor for every boarded passenger (`(boarded ?p)`).
    5.  **Estimate Movement Cost:**
        - Convert the required floor names to their integer indices using the
          mapping created during initialization.
        - Find the minimum and maximum indices among the required floors.
        - Find the index of the lift's current floor.
        - The movement cost is estimated as the distance needed to traverse
          the range of required floors (`max_idx - min_idx`) plus the
          distance from the current lift floor to reach this range if it's
          outside the range. If the lift is below the minimum required floor,
          it must travel up to at least the minimum (`min_idx - current_idx`).
          If it's above the maximum required floor, it must travel down to at
          least the maximum (`current_idx - max_idx`). If it's within the range,
          no extra travel is needed to *reach* the range.
    6.  **Sum Costs:** The total heuristic value is the sum of boarding actions,
        departing actions, and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor structure and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Build floor name to index mapping
        # We assume a linear structure like (above f1 f2), (above f2 f3), ...
        # We need to find the lowest floor and build the mapping upwards.
        above_map = {} # Maps floor_below -> floor_above
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "?f_above", "?f_below"):
                f_above, f_below = get_parts(fact)[1:]
                above_map[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)

        self.floor_to_idx = {}
        if not all_floors:
             # Handle case with no floors or only one floor (no above facts)
             # If there's a lift-at fact, that's the only floor.
             # This case is unlikely in typical miconic problems but good practice.
             for fact in task.initial_state:
                 if match(fact, "lift-at", "?f"):
                     self.floor_to_idx[get_parts(fact)[1]] = 1
                     break
             if not self.floor_to_idx and task.initial_state:
                  # Find any floor mentioned if lift-at is missing (unlikely)
                  for fact in task.initial_state:
                      parts = get_parts(fact)
                      for part in parts:
                          if part.startswith('f'): # Simple heuristic for floor names
                              self.floor_to_idx[part] = 1
                              break
                      if self.floor_to_idx: break


        else:
            # Find the lowest floor: a floor that is a key in above_map but not a value
            # (or a floor that is a value but never a key if above_map maps high->low)
            # Let's stick to above_map: floor_below -> floor_above
            # Lowest floor is one that is a value but never a key.
            # Example: (above f1 f2), (above f2 f3). above_map = {f2: f1, f3: f2}.
            # Keys: {f2, f3}. Values: {f1, f2}. Floor in values but not keys: f1? No.
            # Let's reverse: above_map_rev: floor_above -> floor_below
            above_map_rev = {} # Maps floor_above -> floor_below
            for fact in static_facts:
                 if match(fact, "above", "?f_above", "?f_below"):
                     f_above, f_below = get_parts(fact)[1:]
                     above_map_rev[f_above] = f_below

            # Find the highest floor: a floor that is a key but not a value in above_map_rev
            highest_floor = None
            for f in all_floors:
                if f in above_map_rev and f not in above_map_rev.values():
                    highest_floor = f
                    break
            # If there's only one floor, it's both highest and lowest.
            if highest_floor is None and len(all_floors) == 1:
                 highest_floor = list(all_floors)[0]
            elif highest_floor is None:
                 # This case implies a non-linear or disconnected floor structure,
                 # or an error in parsing/domain definition.
                 # For this heuristic, we assume linear. Find *a* floor that is a key but not a value.
                 potential_highest = set(above_map_rev.keys()) - set(above_map_rev.values())
                 if potential_highest:
                     highest_floor = list(potential_highest)[0] # Pick one if multiple disconnected chains? Unlikely.
                 else:
                     # Still no highest floor found, maybe a cycle or single floor not in above.
                     # Fallback: just use the first floor found and assign index 1.
                     if all_floors:
                         highest_floor = list(all_floors)[0]


            if highest_floor:
                current_floor = highest_floor
                current_idx = len(all_floors) # Assign highest floor the highest index
                # Build map downwards
                while current_floor is not None:
                    self.floor_to_idx[current_floor] = current_idx
                    # Find the floor immediately below current_floor
                    floor_below = None
                    for f_high, f_low in above_map_rev.items():
                        if f_high == current_floor:
                             # Need to find the one immediately below.
                             # In a linear chain (f1 above f2, f2 above f3), f2 is immediately below f1.
                             # Check if f_low is a key in above_map_rev. If not, it's the lowest in this chain.
                             # If it is a key, check if there's any floor between current_floor and f_low.
                             # Given the standard miconic structure, the below_map_rev gives the immediate floor below.
                             floor_below = above_map_rev.get(current_floor)
                             break # Found the immediate floor below

                    current_floor = floor_below
                    current_idx -= 1

        # 2. Store passenger destinations
        self.passenger_to_dest = {}
        # Destinations are static facts
        for fact in static_facts:
            if match(fact, "destin", "?p", "?f"):
                p, f = get_parts(fact)[1:]
                self.passenger_to_dest[p] = f
        # Also check goals, although destin facts are usually static
        for goal in self.goals:
             if match(goal, "served", "?p"):
                 # We need the destination for served passengers too,
                 # but it must come from static facts.
                 pass # Destination is stored from static facts

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift floor
        current_lift_floor_name = None
        for fact in state:
            if match(fact, "lift-at", "?f"):
                current_lift_floor_name = get_parts(fact)[1]
                break

        if current_lift_floor_name is None:
             # Should not happen in a valid miconic state, but handle defensively
             return float('inf') # Cannot proceed without lift location

        current_lift_floor_idx = self.floor_to_idx.get(current_lift_floor_name)
        if current_lift_floor_idx is None:
             # Floor name not found in mapping - problem with initialization or state
             return float('inf')


        # Identify unserved passengers and required floors
        unserved_passengers = set()
        waiting_passengers = set()
        boarded_passengers = set()
        served_passengers = set()
        required_floors_names = set()

        for fact in state:
            if match(fact, "served", "?p"):
                served_passengers.add(get_parts(fact)[1])

        # Iterate through all known passengers (from destinations)
        all_passengers = set(self.passenger_to_dest.keys())

        for p in all_passengers:
            if p not in served_passengers:
                unserved_passengers.add(p)

        # Now check state for waiting/boarded status of unserved passengers
        for p in unserved_passengers:
            is_waiting = False
            is_boarded = False
            origin_floor = None

            # Check state facts for this passenger
            for fact in state:
                if match(fact, "origin", p, "?f"):
                    is_waiting = True
                    origin_floor = get_parts(fact)[2]
                    break # Assume only one origin fact per passenger

            if not is_waiting: # If not waiting, check if boarded
                 for fact in state:
                     if match(fact, "boarded", p):
                         is_boarded = True
                         break # Assume only one boarded fact per passenger

            if is_waiting:
                waiting_passengers.add(p)
                required_floors_names.add(origin_floor)
            elif is_boarded:
                boarded_passengers.add(p)
                # Add destination floor to required floors
                dest_floor = self.passenger_to_dest.get(p)
                if dest_floor:
                    required_floors_names.add(dest_floor)
                else:
                    # Should not happen if passenger_to_dest is built correctly
                    return float('inf') # Invalid state/setup


        # If no unserved passengers, goal is reached (or state is invalid)
        if not unserved_passengers:
             return 0 # Goal state

        # Calculate heuristic components
        h = 0

        # Cost for boarding actions needed
        h += len(waiting_passengers)

        # Cost for departing actions needed
        h += len(boarded_passengers)

        # Estimate movement cost
        required_floors_indices = {self.floor_to_idx[f] for f in required_floors_names if f in self.floor_to_idx}

        if required_floors_indices:
            min_idx = min(required_floors_indices)
            max_idx = max(required_floors_indices)

            # Movement cost estimate: distance to cover the range [min_idx, max_idx]
            # plus distance from current floor to reach that range if outside.
            movement_cost = (max_idx - min_idx)

            if current_lift_floor_idx < min_idx:
                movement_cost += (min_idx - current_lift_floor_idx)
            elif current_lift_floor_idx > max_idx:
                movement_cost += (current_lift_floor_idx - max_idx)
            # If current_idx is within [min_idx, max_idx], no extra cost to reach the range.

            h += movement_cost

        # Note: This heuristic is not admissible. It counts boarding/departing
        # actions for each passenger independently and adds a movement cost
        # that might not be strictly necessary (e.g., visiting a floor might
        # serve multiple purposes). However, it provides a reasonable estimate
        # for greedy best-first search.

        return h

