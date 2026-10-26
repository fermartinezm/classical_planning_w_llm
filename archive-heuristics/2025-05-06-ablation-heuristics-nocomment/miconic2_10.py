from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse fact strings
def get_parts(fact):
    """Splits a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the remaining cost by summing the number of board actions needed,
        the number of depart actions needed, and an estimate of the minimum lift
        movement required to visit all necessary floors.

    Assumptions:
        - The heuristic is designed for the Miconic PDDL domain.
        - It is intended for use with greedy best-first search and does not need
          to be admissible, but should be non-negative and zero at goal states.
        - The floor structure is assumed to be a linear chain defined by the 'above' predicate,
          where (above f_higher f_lower) means f_higher is directly above f_lower.
        - The problem is solvable.

    Heuristic Initialization:
        - Parses static facts to determine the linear order of floors based on
          'above' predicates and creates a mapping from floor names to their
          index (level). Handles single-floor cases and falls back to alphabetical
          sorting if linear ordering from 'above' facts is not possible.
        - Parses static facts to store the destination floor for each passenger
          using 'destin' predicates.
        - Collects all passenger names defined in the problem (from 'destin' facts
          and 'served' goals).

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current floor of the lift from the state.
        2. Identify all unserved passengers by finding those with 'origin' or
           'boarded' facts in the state.
        3. Count the number of waiting passengers (those with an 'origin' fact).
           This contributes to the heuristic as each requires a 'board' action.
        4. Count the number of boarded passengers (those with a 'boarded' fact).
           This contributes to the heuristic as each requires a 'depart' action.
        5. Determine the set of floors the lift *must* visit:
           - Origin floors of all waiting passengers.
           - Destination floors of all boarded passengers.
        6. If the set of required floors is empty, all unserved passengers must
           already be at their destination and boarded (or the state is the goal),
           so the estimated move cost is 0. The total heuristic is just the sum
           from steps 3 and 4 (which will be 0 in the goal state).
        7. If required floors exist, find the minimum and maximum floor indices
           among these required floors using the pre-calculated floor order.
        8. Estimate the minimum number of move actions required for the lift to
           travel from its current floor to visit all required floors. This is
           calculated as the distance from the current floor's index to the closest
           extreme (min or max index) of the required floors, plus the distance
           between the minimum and maximum required floor indices.
        9. The total heuristic value is the sum of the counts from steps 3 and 4
           and the estimated move cost from step 8.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse floor order and create floor_to_index map
        above_pairs = []
        all_floors_from_above = set()
        floors_as_lower = set() # Floors that appear as the second arg in (above f_higher f_lower)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                f_higher, f_lower = parts[1], parts[2]
                above_pairs.append((f_higher, f_lower))
                all_floors_from_above.add(f_higher)
                all_floors_from_above.add(f_lower)
                floors_as_lower.add(f_lower)

        self.floors = []
        self.floor_to_index = {}
        all_floors = set() # Collect all floor names mentioned anywhere

        if above_pairs:
            # Linear structure defined by above facts
            lowest_floor = None
            # Lowest floor is in all_floors_from_above but not in floors_as_lower
            potential_lowest = all_floors_from_above - floors_as_lower

            if len(potential_lowest) == 1:
                lowest_floor = potential_lowest.pop()

            if lowest_floor:
                current = lowest_floor
                # Build sorted list (lowest to highest) by following the 'above' chain upwards
                while current is not None:
                    self.floors.append(current)
                    next_floor = None
                    # Find the floor directly above the current one: (above ?next current)
                    for f_higher, f_lower in above_pairs:
                        if f_lower == current:
                            next_floor = f_higher
                            break
                    current = next_floor

            # Collect all floors mentioned in above facts
            all_floors.update(all_floors_from_above)

        # If no above facts, or if ordering failed, try to find floors from other predicates
        if not self.floors:
             for fact in static_facts:
                parts = get_parts(fact)
                # Check predicates that involve floors
                if parts and parts[0] in ['lift-at', 'origin', 'destin']:
                    for part in parts[1:]:
                        if part.startswith('f'): # Simple check for floor names
                            all_floors.add(part)

             # If floors were found from other predicates, sort them alphabetically as a fallback
             if all_floors:
                 self.floors = sorted(list(all_floors))

        # Create floor_to_index map if floors were found
        if self.floors:
            self.floor_to_index = {floor: i for i, floor in enumerate(self.floors)}


        # 2. Parse passenger destinations and collect all passenger names
        self.passenger_destinations = {}
        self.all_passengers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin':
                p, f_destin = parts[1], parts[2]
                self.passenger_destinations[p] = f_destin
                self.all_passengers.add(p)

        # Also add passengers mentioned in goals if not in destin (unlikely in miconic, but safe)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served':
                self.all_passengers.add(parts[1])


    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        # If lift location is unknown or floor parsing failed, cannot estimate move cost reliably.
        # Return a base heuristic based on passengers only.
        # This fallback also applies if no floors were found at all.
        if current_lift_floor is None or not self.floor_to_index:
             # Count waiting and boarded passengers
            waiting_passengers_count = 0
            boarded_passengers_count = 0
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == 'origin' and parts[1] in self.all_passengers:
                    waiting_passengers_count += 1
                elif parts and parts[0] == 'boarded' and parts[1] in self.all_passengers:
                    boarded_passengers_count += 1
            # Simple heuristic: number of board + number of depart actions needed
            return waiting_passengers_count + boarded_passengers_count


        # 2. Identify waiting and boarded passengers relevant to the problem
        waiting_passengers = set()
        boarded_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'origin' and parts[1] in self.all_passengers:
                waiting_passengers.add(parts[1])
            elif parts and parts[0] == 'boarded' and parts[1] in self.all_passengers:
                boarded_passengers.add(parts[1])

        # If no passengers are waiting or boarded, and goal is not reached,
        # this implies unserved passengers are somehow not in these states,
        # which shouldn't happen in a valid miconic state unless they are served.
        # The initial goal check handles the actual goal state.
        # If waiting and boarded are empty, all unserved passengers must be served.
        if not waiting_passengers and not boarded_passengers:
             return 0 # Should be goal state

        # 3. Count board actions needed
        board_actions_needed = len(waiting_passengers)

        # 4. Count depart actions needed
        depart_actions_needed = len(boarded_passengers)

        # 5. Determine required floors to visit
        required_stops = set()
        for p in waiting_passengers:
             # Origin floor of waiting passenger
             # Find origin floor from state fact (origin p f)
             for fact in state:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'origin' and parts[1] == p:
                     required_stops.add(parts[2])
                     break # Found origin for this passenger

        for p in boarded_passengers:
            # Destination floor of boarded passenger
            if p in self.passenger_destinations:
                required_stops.add(self.passenger_destinations[p])
            # Note: If a boarded passenger doesn't have a destination in static,
            # something is wrong with the problem definition. We assume valid problems.


        # 6. Calculate estimated move cost
        estimated_move_cost = 0
        # Ensure required floors are valid floors and current lift floor is valid
        valid_required_indices = {self.floor_to_index[f] for f in required_stops if f in self.floor_to_index}

        if valid_required_indices and current_lift_floor in self.floor_to_index:
            idx_current = self.floor_to_index[current_lift_floor]
            idx_min_req = min(valid_required_indices)
            idx_max_req = max(valid_required_indices)

            # Minimum travel to visit all floors in the range [idx_min_req, idx_max_req]
            # starting from idx_current.
            estimated_move_cost = min(abs(idx_current - idx_min_req), abs(idx_current - idx_max_req)) + (idx_max_req - idx_min_req)
        # else: required_stops is empty or contains invalid floors, or current_lift_floor is invalid. Move cost remains 0.

        # 9. Total heuristic
        total_heuristic = board_actions_needed + depart_actions_needed + estimated_move_cost

        return total_heuristic
