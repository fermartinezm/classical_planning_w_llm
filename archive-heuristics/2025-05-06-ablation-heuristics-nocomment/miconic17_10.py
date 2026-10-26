# Need to import the base class Heuristic
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the number of actions required to reach the goal state
        by summing the estimated lift travel cost and the number of
        pending board and depart actions.

    Assumptions:
        - The floors form a linear sequence ordered by the 'above' predicate,
          where (above f1 f2) means f1 is immediately above f2.
        - All passenger origin and destination floors are provided in the
          static facts.
        - The cost of each action (move, board, depart) is 1.

    Heuristic Initialization:
        - Parses the 'above' predicates from static facts to build a mapping
          between floor names and integer floor numbers. This allows calculating
          distances between floors. It assumes a linear floor structure where
          (above f_i f_j) implies f_i is immediately above f_j. The lowest floor
          is identified as the one not appearing as the second argument in any
          'above' predicate. Includes a fallback to alphabetical sorting if the
          linear structure cannot be fully determined from 'above' predicates.
        - Parses passenger 'destin' predicates from static facts to store
          the destination floor for each passenger.
        - Identifies all passengers present in the problem based on 'destin' facts.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state by verifying if all
           goal facts (typically '(served passenger)') are present in the state.
           If yes, the heuristic is 0.
        2. Identify the current floor of the lift from the state using the
           '(lift-at ?floor)' predicate.
        3. Identify all unserved passengers by comparing the set of all
           passengers in the problem (from static 'destin' facts) with the
           set of passengers marked as 'served' in the current state.
        4. Initialize an empty set `required_event_floors` to store floors
           the lift must visit, and counters for `num_board_actions_needed`
           and `num_depart_actions_needed`.
        5. Iterate through the facts in the current state:
           - If a fact is '(origin ?p ?o)' and ?p is an unserved passenger,
             add floor ?o to `required_event_floors` and increment
             `num_board_actions_needed`. Ensure ?o is in the floor mapping.
           - If a fact is '(boarded ?p)' and ?p is an unserved passenger,
             look up ?p's destination floor ?d from the pre-parsed static data.
             Add floor ?d to `required_event_floors` and increment
             `num_depart_actions_needed`. Ensure ?d is in the floor mapping.
        6. Calculate the estimated travel cost:
           - If `required_event_floors` is empty, travel cost is 0.
           - If not empty, convert the required floor names to their integer
             representations. Find the minimum (`min_int`) and maximum (`max_int`)
             integer floor numbers among these.
           - Get the integer representation of the current lift floor (`current_int`).
           - The estimated travel cost is calculated as:
             `(max_int - min_int) + min(abs(current_int - min_int), abs(current_int - max_int))`.
             Handle cases where the current lift floor or required floors are not in the floor mapping (e.g., due to incomplete static data or mapping fallback issues) by setting travel cost to 0.
        7. The total heuristic value is the sum of the estimated travel cost
           and the total count of `num_board_actions_needed` and
           `num_depart_actions_needed`.
    """
    def __init__(self, task):
        super().__init__(task) # Call base class constructor
        self.goals = task.goals
        static_facts = task.static

        # 1. Build floor mapping
        self.floor_to_int = {}
        self.int_to_floor = {}
        # Maps a floor f2 to f1 if (above f1 f2) i.e. f1 is immediately above f2
        floor_above_map = {}
        all_floors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                f1, f2 = parts[1], parts[2]
                floor_above_map[f2] = f1 # f1 is above f2, so f2 is lower
                all_floors.add(f1)
                all_floors.add(f2)

        if not all_floors:
             # Handle case with no floors or no above predicates
             # If there's a lift-at fact, assume that's the only floor and map it to 1
             lift_at_facts = [f for f in static_facts if get_parts(f)[0] == 'lift-at']
             if len(lift_at_facts) >= 1: # Use first lift-at fact to get a floor name
                 floor_name = get_parts(lift_at_facts[0])[1]
                 self.floor_to_int[floor_name] = 1
                 self.int_to_floor[1] = floor_name
             # If no floors and no lift-at, floor mapping remains empty. Heuristic will be 0.

        else:
            # Find the lowest floor (a floor f that is not a key in floor_above_map)
            lowest_floor = None
            floors_that_are_below_others = set(floor_above_map.keys())
            for floor in all_floors:
                if floor not in floors_that_are_below_others:
                    lowest_floor = floor
                    break # Assuming a single lowest floor in a linear structure

            if lowest_floor:
                current_floor = lowest_floor
                current_floor_num = 1
                self.floor_to_int[current_floor] = current_floor_num
                self.int_to_floor[current_floor_num] = current_floor

                # Build mapping upwards
                while len(self.floor_to_int) < len(all_floors):
                    next_floor = floor_above_map.get(current_floor)
                    if next_floor:
                        current_floor_num += 1
                        self.floor_to_int[next_floor] = current_floor_num
                        self.int_to_floor[current_floor_num] = next_floor
                        current_floor = next_floor
                    else:
                        # Should only happen when current_floor is the highest floor
                        # If it happens before all floors are mapped, something is wrong
                        if len(self.floor_to_int) < len(all_floors):
                             # Fallback to alphabetical if linear structure wasn't fully mapped
                             unmapped_floors = sorted(list(all_floors - set(self.floor_to_int.keys())))
                             start_num = len(self.floor_to_int) + 1
                             for i, floor in enumerate(unmapped_floors):
                                 self.floor_to_int[floor] = start_num + i
                                 self.int_to_floor[start_num + i] = floor
                        break # Stop mapping

            else:
                 # Fallback to alphabetical if no unique lowest floor was found
                 if all_floors:
                     sorted_floors = sorted(list(all_floors))
                     for i, floor in enumerate(sorted_floors):
                         self.floor_to_int[floor] = i + 1
                         self.int_to_floor[i + 1] = floor


        # 2. Store passenger destinations
        self.passenger_destin = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                person, floor = parts[1], parts[2]
                self.passenger_destin[person] = floor

        # 3. Identify all passengers defined by destin facts
        self.all_passengers = set(self.passenger_destin.keys())


    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
             return 0

        # 2. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        # Flag to indicate if travel cost can be calculated
        can_calculate_travel = (current_lift_floor is not None and current_lift_floor in self.floor_to_int)
        if can_calculate_travel:
             current_int = self.floor_to_int[current_lift_floor]
        # else: Warning printed in init if mapping failed or lift-at missing


        # 3. Identify unserved passengers and their required floors/actions
        required_event_floors = set()
        num_board_actions_needed = 0
        num_depart_actions_needed = 0

        # Find all unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        unserved_passengers = self.all_passengers - served_passengers

        # Iterate through state facts to find unserved passengers' locations/status
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "origin":
                p, origin_floor = parts[1], parts[2]
                if p in unserved_passengers:
                    if origin_floor in self.floor_to_int:
                        required_event_floors.add(origin_floor)
                        num_board_actions_needed += 1
                    # else: Warning about floor mapping printed in init

            elif predicate == "boarded":
                p = parts[1]
                if p in unserved_passengers:
                    destin_floor = self.passenger_destin.get(p)
                    if destin_floor and destin_floor in self.floor_to_int:
                        required_event_floors.add(destin_floor)
                        num_depart_actions_needed += 1
                    # else: Warning about destin/floor mapping printed in init


        # 4. Calculate travel cost
        travel_cost = 0
        if can_calculate_travel and required_event_floors:
            required_ints = {self.floor_to_int[f] for f in required_event_floors if f in self.floor_to_int}

            if required_ints: # Ensure required_ints is not empty after filtering
                min_int = min(required_ints)
                max_int = max(required_ints)

                # Estimated travel to visit all floors in the range [min_int, max_int]
                # starting from current_int.
                travel_cost = (max_int - min_int) + min(abs(current_int - min_int), abs(current_int - max_int))
            # else: required_event_floors was not empty, but none of the floors were in the mapping.
            # travel_cost remains 0, warning printed in init.


        # 5. Action counts are already calculated inside the loop
        # num_board_actions_needed
        # num_depart_actions_needed

        # 6. Total heuristic
        total_cost = travel_cost + num_board_actions_needed + num_depart_actions_needed

        return total_cost

