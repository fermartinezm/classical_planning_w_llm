from fnmatch import fnmatch
import math

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.passenger_destinations[passenger] = destination_floor

        floor_below_to_floor_above_immediate = {}
        all_floors = set()
        above_pairs = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_above, f_below = get_parts(fact)
                above_pairs.add((f_above, f_below))
                all_floors.add(f_above)
                all_floors.add(f_below)

        if not all_floors:
             self.floor_to_level = {}
             self.floors = []
             return

        immediate_above_pairs = set()
        for f_above, f_below in above_pairs:
            is_immediate = True
            for f_mid in all_floors:
                if (f_above, f_mid) in above_pairs and (f_mid, f_below) in above_pairs:
                    is_immediate = False
                    break
            if is_immediate:
                floor_below_to_floor_above_immediate[f_below] = f_above
                immediate_above_pairs.add((f_above, f_below))

        lowest_floor = None
        immediate_below_floors = set(floor_below_to_floor_above_immediate.keys())
        immediate_above_floors = set(floor_below_to_floor_above_immediate.values())

        potential_lowest = set
