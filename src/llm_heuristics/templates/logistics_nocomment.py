from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    return fact[1:-1].split()


def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class LogisticsHeuristic(Heuristic):    
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.location_to_city = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts
            if match(fact, "in-city", "*", "*")
        }

        self.airports = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "airport", "*")
        }

        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

    def __call__(self, node):
        state = node.state

        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:
                obj, location = args
                current_locations[obj] = location

        total_cost = 0

        for package, goal_location in self.goal_locations.items():            
            current_location = current_locations[package]

            in_vehicle = current_location not in self.location_to_city

            if in_vehicle:            
                in_plane = current_location.startswith("plane")
                in_truck = current_location.startswith("truck")
                assert in_plane ^ in_truck, f"Invalid state: {current_location}"

                current_location = current_locations[current_location]
            else:
                in_plane = False
                in_truck = False

            current_city = self.location_to_city[current_location]
            goal_city = self.location_to_city[goal_location]

            if current_city == goal_city:
                if in_plane:
                    total_cost += 1

                if current_location != goal_location and not in_truck:
                    total_cost += 1

                if current_location != goal_location or in_truck:
                    total_cost += 1

            else:
                if current_location not in self.airports and not in_truck:
                    total_cost += 1

                if current_location not in self.airports or in_truck:
                    total_cost += 1

                if not in_plane:
                    total_cost += 1

                total_cost += 1
                
                if goal_location not in self.airports:
                    total_cost += 1
                    total_cost += 1

        return total_cost
