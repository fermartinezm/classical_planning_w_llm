from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the minimal moves required for the lift to reach each passenger's origin and destination.

    # Assumptions:
    - The lift can move up or down one floor at a time.
    - Each passenger must be boarded at their origin floor and departed at their destination floor.
    - The heuristic assumes that the lift serves each passenger one after another, which may not be optimal but is computationally feasible.

    # Heuristic Initialization
    - Extract the goal conditions and static facts (above relationships) from the task.
    - Build a hierarchy of floors based on the 'above' relationships to compute the distance between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current floor of the lift.
    2. For each passenger not yet served:
       a. If the passenger is not boarded, calculate the distance from the lift's current floor to their origin, then to their destination. Add the number of actions required for boarding and departing.
       b. If the passenger is boarded, calculate the distance from the lift's current floor to their destination and add the action for departing.
    3. Sum all the calculated distances and actions to get the total heuristic value.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Extract above relationships
        self.above = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f1, f2 = get_parts(fact)
                self.above[f2] = f1  # f1 is above f2

        # Build floor hierarchy and compute levels
        self.floor_levels = {}
        # Find the top floor (no parent)
        top_floors = [f for f in self.above.values() if f not in self.above]
        if top_floors:
            self.top_floor = top_floors[0]
        else:
            # If no above facts, all floors are at level 0
            self.top_floor = None

        # Compute levels for each floor
        if self.top_floor is not None:
            # Use BFS to compute levels
            visited = set()
            queue = deque()
            queue.append(self.top_floor)
            visited.add(self.top_floor)
            self.floor_levels[self.top_floor] = 0
            while queue:
                current = queue.popleft()
                for f in self.above.get(current, []):
                    if f not in visited:
                        self.floor_levels[f] = self.floor_levels[current] + 1
                        visited.add(f)
                        queue.append(f)
        else:
            # All floors are at level 0
            for fact in static_facts:
                if match(fact, "floor", "*"):
                    f = get_parts(fact)[1]
                    self.floor_levels[f] = 0

        # Extract passengers from goals
        self.passengers = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                p = get_parts(goal)[1]
                self.passengers.add(p)

    def __call__(self, node):
        state = node.state
        current_lift = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift = get_parts(fact)[1]
                break

        if not current_lift:
            return 0

        total_actions = 0

        for p in self.passengers:
            served = any(match(fact, "served", p) for fact in state)
            if served:
                continue

            boarded = any(match(fact, "boarded", p) for fact in state)
            if boarded:
                # Need to move to destination and depart
                dest = None
                for fact in state:
                    if match(fact, "destin", p, "*"):
                        dest = get_parts(fact)[2]
                        break
                if dest is None:
                    continue  # Shouldn't happen
                # Calculate distance from current_lift to dest
                if self.top_floor is None:
                    distance = 0
                else:
                    level_current = self.floor_levels.get(current_lift, 0)
                    level_dest = self.floor_levels.get(dest, 0)
                    distance = abs(level_current - level_dest)
                total_actions += distance + 1  # depart action
            else:
                # Need to move to origin, board, then to dest, depart
                origin = None
                dest = None
                for fact in state:
                    if match(fact, "origin", p, "*"):
                        origin = get_parts(fact)[2]
                        break
                    if match(fact, "destin", p, "*"):
                        dest = get_parts(fact)[2]
                        break
                if origin is None or dest is None:
                    continue  # Shouldn't happen
                # Distance from current_lift to origin
                if self.top_floor is None:
                    distance_origin = 0
                else:
                    level_current = self.floor_levels.get(current_lift, 0)
                    level_origin = self.floor_levels.get(origin, 0)
                    distance_origin = abs(level_current - level_origin)
                # Distance from origin to dest
                if self.top_floor is None:
                    distance_dest = 0
                else:
                    level_origin = self.floor_levels.get(origin, 0)
                    level_dest = self.floor_levels.get(dest, 0)
                    distance_dest = abs(level_origin - level_dest)
                total_actions += distance_origin + distance_dest + 2  # board and depart

        return total_actions
