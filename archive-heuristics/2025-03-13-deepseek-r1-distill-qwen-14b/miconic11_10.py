from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the required movements of the lift and the boarding/departing actions.

    # Assumptions:
    - The lift can move between floors and serve passengers.
    - Each passenger must be boarded at their origin floor and departed at their destination floor.
    - The minimal number of actions is calculated by considering the shortest path between floors and the necessary boarding/departing actions.

    # Heuristic Initialization
    - Extract static facts to build a graph of floor connections.
    - Identify the destination floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    1. Build a graph of floor connections using the 'above' relationships from static facts.
    2. For each passenger, check if they are already served. If not, calculate the required actions:
       a. Calculate the shortest path from the lift's current floor to the passenger's origin floor.
       b. Add actions for boarding the passenger.
       c. Calculate the shortest path from the origin floor to the passenger's destination floor.
       d. Add actions for departing the passenger.
    3. Sum the actions required for all passengers to get the total estimated actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and building floor connections."""
        self.goals = task.goals
        static_facts = task.static

        # Build floor hierarchy and adjacency graph
        self.floors = set()
        self.above_graph = {}
        self.below_graph = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f1, f2 = get_parts(fact)[1], get_parts(fact)[2]
                self.floors.add(f1)
                self.floors.add(f2)
                if f1 not in self.above_graph:
                    self.above_graph[f1] = []
                self.above_graph[f1].append(f2)
                if f2 not in self.below_graph:
                    self.below_graph[f2] = []
                self.below_graph[f2].append(f1)

        # Map each floor to its possible reachable floors
        self.reachable = {f: set() for f in self.floors}
        for f in self.floors:
            queue = deque()
            queue.append(f)
            visited = set()
            visited.add(f)
            while queue:
                current = queue.popleft()
                for neighbor in self.above_graph.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                for neighbor in self.below_graph.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            self.reachable[f] = visited

        # Extract destination floors for each passenger
        self.destinations = {}
        for goal in self.goals:
            if match(goal, "destin", "*", "*"):
                p, dest = get_parts(goal)[1], get_parts(goal)[2]
                self.destinations[p] = dest

    def __call__(self, node):
        """Estimate the minimum number of actions to serve all passengers."""
        state = node.state
        current_lift_floor = None

        # Find current lift floor
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
            return 0  # No lift position, assume already at goal

        # Track boarded and served passengers
        boarded = set()
        served = set()
        for fact in state:
            if match(fact, "boarded", "*"):
                boarded.add(get_parts(fact)[1])
            if match(fact, "served", "*"):
                served.add(get_parts(fact)[1])

        # Get all passengers
        passengers = set()
        for fact in state:
            if match(fact, "origin", "*", "*"):
                p = get_parts(fact)[1]
                passengers.add(p)

        total_actions = 0

        for p in passengers:
            if p in served:
                continue  # Already served

            # Get origin and destination floors
            origin = None
            for fact in state:
                if match(fact, "origin", p, "*"):
                    origin = get_parts(fact)[2]
                    break
            dest = self.destinations.get(p, None)

            if not origin or not dest:
                continue  # Invalid state

            # Check if already boarded
            if p in boarded:
                # Need to move to destination and depart
                # Calculate shortest path from current_lift_floor to dest
                if dest == current_lift_floor:
                    actions = 0
                else:
                    # BFS to find shortest path
                    visited = set()
                    queue = deque()
                    queue.append((current_lift_floor, 0))
                    visited.add(current_lift_floor)
                    found = False
                    while queue:
                        f, d = queue.popleft()
                        if f == dest:
                            actions = d
                            found = True
                            break
                        for nf in self.above_graph.get(f, []):
                            if nf not in visited:
                                visited.add(nf)
                                queue.append((nf, d + 1))
                        for nf in self.below_graph.get(f, []):
                            if nf not in visited:
                                visited.add(nf)
                                queue.append((nf, d + 1))
                    if not found:
                        continue  # No path, assume infinite cost

                total_actions += actions + 1  # Depart action
                current_lift_floor = dest
            else:
                # Need to move to origin, board, then move to dest, depart
                # Calculate path to origin
                if origin == current_lift_floor:
                    actions_origin = 0
                else:
                    # BFS to find shortest path
                    visited = set()
                    queue = deque()
                    queue.append((current_lift_floor, 0))
                    visited.add(current_lift_floor)
                    found = False
                    while queue:
                        f, d = queue.popleft()
                        if f == origin:
                            actions_origin = d
                            found = True
                            break
                        for nf in self.above_graph.get(f, []):
                            if nf not in visited:
                                visited.add(nf)
                                queue.append((nf, d + 1))
                        for nf in self.below_graph.get(f, []):
                            if nf not in visited:
                                visited.add(nf)
                                queue.append((nf, d + 1))
                    if not found:
                        continue  # No path, assume infinite cost

                total_actions += actions_origin + 1  # Board action

                # Now at origin, need to move to dest
                if origin == dest:
                    actions_dest = 0
                else:
                    # BFS to find shortest path
                    visited = set()
                    queue = deque()
                    queue.append((origin, 0))
                    visited.add(origin)
                    found = False
                    while queue:
                        f, d = queue.popleft()
                        if f == dest:
                            actions_dest = d
                            found = True
                            break
                        for nf in self.above_graph.get(f, []):
                            if nf not in visited:
                                visited.add(nf)
                                queue.append((nf, d + 1))
                        for nf in self.below_graph.get(f, []):
                            if nf not in visited:
                                visited.add(nf)
                                queue.append((nf, d + 1))
                    if not found:
                        continue  # No path, assume infinite cost

                total_actions += actions_dest + 1  # Depart action
                current_lift_floor = dest

        return total_actions

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))
