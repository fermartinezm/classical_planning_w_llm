from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their goal locations.

    # Assumptions:
    - Each package must be transported from its current location to its goal location.
    - Vehicles can carry packages and move between locations connected by roads.
    - The heuristic assumes that a vehicle is available to transport each package.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Precomputes the shortest paths between all pairs of locations using BFS.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each package, determine its current location and goal location.
    2. If the package is already at its goal, no actions are needed.
    3. Otherwise, compute the shortest path from the package's current location to its goal location.
    4. Each package requires driving to its location (if not already there), picking it up, driving to the goal, and dropping it off.
    5. Sum the actions for all packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing shortest paths.
        """
        # Extract goal locations for each package
        self.goals = {}
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goals[package] = location

        # Precompute shortest paths between all locations using BFS
        static_facts = task.static
        road_facts = [fact for fact in static_facts if match(fact, "road", "*", "*")]
        locations = set()
        for fact in road_facts:
            l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
            locations.add(l1)
            locations.add(l2)
        locations = list(locations)

        self.shortest_paths = {}
        for src in locations:
            self.shortest_paths[src] = {}
            queue = deque()
            queue.append((src, 0))
            visited = {src}
            while queue:
                current, dist = queue.popleft()
                self.shortest_paths[src][current] = dist
                for fact in road_facts:
                    if get_parts(fact)[0] != "road":
                        continue
                    l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                    if current == l1 and l2 not in visited:
                        visited.add(l2)
                        queue.append((l2, dist + 1))
                    if current == l2 and l1 not in visited:
                        visited.add(l1)
                        queue.append((l1, dist + 1))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        total_cost = 0

        for package in self.goals:
            goal_location = self.goals[package]
            if f"(at {package} {goal_location})" in state:
                continue  # Package is already at the goal

            # Determine the package's current location
            current_location = None
            # Check if the package is in a vehicle
            in_vehicle = None
            for fact in state:
                if fact.startswith(f"(in {package} "):
                    parts = get_parts(fact)
                    if parts[0] == "in" and parts[1] == package:
                        in_vehicle = parts[2]
                        break
            if in_vehicle is not None:
                # Find the vehicle's current location
                for fact in state:
                    if match(fact, f"at {in_vehicle} *"):
                        current_location = get_parts(fact)[2]
                        break
            else:
                # Find the package's current location
                for fact in state:
                    if fact.startswith(f"(at {package} "):
                        parts = get_parts(fact)
                        if parts[0] == "at" and parts[1] == package:
                            current_location = parts[2]
                            break

            if current_location is None:
                # Package is not located; assume it's in a vehicle that's at some location
                continue

            if current_location == goal_location:
                continue

            # Calculate the shortest path from current location to goal
            if current_location not in self.shortest_paths or goal_location not in self.shortest_paths[current_location]:
                # If no path exists, return infinity (problem unsolvable from this state)
                return float('inf')

            drive_actions = self.shortest_paths[current_location][goal_location]
            total_cost += drive_actions + 2  # +2 for pick-up and drop-off

        return total_cost
