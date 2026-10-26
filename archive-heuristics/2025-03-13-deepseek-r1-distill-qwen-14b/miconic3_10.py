from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the minimal distance the lift needs to travel to each passenger's origin and destination, plus the necessary boarding and departing actions.

    # Assumptions:
    - The lift can move between any two connected floors in one action per floor.
    - Each passenger must be boarded and then transported to their destination before being served.
    - If a passenger is already boarded, only the distance to their destination and the departing action are considered.

    # Heuristic Initialization
    - Extracts static facts to build a graph of floor connections.
    - Precomputes the minimal distances between all pairs of floors using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. For each passenger not yet served:
       a. If not boarded, calculate the distance from the lift's current floor to their origin, add 1 action for boarding.
       b. Calculate the distance from the origin to their destination, add 1 action for departing.
       c. If already boarded, calculate the distance from the lift's current floor to their destination, add 1 action for departing.
    3. Sum all the calculated distances and actions to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and building the floor graph."""
        self.goals = task.goals
        static_facts = task.static

        # Build the floor graph based on 'above' relations
        self.floor_graph = {}
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f1, f2 = get_parts(fact)
                if f1 not in self.floor_graph:
                    self.floor_graph[f1] = []
                self.floor_graph[f1].append(f2)
                if f2 not in self.floor_graph:
                    self.floor_graph[f2] = []
                self.floor_graph[f2].append(f1)

        # Precompute minimal distances between all pairs of floors using BFS
        self.floor_distances = {}
        for floor in self.floor_graph:
            self.floor_distances[floor] = {}
            visited = {floor: 0}
            queue = deque([floor])
            while queue:
                current = queue.popleft()
                for neighbor in self.floor_graph[current]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            # Update distances for all floors
            for f in self.floor_graph:
                if f not in visited:
                    visited[f] = float('inf')  # Shouldn't happen in miconic
                self.floor_distances[floor][f] = visited[f]

    def __call__(self, node):
        """Compute the heuristic value for the given node."""
        state = node.state

        # Extract current lift floor
        current_lift = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift = get_parts(fact)[1]
                break

        # Extract passenger information
        passengers = {}
        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)
                if p not in passengers:
                    passengers[p] = {'origin': f}
                passengers[p]['origin'] = f
            if match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)
                if p not in passengers:
                    passengers[p] = {}
                passengers[p]['destin'] = f
            if match(fact, "boarded", "*"):
                passengers[p]['boarded'] = True
            if match(fact, "served", "*"):
                passengers[p]['served'] = True

        total_cost = 0

        for p, data in passengers.items():
            if data.get('served', False):
                continue
            if not data.get('boarded', False):
                # Need to board
                origin = data['origin']
                # Distance from current_lift to origin
                distance_to_origin = self.floor_distances[current_lift].get(origin, float('inf'))
                if distance_to_origin == float('inf'):
                    distance_to_origin = 0  # Assuming all floors are connected
                total_cost += distance_to_origin + 1  # board action
                # Now, move to destination
                destin = data['destin']
                # Distance from origin to destination
                distance_origin_to_destin = self.floor_distances[origin].get(destin, float('inf'))
                if distance_origin_to_destin == float('inf'):
                    distance_origin_to_destin = 0
                total_cost += distance_origin_to_destin + 1  # depart action
            else:
                # Already boarded, just need to move to destination and depart
                destin = data['destin']
                # Distance from current_lift to destination
                distance_to_destin = self.floor_distances[current_lift].get(destin, float('inf'))
                if distance_to_destin == float('inf'):
                    distance_to_destin = 0
                total_cost += distance_to_destin + 1  # depart action

        return total_cost
