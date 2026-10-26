from heuristics.heuristic_base import Heuristic
from collections import deque
import sys

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts by considering:
    - The man's current location.
    - The locations of all loose nuts.
    - The availability and proximity of spanners.

    # Assumptions:
    - The man can carry multiple spanners, but each tightening action uses one spanner.
    - The man must move to the location of each nut to tighten it.
    - If the man isn't carrying a spanner, he must retrieve one before tightening any nuts.

    # Heuristic Initialization
    - Extracts static facts to build a graph of locations and precomputes shortest paths between all pairs of locations.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Count the number of loose nuts and their locations.
    3. Check if the man is carrying any spanners.
    4. If not carrying spanners, find the nearest spanner location and calculate the distance to it.
    5. For each loose nut, calculate the distance from the man's current location (or nearest spanner location) to the nut's location.
    6. Sum the distances, the number of pickup actions, and the number of tighten actions to estimate the total number of actions required.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and building a graph of locations.
        """
        self.task = task
        self.graph = {}
        for fact in task.static:
            if fact.startswith('(link '):
                loc1 = fact.split()[1]
                loc2 = fact.split()[2]
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                self.graph[loc1].add(loc2)
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc2].add(loc1)
        # Precompute all pairs shortest paths using BFS
        self.graph_distance = {}
        for loc in self.graph:
            self.graph_distance[loc] = {}
            queue = deque()
            queue.append((loc, 0))
            visited = set()
            visited.add(loc)
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.graph_distance[loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        # Extract man's current location
        man_location = None
        for fact in state:
            if fact.startswith('(at ') and ' - man' in fact:
                parts = fact.split()
                man_location = parts[-1]
                break
        if man_location is None:
            return 0  # man's location not found, should not happen in valid state

        # Count loose nuts and their locations
        loose_nuts = []
        for fact in state:
            if fact.startswith('(loose ') and ' - nut' in fact:
                parts = fact.split()
                nut_location = parts[-1]
                loose_nuts.append(nut_location)
        L = len(loose_nuts)
        if L == 0:
            return 0

        # Check if the man is carrying any spanners
        has_spanners = any(fact.startswith('(carrying ') and ' - man' in fact and ' - spanner' in fact for fact in state)

        total_actions = 0
        current_position = man_location

        if not has_spanners:
            # Find all spanner locations from static facts
            spanner_locations = []
            for fact in self.task.static:
                if fact.startswith('(at ') and ' - spanner' in fact:
                    loc = fact.split()[-1]
                    spanner_locations.append(loc)
            if not spanner_locations:
                return float('inf')  # no spanners available, state is unsolvable

            # Find the nearest spanner location to the man's current position
            min_distance = float('inf')
            nearest_spanner = None
            for spanner_loc in spanner_locations:
                distance = self.graph_distance[man_location].get(spanner_loc, float('inf'))
                if distance < min_distance:
                    min_distance = distance
                    nearest_spanner = spanner_loc
            if nearest_spanner is None:
                return float('inf')

            # Move to the nearest spanner and pick up L spanners
            total_actions += min_distance + L
            current_position = nearest_spanner

        # Calculate the sum of distances from current position to each nut's location
        sum_d = 0
        for nut_loc in loose_nuts:
            distance = self.graph_distance[current_position].get(nut_loc, float('inf'))
            if distance == float('inf'):
                return float('inf')  # no path, state is unsolvable
            sum_d += distance

        # Add the number of tighten actions
        total_actions += sum_d + L

        return total_actions
