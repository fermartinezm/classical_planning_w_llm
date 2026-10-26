from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all nuts by considering the man's current location, the locations of the nuts, and the availability of spanners.

    # Assumptions:
    - The man can carry only one spanner at a time.
    - Each spanner can be used only once.
    - The man must be at the same location as a nut and a usable spanner to tighten it.
    - The man must return to the initial location after each trip if not already there.

    # Heuristic Initialization
    - Extract the static facts to build the location graph and identify spanner locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each nut that is not yet tightened:
       a. Determine if the man is carrying a spanner and is at the nut's location.
       b. If not, find the nearest spanner location.
       c. Calculate the distance from the man's current location to the spanner's location.
       d. Calculate the distance from the spanner's location to the nut's location.
       e. Sum these distances and add 2 actions (pick up spanner and tighten nut).
    2. Sum the estimated actions for all nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        self.task = task
        self.adjacency_list = self.build_adjacency_list(task.static)
        self.spanner_locations = self.get_spanner_locations(task.static)
        self.nuts = self.get_nuts(task.goals)

    def build_adjacency_list(self, static_facts):
        """Build the adjacency list for the location graph."""
        adj_list = {}
        for fact in static_facts:
            if fact.startswith('(link'):
                parts = fact[5:-1].split()
                loc1, loc2 = parts[0], parts[1]
                if loc1 not in adj_list:
                    adj_list[loc1] = []
                adj_list[loc1].append(loc2)
                if loc2 not in adj_list:
                    adj_list[loc2] = []
                adj_list[loc2].append(loc1)
        return adj_list

    def get_spanner_locations(self, static_facts):
        """Identify the initial locations of all spanners."""
        spanners = set()
        for fact in static_facts:
            if fact.startswith('(at spanner'):
                parts = fact[5:-1].split()
                spanner, loc = parts[0], parts[1]
                spanners.add(loc)
        return spanners

    def get_nuts(self, goals):
        """Extract the nuts from the goals."""
        nuts = set()
        for goal in goals:
            if goal.startswith('(tightened nut'):
                nut = goal[10:-1].split()[0]
                nuts.add(nut)
        return nuts

    def __call__(self, node):
        """Compute the heuristic value for the given node."""
        state = node.state
        man_location = self.get_man_location(state)
        total_cost = 0

        for nut in self.nuts:
            if self.is_nut_tightened(state, nut):
                continue

            if self.man_has_spanner_and_at_nut(state, nut, man_location):
                total_cost += 1
                continue

            spanner_loc = self.find_nearest_spanner(state, man_location)
            if spanner_loc is None:
                return float('inf')

            distance_to_spanner = self.bfs_distance(man_location, spanner_loc)
            nut_loc = self.get_nut_location(nut)
            if nut_loc is None:
                return float('inf')
            distance_from_spanner_to_nut = self.bfs_distance(spanner_loc, nut_loc)

            total_cost += distance_to_spanner + distance_from_spanner_to_nut + 2

        return total_cost

    def get_man_location(self, state):
        """Find the current location of the man."""
        for fact in state:
            if fact.startswith('(at bob'):
                return fact[7:-1].split()[1]
        return None  # Should not happen in valid state

    def is_nut_tightened(self, state, nut):
        """Check if the nut is already tightened."""
        for fact in state:
            if fact.startswith('(tightened', nut):
                return True
        return False

    def man_has_spanner_and_at_nut(self, state, nut, man_location):
        """Check if the man is carrying a spanner and is at the nut's location."""
        for fact in state:
            if fact.startswith('(carrying bob'):
                if fact[9:-1].split()[1] == 'spanner':  # Man has a spanner
                    nut_loc = self.get_nut_location(nut)
                    return man_location == nut_loc
        return False

    def find_nearest_spanner(self, state, current_location):
        """Find the nearest spanner location."""
        # Check if any spanner is usable
        for fact in state:
            if fact.startswith('(at spanner') and '(usable' in fact:
                loc = fact[7:-1].split()[1]
                return loc
        # If no spanner is usable, check static spanner locations
        for loc in self.spanner_locations:
            # Check if the spanner at loc is usable
            for fact in state:
                if fact.startswith('(at spanner') and fact[7:-1].split()[1] == loc:
                    if any(fact_usable.startswith('(usable spanner') and fact_usable[8:-1].split()[1] == loc for fact_usable in state):
                        return loc
        return None

    def get_nut_location(self, nut):
        """Find the current location of a nut."""
        for fact in self.task.goals:
            if fact.startswith('(at nut', nut):
                return fact[7:-1].split()[1]
        return None

    def bfs_distance(self, start, goal):
        """Compute the shortest path distance using BFS."""
        if start == goal:
            return 0
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.adjacency_list.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')  # No path found
