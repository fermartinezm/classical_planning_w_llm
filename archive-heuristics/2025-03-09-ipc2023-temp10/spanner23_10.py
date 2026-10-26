from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class spanner23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the spanner's location, and the nut's location.
    It also considers whether the man is carrying a usable spanner.

    # Assumptions:
    - The man can only carry one spanner at a time.
    - A spanner must be usable to tighten a nut.
    - The heuristic assumes that the shortest path between locations is always used.

    # Heuristic Initialization
    - Extract the locations of all objects (man, spanners, nuts) from the initial state.
    - Extract the link information between locations from the static facts.
    - Build a location graph to calculate the shortest path between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the loose nuts that need to be tightened.
    2. Check if the man is carrying a usable spanner.
    3. If not, find the closest usable spanner and estimate the cost to pick it up.
    4. For each loose nut, estimate the cost to reach the nut's location.
    5. If the man is not at the nut's location, estimate the cost to walk to the nut's location.
    6. Estimate the cost to tighten the nut (1 action).
    7. Sum up the costs for all loose nuts and the cost to acquire a spanner (if needed).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and building the location graph."""
        self.goals = task.goals
        static_facts = task.static

        # Extract link information to build the location graph.
        self.location_graph = {}
        for fact in static_facts:
            if self.match(fact, "link", "*", "*"):
                parts = self.get_parts(fact)
                l1, l2 = parts[1], parts[2]
                if l1 not in self.location_graph:
                    self.location_graph[l1] = []
                if l2 not in self.location_graph:
                    self.location_graph[l2] = []
                self.location_graph[l1].append(l2)
                self.location_graph[l2].append(l1)

    def __call__(self, node):
        """Estimate the minimum cost to tighten all loose nuts."""
        state = node.state

        # Extract information from the current state.
        man_location = None
        carrying_spanner = None
        usable_spanners = set()
        loose_nuts = set()
        spanner_locations = {}
        nut_locations = {}
        man_name = None

        for fact in state:
            if self.match(fact, "at", "*", "*"):
                parts = self.get_parts(fact)
                obj, location = parts[1], parts[2]
                if self.is_man(obj):
                    man_location = location
                    man_name = obj
                elif self.is_spanner(obj):
                    spanner_locations[obj] = location
                elif self.is_nut(obj):
                    nut_locations[obj] = location
            elif self.match(fact, "carrying", "*", "*"):
                parts = self.get_parts(fact)
                carrying_spanner = parts[2]
            elif self.match(fact, "usable", "*"):
                parts = self.get_parts(fact)
                usable_spanners.add(parts[1])
            elif self.match(fact, "loose", "*"):
                parts = self.get_parts(fact)
                loose_nuts.add(parts[1])

        # Check if the goal is already reached.
        if self.goal_reached(state):
            return 0

        total_cost = 0

        # Check if a usable spanner is being carried.
        if carrying_spanner is None or carrying_spanner not in usable_spanners:
            # Find the closest usable spanner.
            closest_spanner = None
            min_distance = float('inf')
            for spanner, location in spanner_locations.items():
                if spanner in usable_spanners:
                    distance = self.shortest_path_length(man_location, location)
                    if distance < min_distance:
                        min_distance = distance
                        closest_spanner = spanner
            if closest_spanner:
                spanner_location = spanner_locations[closest_spanner]
                total_cost += self.shortest_path_length(man_location, spanner_location)  # Walk to spanner
                total_cost += 1  # Pick up spanner
                man_location = spanner_location #update man location

        # Tighten each loose nut.
        for nut in loose_nuts:
            nut_location = nut_locations[nut]
            total_cost += self.shortest_path_length(man_location, nut_location)  # Walk to nut
            total_cost += 1  # Tighten nut
            man_location = nut_location #update man location

        return total_cost

    def get_parts(self, fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        return fact[1:-1].split()

    def match(self, fact, *args):
        """Check if a PDDL fact matches a given pattern."""
        parts = self.get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def is_man(self, obj):
        """Check if an object is a man."""
        return 'bob' in obj.lower()

    def is_spanner(self, obj):
        """Check if an object is a spanner."""
        return 'spanner' in obj.lower()

    def is_nut(self, obj):
        """Check if an object is a nut."""
        return 'nut' in obj.lower()

    def goal_reached(self, state):
        """Check if the goal is reached."""
        return self.goals <= state

    def shortest_path_length(self, start, end):
        """Calculate the shortest path length between two locations using BFS."""
        if start == end:
            return 0

        queue = [(start, 0)]
        visited = {start}

        while queue:
            location, distance = queue.pop(0)
            if location == end:
                return distance

            if location in self.location_graph:
                for neighbor in self.location_graph[location]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))

        return float('inf')  # Return infinity if no path is found
