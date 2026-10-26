from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(link loc1 loc2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the spanner's location, and the nut's location,
    estimating the cost of walking, picking up the spanner, and tightening the nut.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A spanner must be picked up before tightening a nut.
    - The heuristic assumes the shortest path between locations.
    - Only one man exists.

    # Heuristic Initialization
    - Extract the link information between locations from the static facts to build a location graph.
    - Identify all nuts and their initial locations.
    - Identify all spanners and their initial locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts.
    3. For each loose nut:
       a. Find the closest usable spanner.
       b. Estimate the cost to reach the spanner (walk actions).
       c. Estimate the cost to pick up the spanner (pickup_spanner action).
       d. Estimate the cost to reach the nut (walk actions).
       e. Estimate the cost to tighten the nut (tighten_nut action).
    4. Sum the costs for all loose nuts.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build a location graph from the static link facts.
        self.location_graph = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = []
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1)

        # Identify all nuts and their initial locations.  This is not needed as we check the state.
        # self.nuts = {}
        # for fact in static_facts:
        #    if match(fact, "at", "*", "*") and "nut" in fact:
        #        nut, location = get_parts(fact)[1], get_parts(fact)[2]
        #        self.nuts[nut] = location

        # Identify all spanners and their initial locations. This is not needed as we check the state.
        # self.spanners = {}
        # for fact in static_facts:
        #    if match(fact, "at", "*", "*") and "spanner" in fact:
        #        spanner, location = get_parts(fact)[1], get_parts(fact)[2]
        #        self.spanners[spanner] = location

    def __call__(self, node):
        """Estimate the number of actions needed to tighten all loose nuts."""
        state = node.state

        # Check if the goal is reached.
        if self.goals <= state:
            return 0

        # Identify the man's current location.
        for fact in state:
            if match(fact, "at", "*", "*") and "bob" in fact:
                man_location = get_parts(fact)[1]
                break
        else:
            return float('inf')  # Man's location not found.

        # Identify all loose nuts.
        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                loose_nuts.append(get_parts(fact)[1])

        total_cost = 0
        for nut in loose_nuts:
            # Find the nut's location.
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break
            else:
                return float('inf')  # Nut's location not found.

            # Find the closest usable spanner.
            closest_spanner = None
            min_distance = float('inf')

            for fact in state:
                if match(fact, "at", "*", "*") and "spanner" in fact and match(fact, "usable", "*") and get_parts(fact)[1] in state:
                    spanner = get_parts(fact)[1]
                    for fact2 in state:
                        if match(fact2, "at", spanner, "*"):
                            spanner_location = get_parts(fact2)[2]
                            break
                    else:
                        return float('inf') # Spanner location not found

                    distance = self.shortest_path_length(man_location, spanner_location)
                    if distance < min_distance:
                        min_distance = distance
                        closest_spanner = spanner

            if closest_spanner is None:
                return float('inf')  # No usable spanner found.

            # Estimate the cost to reach the spanner.
            spanner_cost = self.shortest_path_length(man_location, spanner_location)

            # Estimate the cost to pick up the spanner.
            pickup_cost = 1

            # Estimate the cost to reach the nut.
            nut_cost = self.shortest_path_length(spanner_location, nut_location)

            # Estimate the cost to tighten the nut.
            tighten_cost = 1

            total_cost += spanner_cost + pickup_cost + nut_cost + tighten_cost

        return total_cost

    def shortest_path_length(self, start, end):
        """Compute the shortest path length between two locations using BFS."""
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

        return float('inf')  # No path found.
