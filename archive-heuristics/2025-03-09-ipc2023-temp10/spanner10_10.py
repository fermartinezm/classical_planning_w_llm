from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(link location1 location2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the location of the nuts, the location of the spanners,
    and whether the man is carrying a usable spanner.

    # Assumptions
    - The heuristic assumes that the shortest path between locations is always used.
    - It assumes that the man can only carry one spanner at a time.
    - It assumes that a spanner can only be used once.

    # Heuristic Initialization
    - Extract the links between locations from the static facts.
    - Build a graph representing the locations and their connections.
    - Store the initial locations of spanners and their usability.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current state information: man's location, carried spanner, loose nuts locations, usable spanners locations.
    2. If all nuts are tightened, return 0.
    3. For each loose nut:
       a. If the man is not at the nut's location:
          i. Calculate the shortest path distance from the man's current location to the nut's location.
          ii. Add the path length to the cost.
          iii. Add 1 to the cost for walking to the nut.
       b. If the man is not carrying a usable spanner:
          i. Find the closest usable spanner.
          ii. Calculate the shortest path distance from the man's current location to the closest usable spanner's location.
          iii. Add the path length to the cost.
          iv. Add 1 to the cost for walking to the spanner.
          v. Add 1 to the cost for picking up the spanner.
          vi. Add the shortest path distance from the spanner location to the nut location if different.
          vii. Add 1 to the cost for walking to the nut.
       c. Add 1 to the cost for tightening the nut.
    4. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting links between locations from static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build a graph of locations and their links.
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                if l1 not in self.links:
                    self.links[l1] = []
                if l2 not in self.links:
                    self.links[l2] = []
                self.links[l1].append(l2)
                self.links[l2].append(l1)

        # Store initial locations of spanners and their usability.
        self.spanner_locations = {}
        self.usable_spanners = set()
        for op in task.operators:
            for effect in op.add_effects:
                if match(effect, "at", "*", "*") and get_parts(effect)[1] not in ['bob', 'nut1', 'nut2', 'nut3', 'nut4', 'nut5', 'nut6', 'nut7', 'nut8', 'nut9', 'nut10']:
                    self.spanner_locations[get_parts(effect)[1]] = get_parts(effect)[2]
                if match(effect, "usable", "*"):
                    self.usable_spanners.add(get_parts(effect)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to tighten all loose nuts."""
        state = node.state

        # Extract current state information.
        man_location = None
        carried_spanner = None
        loose_nuts = []
        usable_spanners = set()
        spanner_locations = {}

        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_location = get_parts(fact)[3]
            if match(fact, "carrying", "bob", "*"):
                carried_spanner = get_parts(fact)[2]
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                nut_location = next((get_parts(f)[2] for f in state if match(f, "at", nut, "*")), None)
                loose_nuts.append((nut, nut_location))
            if match(fact, "usable", "*"):
                usable_spanners.add(get_parts(fact)[1])
            if match(fact, "at", "*", "*") and get_parts(fact)[1] not in ['bob', 'nut1', 'nut2', 'nut3', 'nut4', 'nut5', 'nut6', 'nut7', 'nut8', 'nut9', 'nut10']:
                spanner_locations[get_parts(fact)[1]] = get_parts(fact)[2]

        # If all nuts are tightened, return 0.
        all_tightened = True
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                if not any(match(fact, "tightened", nut) for fact in state):
                    all_tightened = False
                    break
        if all_tightened:
            return 0

        total_cost = 0

        for nut, nut_location in loose_nuts:
            if any(match(fact, "tightened", nut) for fact in state):
                continue

            # If the man is not at the nut's location, move him there.
            if man_location != nut_location:
                path_length = self.shortest_path_length(man_location, nut_location)
                total_cost += path_length
                total_cost += 1  # walk action
                man_location = nut_location

            # If the man is not carrying a usable spanner, pick one up.
            if carried_spanner is None or carried_spanner not in usable_spanners:
                closest_spanner = None
                min_distance = float('inf')

                for spanner in usable_spanners:
                    if spanner in spanner_locations:
                        spanner_location = spanner_locations[spanner]
                        distance = self.shortest_path_length(man_location, spanner_location)
                        if distance < min_distance:
                            min_distance = distance
                            closest_spanner = spanner

                if closest_spanner:
                    spanner_location = spanner_locations[closest_spanner]
                    path_length = self.shortest_path_length(man_location, spanner_location)
                    total_cost += path_length
                    total_cost += 1  # walk action
                    total_cost += 1  # pickup_spanner action
                    man_location = spanner_location
                    carried_spanner = closest_spanner
                    if spanner_location != nut_location:
                        path_length = self.shortest_path_length(spanner_location, nut_location)
                        total_cost += path_length
                        total_cost += 1  # walk action
                        man_location = nut_location

            total_cost += 1  # tighten_nut action

        return total_cost

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

            for neighbor in self.links.get(location, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf')  # Return infinity if no path is found
