from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.

    # Assumptions:
    - The man can carry at most one spanner at a time.
    - The man must pick up a spanner before tightening a nut.
    - The man starts at the 'shed' location if not specified otherwise.
    - The shortest path between locations is considered based on the given links.

    # Heuristic Initialization
    - Extract static facts to build a map of location links.
    - Store the goal locations for each nut.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current location of the man and all nuts.
    2. For each loose nut:
        a. If the man does not have a spanner, calculate the actions to pick up a spanner and then tighten the nut.
        b. If the man already has a spanner, calculate the actions to move to the nut's location and tighten it.
    3. Sum the actions for all loose nuts, ensuring that spanner pickup is only counted once.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Build a map of location links from static facts
        self.location_links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = fact[1:-1].split()[1], fact[1:-1].split()[2]
                if loc1 not in self.location_links:
                    self.location_links[loc1] = []
                self.location_links[loc1].append(loc2)
                if loc2 not in self.location_links:
                    self.location_links[loc2] = []
                self.location_links[loc2].append(loc1)

        # Store goal locations for each nut
        self.goal_nuts = {}
        for goal in self.goals:
            if match(goal, "(tightened", "*"):
                nut = goal[1:-1].split()[1]
                self.goal_nuts[nut] = True

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        def match(fact, *args):
            """Check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        def get_parts(fact):
            """Extract the components of a PDDL fact."""
            return fact[1:-1].split()

        # Extract current locations
        current_locations = {}
        carrying_spanner = False
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "location*"):
                obj, loc = get_parts(fact)
                current_locations[obj] = loc
            elif match(fact, "carrying", "*", "*"):
                man, spanner = get_parts(fact)
                carrying_spanner = True
            elif match(fact, "at", "bob", "*"):
                man_location = get_parts(fact)[2]

        # Find all nuts that are still loose
        loose_nuts = []
        for goal in self.goal_nuts:
            if f"(loose {goal})" in state:
                loose_nuts.append(goal)

        if not loose_nuts:
            return 0

        # Find the location of each loose nut
        nut_locations = {}
        for fact in state:
            if match(fact, "at", "*", "location*") and fact.startswith("(at nut"):
                nut, loc = get_parts(fact)
                nut_locations[nut] = loc

        total_cost = 0
        first_nut = True

        for nut in loose_nuts:
            nut_loc = nut_locations[nut]

            if first_nut:
                # Calculate steps to get a spanner and then to the nut
                # Find the nearest spanner location
                spanner_locations = [loc for loc in self.location_links if loc.endswith("location")]
                min_distance = float('inf')
                for spanner_loc in spanner_locations:
                    distance = self.get_distance(man_location, spanner_loc)
                    if distance < min_distance:
                        min_distance = distance
                total_cost += min_distance + 2  # walk to spanner, pickup, walk to nut
                first_nut = False
            else:
                # Calculate steps to move from current location to nut's location
                current_pos = man_location
                distance = self.get_distance(current_pos, nut_loc)
                total_cost += distance + 1  # walk to nut, tighten

        return total_cost

    def get_distance(self, start, end):
        """Calculate the shortest path distance between two locations."""
        if start == end:
            return 0
        visited = set()
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            if current == end:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.location_links.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')  # If end is unreachable, though it shouldn't happen in valid states
