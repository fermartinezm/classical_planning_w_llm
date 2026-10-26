from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class spanner8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the location of the nuts, the location of the spanners,
    and whether the man is carrying a usable spanner.

    # Assumptions:
    - The heuristic assumes that the agent will always pick up the closest available spanner.
    - It assumes that the agent will always walk to the closest nut that needs tightening.
    - It assumes that there are enough usable spanners to tighten all loose nuts.

    # Heuristic Initialization
    - The heuristic initializes by extracting the locations of all objects (man, spanners, nuts)
      and the link information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners and their locations.
    4. Determine if the man is carrying a usable spanner.
    5. If not carrying a usable spanner, estimate the cost to pick up the closest usable spanner:
       - Find the closest usable spanner.
       - Estimate the cost to walk to the spanner's location.
       - Add 1 to the cost for the pickup_spanner action.
    6. For each loose nut:
       - Estimate the cost to walk to the nut's location.
       - Add 1 to the cost for the tighten_nut action.
    7. Sum the costs to get the final heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting object locations and link information."""
        self.goals = task.goals
        static_facts = task.static

        self.links = {}
        for fact in static_facts:
            if self.match(fact, "link", "*", "*"):
                parts = self.get_parts(fact)
                l1, l2 = parts[1], parts[2]
                if l1 not in self.links:
                    self.links[l1] = []
                self.links[l1].append(l2)
                if l2 not in self.links:
                    self.links[l2] = []
                self.links[l2].append(l1)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        man_location = None
        for fact in state:
            if self.match(fact, "at", "*", "*") and self.is_man(fact):
                man_location = self.get_parts(fact)[2]
                break

        loose_nuts = []
        for fact in state:
            if self.match(fact, "loose", "*"):
                loose_nuts.append(self.get_parts(fact)[1])

        nut_locations = {}
        for nut in loose_nuts:
            for fact in state:
                if self.match(fact, "at", "*", "*") and self.is_nut(fact, nut):
                    nut_locations[nut] = self.get_parts(fact)[2]
                    break

        usable_spanners = []
        for fact in state:
            if self.match(fact, "usable", "*"):
                usable_spanners.append(self.get_parts(fact)[1])

        spanner_locations = {}
        for spanner in usable_spanners:
            for fact in state:
                if self.match(fact, "at", "*", "*") and self.is_spanner(fact, spanner):
                    spanner_locations[spanner] = self.get_parts(fact)[2]
                    break

        carrying_usable_spanner = False
        for fact in state:
            if self.match(fact, "carrying", "*", "*"):
                man, spanner = self.get_parts(fact)[1], self.get_parts(fact)[2]
                if self.is_man(f"(at {man} dummy)") and self.is_spanner(f"(at dummy {spanner})") and spanner in usable_spanners:
                    carrying_usable_spanner = True
                    break

        total_cost = 0

        if not carrying_usable_spanner and usable_spanners:
            # Find the closest usable spanner
            min_dist = float('inf')
            closest_spanner_location = None
            for spanner in usable_spanners:
                if spanner in spanner_locations:
                    spanner_location = spanner_locations[spanner]
                    dist = self.shortest_path(man_location, spanner_location)
                    if dist < min_dist:
                        min_dist = dist
                        closest_spanner_location = spanner_location
            if closest_spanner_location:
                total_cost += min_dist
                total_cost += 1  # Cost for pickup_spanner action

        for nut in loose_nuts:
            nut_location = nut_locations[nut]
            dist = self.shortest_path(man_location, nut_location)
            total_cost += dist
            total_cost += 1  # Cost for tighten_nut action

        # Check if the goal is reached
        goal_reached = True
        for goal in self.goals:
            if goal not in state:
                goal_reached = False
                break

        if goal_reached:
            return 0
        else:
            return total_cost

    def get_parts(self, fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        return fact[1:-1].split()

    def match(self, fact, *args):
        """Check if a PDDL fact matches a given pattern."""
        parts = self.get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def is_man(self, fact):
        """Check if the fact refers to a man."""
        return "bob" in fact.lower()  # Assuming 'bob' is the man's name

    def is_nut(self, fact, nut):
        """Check if the fact refers to a specific nut."""
        return nut in fact

    def is_spanner(self, fact, spanner):
        """Check if the fact refers to a specific spanner."""
        return spanner in fact

    def shortest_path(self, start, end):
        """Compute the shortest path between two locations using BFS."""
        if start == end:
            return 0

        queue = [(start, 0)]
        visited = {start}

        while queue:
            location, distance = queue.pop(0)

            if location == end:
                return distance

            if location in self.links:
                for neighbor in self.links[location]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))

        return float('inf')  # Return infinity if no path is found
