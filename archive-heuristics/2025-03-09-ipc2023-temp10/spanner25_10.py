from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at nut1 gate)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the cost
    of walking, picking up a spanner, and tightening a nut.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A spanner must be usable to tighten a nut.
    - The heuristic assumes that the agent will always pick up the closest usable spanner.
    - It also assumes that the agent will always tighten the closest loose nut.

    # Heuristic Initialization
    - Extract the locations of all objects (man, spanners, nuts) from the initial state.
    - Identify the links between locations.
    - Identify the usable spanners.
    - Identify the loose nuts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state. If so, return 0.
    2. Identify the man's current location.
    3. Identify the loose nuts.
    4. For each loose nut:
       a. Find the closest usable spanner.
       b. Estimate the cost of:
          i. Walking to the spanner.
          ii. Picking up the spanner.
          iii. Walking to the nut.
          iv. Tightening the nut.
    5. Sum the costs for all loose nuts.
    6. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract link information
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                if l1 not in self.links:
                    self.links[l1] = []
                self.links[l1].append(l2)
                if l2 not in self.links:
                    self.links[l2] = []
                self.links[l2].append(l1)

    def __call__(self, node):
        """Estimate the minimum cost to tighten all loose nuts."""
        state = node.state

        if self.goal_reached(state):
            return 0

        # Extract man's location
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] == "bob":
                    man_location = parts[2]
                    break

        # Extract loose nuts
        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                parts = get_parts(fact)
                loose_nuts.append(parts[1])

        # Extract usable spanners
        usable_spanners = []
        for fact in state:
            if match(fact, "usable", "*"):
                parts = get_parts(fact)
                usable_spanners.append(parts[1])

        # Extract spanner locations
        spanner_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] in usable_spanners:
                    spanner_locations[parts[1]] = parts[2]

        # Extract nut locations
        nut_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] in loose_nuts:
                    nut_locations[parts[1]] = parts[2]

        # Extract if man is carrying a spanner
        carrying_spanner = None
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                parts = get_parts(fact)
                carrying_spanner = parts[2]
                break

        total_cost = 0
        for nut in loose_nuts:
            nut_location = nut_locations[nut]
            # If already carrying a usable spanner, check if it can be used
            if carrying_spanner is not None and carrying_spanner in usable_spanners:
                # Check if man and nut are at the same location
                if man_location == nut_location:
                    total_cost += 1 # Tighten nut
                else:
                    total_cost += self.shortest_path_length(man_location, nut_location) # Walk to nut
                    total_cost += 1 # Tighten nut
            else:
                # Find closest usable spanner
                closest_spanner = None
                min_distance = float('inf')
                for spanner, spanner_location in spanner_locations.items():
                    distance = self.shortest_path_length(man_location, spanner_location)
                    if distance < min_distance:
                        min_distance = distance
                        closest_spanner = spanner
                        closest_spanner_location = spanner_location

                # Estimate cost
                if closest_spanner is not None:
                    total_cost += self.shortest_path_length(man_location, closest_spanner_location) # Walk to spanner
                    total_cost += 1 # Pick up spanner
                    total_cost += self.shortest_path_length(closest_spanner_location, nut_location) # Walk to nut
                    total_cost += 1 # Tighten nut
                else:
                    # No usable spanner available
                    return float('inf')

        return total_cost

    def goal_reached(self, state):
        """Check if the goal has been reached."""
        return self.goals <= state

    def shortest_path_length(self, start, end):
        """Compute the shortest path length between two locations."""
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

        return float('inf')
