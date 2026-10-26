from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the cost
    of walking, picking up spanners, and tightening nuts.

    # Assumptions
    - The heuristic assumes that the man can carry at most two spanners at a time.
    - It assumes that a spanner can only be used once.
    - It assumes that the shortest path between locations is used.

    # Heuristic Initialization
    - The heuristic initializes the locations of all objects (man, spanners, nuts).
    - It also extracts the link information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current state information: man's location, carried spanners,
       available spanners, loose nuts, and tightened nuts.
    2. If all nuts are tightened, the heuristic value is 0.
    3. For each loose nut, estimate the cost to tighten it:
       a. Find the closest usable spanner.
       b. Estimate the cost to walk to the spanner.
       c. Estimate the cost to walk to the nut.
       d. Add the cost of picking up the spanner and tightening the nut.
    4. Sum the costs for all loose nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract link information from static facts
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
        """Estimate the number of actions needed to tighten all loose nuts."""
        state = node.state

        # Extract information from the current state
        man_location = None
        carried_spanners = []
        available_spanners = {}
        loose_nuts = []
        tightened_nuts = []

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj, location = parts[1], parts[2]
                if obj == "bob":
                    man_location = location
                elif "spanner" in obj:
                    available_spanners[obj] = location
                elif "nut" in obj:
                    loose_nuts.append((obj, location))
            elif match(fact, "carrying", "*", "*"):
                parts = get_parts(fact)
                man, spanner = parts[1], parts[2]
                if man == "bob":
                    carried_spanners.append(spanner)
            elif match(fact, "usable", "*"):
                parts = get_parts(fact)
                spanner = parts[1]
                if spanner not in available_spanners:
                    available_spanners[spanner] = None  # Location unknown
            elif match(fact, "tightened", "*"):
                parts = get_parts(fact)
                nut = parts[1]
                tightened_nuts.append(nut)
            elif match(fact, "loose", "*"):
                parts = get_parts(fact)
                nut = parts[1]
                if (nut, None) not in loose_nuts:
                    loose_nuts.append((nut, None))

        # Check if all nuts are tightened
        all_goals_met = True
        for goal in self.goals:
            if goal not in state:
                all_goals_met = False
                break

        if all_goals_met:
            return 0

        # Calculate the heuristic value
        total_cost = 0
        for nut, nut_location in loose_nuts:
            if nut in tightened_nuts:
                continue

            # Find the closest usable spanner
            closest_spanner = None
            min_distance = float('inf')

            for spanner, spanner_location in available_spanners.items():
                if spanner in carried_spanners:
                    closest_spanner = spanner
                    min_distance = 0
                    break
                if spanner_location is None:
                    continue

                # Calculate the distance between the man and the spanner
                if man_location is not None:
                    distance_man_spanner = self.shortest_path_distance(man_location, spanner_location)
                else:
                    distance_man_spanner = float('inf')

                if distance_man_spanner < min_distance:
                    min_distance = distance_man_spanner
                    closest_spanner = spanner

            if closest_spanner is None:
                # No usable spanner available
                return float('inf')

            # Estimate the cost to walk to the spanner
            if man_location is not None and available_spanners[closest_spanner] is not None:
                distance_man_spanner = self.shortest_path_distance(man_location, available_spanners[closest_spanner])
                total_cost += distance_man_spanner
                man_location = available_spanners[closest_spanner]
            else:
                return float('inf')

            # Estimate the cost to walk to the nut
            if nut_location is not None:
                distance_spanner_nut = self.shortest_path_distance(man_location, nut_location)
                total_cost += distance_spanner_nut
            else:
                return float('inf')

            # Add the cost of picking up the spanner and tightening the nut
            total_cost += 2  # pickup_spanner + tighten_nut
            available_spanners.pop(closest_spanner)

        return total_cost

    def shortest_path_distance(self, start, end):
        """Calculate the shortest path distance between two locations."""
        if start == end:
            return 0

        if start not in self.links or end not in self.links:
            return float('inf')

        queue = [(start, 0)]
        visited = {start}

        while queue:
            location, distance = queue.pop(0)

            if location == end:
                return distance

            for neighbor in self.links[location]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf')
