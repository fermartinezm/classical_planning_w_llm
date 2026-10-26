from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.

    # Assumptions:
    - The man can carry multiple spanners but needs one to tighten a nut.
    - The man must be at the same location as a loose nut to tighten it.
    - If the man is not carrying a spanner, he must pick one up before tightening a nut.

    # Heuristic Initialization
    - Extract the goal locations for each nut.
    - Extract static facts, particularly the links between locations.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all loose nuts and their current locations.
    2. For each loose nut:
        a. If the man is already carrying a spanner and is at the nut's location, only one action (tighten) is needed.
        b. If the man is not at the nut's location, calculate the number of moves required to reach the nut's location.
        c. If the man is not carrying a spanner, add the actions needed to pick up a spanner.
        d. Add the action to tighten the nut.
    3. Sum the actions for all nuts, considering that the man can carry multiple spanners and may not need to return to the starting point after the last nut.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each nut.
        - Static facts (links between locations).
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract location links to build adjacency list
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

        # Extract goal locations for each nut
        self.goal_locations = {}
        for goal in self.goals:
            predicate, nut, loc = get_parts(goal)
            if predicate == "tightened":
                self.goal_locations[nut] = loc

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Extract current locations of relevant objects
        current_locations = {}
        carrying_spanner = False
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == "carrying":
                man, spanner = parts[1], parts[2]
                carrying_spanner = True

        # Identify all loose nuts
        loose_nuts = []
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                if f"(loose {nut})" in state:
                    loose_nuts.append(nut)

        total_cost = 0

        # For each loose nut, calculate the required actions
        for nut in loose_nuts:
            nut_loc = None
            for fact in state:
                if match(fact, f"at {nut} *"):
                    nut_loc = get_parts(fact)[2]
                    break

            if nut_loc is None:
                continue  # Nut not present, skip

            # Check if nut is already at goal location and not tightened (shouldn't happen as per problem)
            if nut_loc != self.goal_locations[nut]:
                continue  # Nut needs to be moved, but in our domain, nuts are stationary

            # Determine if the man is at the nut's location
            man_loc = None
            for fact in state:
                if match(fact, "at bob *"):
                    man_loc = get_parts(fact)[2]
                    break

            # Calculate distance from man's current location to nut's location
            def shortest_path(start, end):
                visited = set()
                queue = [(start, 0)]
                while queue:
                    current, dist = queue.pop(0)
                    if current == end:
                        return dist
                    if current in visited:
                        continue
                    visited.add(current)
                    if current in self.location_links:
                        for neighbor in self.location_links[current]:
                            if neighbor not in visited:
                                queue.append((neighbor, dist + 1))
                return float('inf')  # No path found

            distance = shortest_path(man_loc, nut_loc) if man_loc else float('inf')

            # Check if the man is carrying a spanner
            if carrying_spanner:
                # Only need to move to nut's location and tighten
                total_cost += distance + 1
            else:
                # Need to move to a spanner's location, pick it up, then move to nut's location and tighten
                # Find the closest spanner
                closest_spanner = None
                min_spanner_dist = float('inf')
                for spanner in [s for s in current_locations if s.startswith('spanner')]:
                    spanner_loc = current_locations[spanner]
                    dist = shortest_path(man_loc, spanner_loc)
                    if dist < min_spanner_dist:
                        min_spanner_dist = dist
                        closest_spanner = spanner_loc

                # Move to spanner, pick it up, then move to nut
                total_cost += min_spanner_dist + 1 + distance + 1

        return total_cost
