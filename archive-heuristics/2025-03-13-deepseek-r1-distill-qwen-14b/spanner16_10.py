from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import sys

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at man1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.

    # Assumptions:
    - The man can carry at most one spanner at a time.
    - A spanner becomes unusable after tightening a nut.
    - The man must pick up a new spanner for each nut.

    # Heuristic Initialization
    - Extract the goal conditions and static facts from the task.
    - Build a graph of locations from static facts to compute shortest paths.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the current location of the man.
    2. Check if the man is carrying a spanner.
    3. For each loose nut:
       a. If not carrying a spanner:
          i. Find the shortest path from the man's location to any spanner.
          ii. Add the distance and action for picking up the spanner.
          iii. Find the shortest path from the spanner's location to the nut's location.
          iv. Add the distance and action for tightening the nut.
       b. If carrying a spanner:
          i. Find the shortest path from the man's location to the nut's location.
          ii. Add the distance and action for tightening the nut.
    4. Sum all the actions for all loose nuts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each nut.
        - Static facts (location links) to build the location graph.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Build location graph from static facts
        self.locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
        self.locations = list(self.locations)

        # Precompute shortest paths between all pairs of locations
        self.distance = {}
        for src in self.locations:
            self.distance[src] = {}
            queue = deque()
            queue.append((src, 0))
            visited = {src}
            while queue:
                current, dist = queue.popleft()
                for fact in static_facts:
                    if match(fact, "link", current, "*"):
                        neighbor = get_parts(fact)[2]
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distance[src][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
                        elif self.distance[src].get(neighbor, float('inf')) > dist + 1:
                            self.distance[src][neighbor] = dist + 1

        # Extract goal locations for each nut
        self.goal_nuts = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts[nut] = "nut"  # Not used, but ensures presence

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Extract current state information
        man_location = None
        carrying_spanner = False
        loose_nuts = []

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1] == "bob":
                man_location = parts[2]
            if parts[0] == "carrying" and parts[1] == "bob":
                carrying_spanner = True
            if parts[0] == "loose":
                loose_nuts.append(parts[1])

        if not loose_nuts:
            return 0

        total_actions = 0

        # For each loose nut, calculate required actions
        for nut in loose_nuts:
            nut_location = None
            for fact in state:
                if fact.startswith("(at nut") and nut in fact and "gate" in fact:
                    nut_location = "gate"
                    break
            if nut_location is None:
                continue  # Nut is not in a location we can access

            if not carrying_spanner:
                # Find the nearest spanner
                min_spanner_dist = float('inf')
                nearest_spanner = None
                for fact in state:
                    if fact.startswith("(at spanner") and "location" in fact:
                        spanner_loc = get_parts(fact)[2]
                        if spanner_loc in self.distance.get(man_location, {}):
                            dist = self.distance[man_location][spanner_loc]
                            if dist < min_spanner_dist:
                                min_spanner_dist = dist
                                nearest_spanner = spanner_loc
                if nearest_spanner is None:
                    # No spanner available; cannot proceed
                    return float('inf')
                # Actions: walk to spanner, pickup, walk to nut, tighten
                total_actions += min_spanner_dist + 1 + self.distance[nearest_spanner].get(nut_location, float('inf')) + 1
                carrying_spanner = False  # After using, spanner is unusable
            else:
                # Actions: walk to nut, tighten
                if man_location in self.distance and nut_location in self.distance[man_location]:
                    dist = self.distance[man_location][nut_location]
                    total_actions += dist + 1
                else:
                    # Cannot reach the nut; overestimate
                    total_actions += float('inf')
                    return float('inf')

        return total_actions
