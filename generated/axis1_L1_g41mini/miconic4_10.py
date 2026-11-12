from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact by stripping parentheses and splitting.
    Example: "(kut2ksnk p1 f6)" -> ["kut2ksnk", "p1", "f6"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches a pattern with wildcards.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the required number of actions to reach the goal state in the secret_domain.
    The main goal is to make all specified nxbd1hc3 objects hold the predicate (a7sb56om ?praz5).
    To achieve this, each such object transitions through these key stages:
      1. Initially, it may be linked by (kut2ksnk ?praz5 ?w3okg).
      2. Applying action diwlmgg9 switches the object from (kut2ksnk) to (f2xo7hb6).
      3. Then, if linked by (klw11kjf ?praz5 ?w3okg) and having (f2xo7hb6), action y7nik9q2 produces (a7sb56om).
    The predicate (omggbn9z ?w3okg) models locations (like rooms or nodes)
    and (acovodrf ?loc1 ?loc2) connects them bidirectionally.

    Assumptions
    -----------
    - The goal is always to have all required nxbd1hc3 objects satisfy (a7sb56om ?praz5).
    - There may be multiple t4rdxh32 objects representing locations.
    - Movement between locations is possible via acovodrf edges (both directions must be considered).
    - Actions diwlmgg9 and y7nik9q2 transform object states, requiring presence at specific linked locations.
    - Movement actions (qs8pdrqj, i0veo5kv) allow relocating the robot/location marker (omggbn9z).
    - The heuristic is not required to be admissible, but should be informative to minimize node expansions.

    Heuristic Initialization
    ------------------------
    - Extract static connectivity graph from (acovodrf loc1 loc2) facts, building undirected graph of locations.
    - Extract static links from (kut2ksnk obj loc) and (klw11kjf obj loc) to know where each object can be acted upon.
    - Store the set of objects nxbd1hc3 of interest (those found in goal conditions (a7sb56om ?praz5)).
    - Static facts (klw11kjf, kut2ksnk, acovodrf) never change.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each object that needs (a7sb56om) but does not have it yet:
      1. Identify its current "stage":
         - Stage 3: Already (a7sb56om) → no cost.
         - Stage 2: Has (f2xo7hb6) → requires action y7nik9q2 at a location linked by (klw11kjf ?obj ?loc).
           Preconditions: robot/location marker has to be at that location with (omggbn9z ?loc).
           Cost approx: 
             - cost to move robot from current location to one of the klw11kjf locations (shortest path)
             - plus 1 action for y7nik9q2.
         - Stage 1: Otherwise, if (kut2ksnk), need to apply diwlmgg9 first at one of the kut2ksnk linked locations,
           then proceed as stage 2.
           Cost approx:
             - cost to move robot to that kut2ksnk location,
             - plus 1 for diwlmgg9,
             - then cost to move to one of the klw11kjf locations,
             - plus 1 for y7nik9q2.
      2. Estimate minimal distance between the robot's current location (locations where (omggbn9z) holds) and required locations,
         using shortest path in the undirected connectivity graph.
      3. Movement between adjacent locations costs 1 per action (qs8pdrqj or i0veo5kv).
      4. Sum these costs for all needed objects as heuristic.

    This heuristic is finite for solvable states because:
      - The connectivity graph is always defined from static facts.
      - Actions exist to move the robot to needed locations and apply the required actions.
      - Each needed transition costs at least one movement step plus one domain action.

    The heuristic is zero if and only if all goal (a7sb56om) facts are true in the state.
    """

    def __init__(self, task):
        # Goals must be all (a7sb56om ?obj) facts.
        self.goal_objects = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "a7sb56om":
                self.goal_objects.add(parts[1])

        # Build undirected location graph from static (acovodrf loc1 loc2).
        self.graph = {}
        self.kut2ksnk_links = {}  # obj -> set of locations where (kut2ksnk obj loc)
        self.klw11kjf_links = {}  # obj -> set of locations where (klw11kjf obj loc)
        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "acovodrf":
                loc1, loc2 = parts[1], parts[2]
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1)
            elif pred == "kut2ksnk":
                obj, loc = parts[1], parts[2]
                self.kut2ksnk_links.setdefault(obj, set()).add(loc)
            elif pred == "klw11kjf":
                obj, loc = parts[1], parts[2]
                self.klw11kjf_links.setdefault(obj, set()).add(loc)

        # Precompute all locations indexed for fast membership.
        self.all_locations = set(self.graph.keys())

    def shortest_distance(self, start_locs, goal_locs):
        """
        Compute minimal distance (number of edges) in undirected graph between any of the start_locs and goal_locs.
        Use BFS from all start_locs simultaneously.
        Return infinite (large int) if no path exists.
        """
        from collections import deque

        if not start_locs or not goal_locs:
            return float("inf")

        visited = set()
        queue = deque()
        for loc in start_locs:
            if loc in goal_locs:
                return 0
            if loc in self.all_locations:
                queue.append((loc, 0))
                visited.add(loc)

        while queue:
            loc, dist = queue.popleft()
            for neigh in self.graph.get(loc, []):
                if neigh in visited:
                    continue
                if neigh in goal_locs:
                    return dist + 1
                visited.add(neigh)
                queue.append((neigh, dist + 1))

        return float("inf")

    def __call__(self, node):
        state = node.state

        # Quick check: goal reached → heuristic 0.
        # This also enforces heuristic zero iff goal is reached.
        if self.goal_objects.issubset(
            {
                parts[1]
                for fact in state
                for parts in [get_parts(fact)]
                if parts[0] == "a7sb56om"
            }
        ):
            return 0

        # Find robot current locations (where (omggbn9z ?loc) holds).
        current_robot_locs = {
            parts[1]
            for fact in state
            for parts in [get_parts(fact)]
            if parts[0] == "omggbn9z"
        }
        if not current_robot_locs:
            # Robot location unknown → heuristic infinite.
            return float("inf")

        total_cost = 0

        # For each goal object that is not achieved yet, estimate cost.
        for obj in self.goal_objects:
            # Check goal predicate.
            goal_achieved = any(
                fact.startswith(f"(a7sb56om {obj})") for fact in state
            )
            if goal_achieved:
                continue  # no cost for this obj

            # Determine its current stage:
            # Stage 2 if (f2xo7hb6 obj)
            stage2 = any(fact == f"(f2xo7hb6 {obj})" for fact in state)

            # Stage 1 if (kut2ksnk obj loc) holds and not stage2
            kut2ksnk_locs = self.kut2ksnk_links.get(obj, set())
            klw11kjf_locs = self.klw11kjf_links.get(obj, set())

            if stage2:
                # Need to move robot to any klw11kjf location to apply y7nik9q2.
                # Cost: movement + 1 action.
                dist = self.shortest_distance(current_robot_locs, klw11kjf_locs)
                if dist == float("inf"):
                    return float("inf")  # unreachable location, problem unsolvable here
                total_cost += dist + 1
                # Assume after this action robot and object state updated appropriately,
                # so no further cost needed for this obj.
            else:
                # Need to do diwlmgg9 first at a kut2ksnk location.
                if not kut2ksnk_locs or not klw11kjf_locs:
                    # No valid static info → no solution or heuristic infinite.
                    return float("inf")

                # Cost: move to kut2ksnk loc + 1 (diwlmgg9) + move to klw11kjf loc + 1 (y7nik9q2)
                dist_to_kut2ksnk = self.shortest_distance(current_robot_locs, kut2ksnk_locs)
                if dist_to_kut2ksnk == float("inf"):
                    return float("inf")

                # We assume after diwlmgg9, robot is at that location for free (optimistic).
                # So robot location updates to the used kut2ksnk location.
                # Estimate cost from kut2ksnk loc to klw11kjf loc (second move).
                dist_kut2ksnk_to_klw11kjf = self.shortest_distance(kut2ksnk_locs, klw11kjf_locs)
                if dist_kut2ksnk_to_klw11kjf == float("inf"):
                    return float("inf")

                total_cost += dist_to_kut2ksnk + 1 + dist_kut2ksnk_to_klw11kjf + 1

                # Update robot location approx to klw11kjf location for next object.
                # To avoid expensive per-object chaining, we will NOT update current_robot_locs here,
                # just accept heuristic inaccuracy.

        # Return the sum of all objects' estimated costs.
        # This may overestimate (non-admissible) but guides search well.
        return total_cost
