from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Test whether the fact matches the provided pattern elements.
    Wildcard '*' is supported (uses fnmatch for convenience).
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the secret_domain (hrpmxpjg). The domain models
    movable carriers (hctyyy8k objects, e.g. "v1") that can walk along a
    location graph (x6ibpitl nodes, e.g. "l1") and pick up/drop
    transportable objects (du66z5od, e.g. "p1"). Goals are typically
    of the form (sjuejzxf <package> <location>) and require placing each
    package at a desired location. Movement along location-edges uses
    the static predicate miud1izi, the predicate sjuejzxf encodes
    "object-at-location" (applies both to carriers and packages),
    and un606moq encodes "package carried by carrier".

    Assumptions
    -----------
    - Packages (du66z5od) can only be moved by carriers (hctyyy8k).
      The heuristic assumes carriers are those objects that appear in
      do9fmoyo facts or as the second argument of un606moq facts.
    - Moving a carrier one edge (miud1izi) costs 1 (the arsslvyt action).
    - Picking up a package (szsevy61) costs 1 action.
    - Dropping a package (tmoxe3x5) costs 1 action.
    - The heuristic is not admissible: it sums per-package minimal
      move+pick+drop estimates independently, but still aims to be
      informative and cheaply computable.
    - If graph distances are unknown or two locations are disconnected,
      a large finite penalty is used (so the heuristic remains finite).

    Heuristic Initialization
    ------------------------
    - Save the goal facts.
    - Extract the static location graph from static facts using
      the predicate miud1izi. Adjacency is taken as directed according
      to the static facts (this matches the domain action preconditions).
    - Extract per-package goal target locations from the goal set.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. If all goal facts are already satisfied in the state, return 0.
    2. Parse the state to determine:
       - Which carriers exist and where they are located (sjuejzxf carrier loc,
         do9fmoyo carrier hub facts and un606moq carrier references help find carriers).
       - Which packages are at which location (sjuejzxf package loc) or are
         currently carried by a carrier (un606moq package carrier). If a
         package is carried, its effective current location is the carrier's location.
    3. For each package that has a goal location:
       - If the package is already at its goal (either directly sjuejzxf pkg goal
         or carried by a carrier currently at goal), its cost contribution is 0.
       - Otherwise estimate the minimum cost to place it at the goal by considering:
         a) If it is currently carried by carrier H:
            cost = distance(H_loc, goal_loc) + drop_cost (1).
         b) If it is at location L:
            for each carrier H compute
              cost_candidate = distance(H_loc, L) + pick_cost (1) + distance(L, goal_loc) + drop_cost (1)
            and take the minimum across carriers.
       - Distances are shortest-path lengths over the miud1izi location graph
         (each edge counts as 1). If a required distance cannot be found
         (disconnected), a large finite penalty is used for that candidate.
    4. Sum the per-package costs to produce the final heuristic value.

    Note: This heuristic intentionally treats package moves independently
    (it does not reason about a carrier picking multiple packages in one trip),
    which makes it fast to compute while still informative enough to guide
    greedy best-first search in typical instances of this domain.
    """

    def __init__(self, task):
        """Extract static information and goals from the task."""
        self.goals = task.goals
        self.static = task.static

        # Build directed adjacency for locations from static miud1izi facts.
        # miud1izi <from> <to>
        self.adj = {}
        for fact in self.static:
            if match(fact, "miud1izi", "*", "*"):
                _, frm, to = get_parts(fact)
                self.adj.setdefault(frm, set()).add(to)

        # Map package -> goal location for goals that are sjuejzxf facts.
        self.goal_locations = {}
        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "sjuejzxf" and len(parts) >= 3:
                subj, loc = parts[1], parts[2]
                self.goal_locations[subj] = loc

        # Predefine common action costs (1 per action in this domain).
        self.move_cost = 1
        self.pick_cost = 1
        self.drop_cost = 1

        # Penalty for unreachable situations (finite but large).
        self.unreachable_penalty = 1000

    def shortest_distance(self, start, goal):
        """
        Directed BFS to compute shortest path length from start to goal
        on the adjacency graph built from miud1izi static facts.
        Returns an integer distance, or None if unreachable.
        """
        if start == goal:
            return 0
        if start not in self.adj:
            return None
        visited = {start}
        dq = deque([(start, 0)])
        while dq:
            node, d = dq.popleft()
            for succ in self.adj.get(node, ()):
                if succ == goal:
                    return d + 1
                if succ not in visited:
                    visited.add(succ)
                    dq.append((succ, d + 1))
        return None

    def __call__(self, node):
        """Compute the heuristic estimate for the given search node."""
        state = node.state

        # 1) If all goal facts already satisfied: heuristic 0.
        if self.goals <= state:
            return 0

        # 2) Parse state facts to collect:
        #    - carrier locations (hctyyy8k objects): inferred from do9fmoyo and sjuejzxf facts
        #    - package locations: sjuejzxf facts for packages or un606moq (carried)
        carrier_locations = {}  # carrier -> location (x6ibpitl)
        package_locations = {}  # package -> location (if on ground)
        package_carried_by = {}  # package -> carrier (if carried)

        # First, collect carriers by looking at do9fmoyo facts (they name hctyyy8k objects)
        carriers = set()
        for fact in state:
            if match(fact, "do9fmoyo", "*", "*"):
                parts = get_parts(fact)
                carrier = parts[1]
                carriers.add(carrier)
        # Also collect carriers that appear as second argument of un606moq in case do9fmoyo not present.
        for fact in state:
            if match(fact, "un606moq", "*", "*"):
                parts = get_parts(fact)
                _, carrier = parts[1], parts[2] if len(parts) > 2 else (parts[1] if len(parts) == 2 else None)
                # un606moq signature: (un606moq <package> <carrier>)
                # defensive: ensure we got two args
                if len(parts) >= 3:
                    carriers.add(parts[2])

        # Now parse sjuejzxf facts which indicate "object at location".
        for fact in state:
            if match(fact, "sjuejzxf", "*", "*"):
                parts = get_parts(fact)
                subj, loc = parts[1], parts[2]
                if subj in carriers:
                    # This is a carrier at location.
                    carrier_locations[subj] = loc
                else:
                    # Treat as a package on the ground at loc.
                    package_locations[subj] = loc

        # Parse un606moq facts to determine carried packages.
        for fact in state:
            if match(fact, "un606moq", "*", "*"):
                parts = get_parts(fact)
                pkg, carrier = parts[1], parts[2]
                package_carried_by[pkg] = carrier
                # If carrier location is known, package effective location is carrier's location.
                if carrier in carrier_locations:
                    package_locations[pkg] = carrier_locations[carrier]
                else:
                    # Carrier location unknown: leave package location absent but mark carried.
                    package_locations.pop(pkg, None)

        total_cost = 0

        # For each package that appears in the goals, estimate cost to put it at goal.
        for pkg, goal_loc in self.goal_locations.items():
            # Check if currently satisfied in state (either ground at goal or carried by carrier at goal).
            if match(f"(sjuejzxf {pkg} {goal_loc})", "sjuejzxf", pkg, goal_loc) and f"(sjuejzxf {pkg} {goal_loc})" in state:
                continue  # already at goal
            already_at_goal = False
            # Check carried-and-carrier-at-goal case:
            if pkg in package_carried_by:
                carrier = package_carried_by[pkg]
                carr_loc = carrier_locations.get(carrier)
                if carr_loc == goal_loc:
                    # A single drop suffices.
                    total_cost += self.drop_cost
                    continue

            # Determine current package location if available
            pkg_loc = package_locations.get(pkg)

            # If package is carried, consider cost from carrier location:
            if pkg in package_carried_by:
                carrier = package_carried_by[pkg]
                carr_loc = carrier_locations.get(carrier)
                if carr_loc is None:
                    # Unknown carrier location: impose penalty but finite.
                    total_cost += self.unreachable_penalty
                    continue
                dist = self.shortest_distance(carr_loc, goal_loc)
                if dist is None:
                    total_cost += self.unreachable_penalty
                else:
                    total_cost += dist * self.move_cost + self.drop_cost

            # If package is on ground at some location, need a carrier to come, pick, move, drop.
            elif pkg_loc is not None:
                best = None
                # If no carriers known -> cannot move: large penalty
                if not carrier_locations:
                    total_cost += self.unreachable_penalty
                    continue
                for carrier, carr_loc in carrier_locations.items():
                    # distance carrier -> package
                    d1 = self.shortest_distance(carr_loc, pkg_loc)
                    # distance package -> goal (carried by carrier)
                    d2 = self.shortest_distance(pkg_loc, goal_loc)
                    if d1 is None or d2 is None:
                        # candidate unreachable, skip but keep in mind
                        continue
                    candidate = d1 * self.move_cost + self.pick_cost + d2 * self.move_cost + self.drop_cost
                    if best is None or candidate < best:
                        best = candidate
                if best is None:
                    # no viable carrier-route found -> large penalty
                    total_cost += self.unreachable_penalty
                else:
                    total_cost += best
            else:
                # Package neither on ground nor known to be carried -> incomplete knowledge,
                # assign a penalty but finite.
                total_cost += self.unreachable_penalty

        return int(total_cost)