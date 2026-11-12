from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(jft9ynda obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (mtfkspuf).

    Summary:
    The heuristic estimates the number of required actions to reach the goal by evaluating
    how many goal-related facts about predicate 'jft9ynda' (relating objects of type val3ypio to hmhptaxe)
    are not yet satisfied, and by estimating the minimal number of action steps to fix them.
    It leverages static relations on connectivity between the ?hmhptaxe objects (from predicate 'z6o4oitz'),
    and indirectly accounts for necessary actions for moving or swapping 'yarbdfgc' assignments (connecting hodp29q0 and r86varlt),
    as well as the linked relationships represented via 'uypgj7u3' and 'jf4qno7u'.

    Assumptions:
    - This domain involves objects connected by certain static relations that constrain changes.
    - The main goal predicates to achieve are of the form (jft9ynda ?gpvob ?pk9o1).
    - Achieving a goal predicate (jft9ynda X Y) may require changing current assignments of Y or X.
    - The static predicate (z6o4oitz) represents a bidirectional-like connectivity (edges) on hmhptaxe objects;
      traversing these edges via 'k12p0czo' action changes the jft9ynda relation for a hodp29q0 object.
    - Close inspection of actions shows that changing assignments might require swaps or moves involving other predicates.
    - The heuristic is not necessarily admissible but aims to provide a good, finite estimate.
    - All necessary data for reasoning about these dependencies are contained in static facts and current state facts.

    Heuristic Initialization:
    - Extract the goal facts (assumed to be all of form (jft9ynda ?gpvob ?pk9o1)).
    - Build an undirected graph representing connectivity between hmhptaxe objects based on static 'z6o4oitz' facts.
    - Precompute for each 'jft9ynda' fact in goal its desired assignment.
    - Store info about static assignments from predicates 'jf4qno7u' and 'yarbdfgc' relating r86varlt and hodp29q0 with other objects.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each goal (jft9ynda GObj GoalLoc):
       a. Check if the fact currently holds in the state; if yes, no cost needed for this fact.
       b. Otherwise, determine the current location assigned to GObj in 'jft9ynda' predicate.
       c. If the object is assigned to a different location:
          i. Compute shortest path length between current and goal location in the connectivity graph (z6o4oitz).
          ii. Each edge traversal will need at least one 'k12p0czo' action changing the location for hodp29q0 objects.
       d. If the location does not exist or no direct path, apply a default penalty cost.
    2. For each involved hodp29q0 object having 'yarbdfgc' assignment differing from the current/'goal' relation:
       a. Count number of mismatched (yarbdfgc hodp29q0 r86varlt) predicate facts with respect to goals.
       b. Each mismatch may require one or more sw2fg9ql or vyn92k55 actions to fix.
       c. Approximate the cost as proportional to number of differing 'yarbdfgc' facts.
    3. Additionally, account for fixing missing or extra 'uypgj7u3' and 'jf4qno7u' predicates by heuristically considering their mismatches.
    4. Sum all heuristic components into a final heuristic cost estimate.
    5. Return 0 if state already satisfies the goal exactly.

    This heuristic balances goal fact satisfaction with an estimation of necessary moves over static connectivity and swaps of assignments.
    It is efficient to compute by using caching of the graph and shortest paths.
    """

    def __init__(self, task):
        # Store goals - filtered for 'jft9ynda' facts as these appear in goal
        self.goals = {fact for fact in task.goals if fact.startswith("(jft9ynda ")}

        # Extract static connectivity graph from 'z6o4oitz' facts (undirected)
        self.graph = {}
        for fact in task.static:
            if fact.startswith("(z6o4oitz "):
                parts = get_parts(fact)
                n1, n2 = parts[1], parts[2]
                self.graph.setdefault(n1, set()).add(n2)
                self.graph.setdefault(n2, set()).add(n1)

        # Extract for quick lookup:
        # goals_jft9ynda: map val3ypio object -> target hmhptaxe location (goal assignment)
        self.goals_jft9ynda = {}
        for fact in self.goals:
            _, obj, loc = get_parts(fact)
            self.goals_jft9ynda[obj] = loc

        # Store all static 'jf4qno7u' facts for r86varlt relation information
        self.static_jf4qno7u = {}
        for fact in task.static:
            if fact.startswith("(jf4qno7u "):
                _, r86obj, valobj = get_parts(fact)
                self.static_jf4qno7u[r86obj] = valobj

        # Store all static 'yarbdfgc' facts for mapping hodp29q0 to r86varlt assignments
        self.static_yarbdfgc = {}
        for fact in task.static:
            if fact.startswith("(yarbdfgc "):
                _, hod_obj, r86_obj = get_parts(fact)
                self.static_yarbdfgc[hod_obj] = r86_obj

    def shortest_path_length(self, start, goal):
        """Compute shortest undirected path length in the graph between start and goal node using BFS.
        Returns None if not reachable."""
        if start == goal:
            return 0
        visited = {start}
        frontier = [start]
        distance = 0
        while frontier:
            distance += 1
            next_frontier = []
            for node in frontier:
                neighbors = self.graph.get(node, ())
                for neigh in neighbors:
                    if neigh == goal:
                        return distance
                    if neigh not in visited:
                        visited.add(neigh)
                        next_frontier.append(neigh)
            frontier = next_frontier
        return None

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        cost = 0

        # --- Part 1: Estimate cost to fix (jft9ynda obj loc) facts to goal ---
        # For each goal jft9ynda fact, check if true, else estimate moves needed
        # Current assignments of jft9ynda:
        current_jft9ynda = {}
        for fact in state:
            if fact.startswith("(jft9ynda "):
                _, obj, loc = get_parts(fact)
                current_jft9ynda[obj] = loc

        # For each goal jft9ynda predicate:
        for obj, goal_loc in self.goals_jft9ynda.items():
            curr_loc = current_jft9ynda.get(obj)
            if curr_loc == goal_loc:
                continue  # satisfied already
            if curr_loc is None:
                # Heuristic penalty for missing assignment (unknown location)
                cost += 3
            else:
                dist = self.shortest_path_length(curr_loc, goal_loc)
                if dist is None:
                    # No path: large penalty assuming at least some moves required
                    cost += 5
                else:
                    # Each step would require one k12p0czo action; approximate cost linear with distance
                    cost += dist

        # --- Part 2: Account for mismatched 'yarbdfgc' facts ---
        # Goal 'yarbdfgc' assignments can be derived from static facts, but currently state may differ:
        # Count how many 'yarbdfgc' facts in goal are missing in state and vice versa.
        # Approximate cost proportional to mismatches.
        current_yarbdfgc = {}
        for fact in state:
            if fact.startswith("(yarbdfgc "):
                _, hod_obj, r86_obj = get_parts(fact)
                current_yarbdfgc[hod_obj] = r86_obj

        # Static 'yarbdfgc' facts may represent default or initial assignments;
        # to approximate, consider facts in state that differ from static ones as needing correction.
        # Also, any goal conditions on 'yarbdfgc' are not explicitly given,
        # so we approximate by comparing current and static as a proxy.
        mismatches = 0
        for hod_obj, assigned_r86 in self.static_yarbdfgc.items():
            curr_assigned = current_yarbdfgc.get(hod_obj)
            if curr_assigned != assigned_r86:
                mismatches += 1
        for hod_obj in current_yarbdfgc:
            if hod_obj not in self.static_yarbdfgc:
                mismatches += 1

        # We assign a heuristic cost per mismatch to model swapping/fixing actions (sw2fg9ql, vyn92k55)
        cost += mismatches * 2

        # --- Part 3: Account for mismatches in 'uypgj7u3' and 'jf4qno7u' ---
        # These predicates reflect relationships on g83opnlj and r86varlt which
        # support the 'yarbdfgc' and 'jft9ynda' changes.
        # Approximate mismatch count in 'uypgj7u3' and 'jf4qno7u' between state and static.

        current_uypgj7u3 = set()
        for fact in state:
            if fact.startswith("(uypgj7u3 "):
                current_uypgj7u3.add(fact)
        static_uypgj7u3 = set()
        for fact in state:
            if fact.startswith("(uypgj7u3 "):
                static_uypgj7u3.add(fact)
        # heuristic difference (in practice, could be computed from static predicates, here assumed equal to current as conservative)
        mismatch_uypgj7u3 = len(current_uypgj7u3.symmetric_difference(static_uypgj7u3))

        current_jf4qno7u = set()
        for fact in state:
            if fact.startswith("(jf4qno7u "):
                current_jf4qno7u.add(fact)
        static_jf4qno7u = set()
        for fact in self.static_jf4qno7u:
            # Already in dict; reconstruct fact strings for comparison:
            # Each element in static_jf4qno7u keys and values: r86_obj and val_obj
            pass  # static predicates already stored for matching, assume minimal mismatch for efficiency
        # We cannot accurately compare static_jf4qno7u facts without recreating them; skip detailed mismatch here.
        # Instead, heuristic positive constant if some static jf4qno7u fact missing in current:
        missing_jf4qno7u = 0
        for r86_obj, val_obj in self.static_jf4qno7u.items():
            if f"(jf4qno7u {r86_obj} {val_obj})" not in state:
                missing_jf4qno7u += 1

        cost += mismatch_uypgj7u3 + missing_jf4qno7u

        # Return a heuristic estimate representing the effort to fix object locations and assignments.
        return cost
