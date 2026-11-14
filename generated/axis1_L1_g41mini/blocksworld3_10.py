from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern parts, wildcards "*" allowed.
    - Returns True if the fact matches the pattern; False otherwise.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal by
    analyzing the current assignments of facts related to the predicates `yfm3frbn`,
    `a4ha3ujl`, `jjtnwk3y`, and `s0899d5u`, as well as the global boolean predicate
    `ug241uf7`. It exploits the structure of the domain where actions toggle status predicates
    on objects and relate pairs via the `s0899d5u` predicate.

    Assumptions:
    - The domain involves objects that have mutually exclusive states expressed by
      `yfm3frbn`, `a4ha3ujl`, and `jjtnwk3y`.
    - The predicate `ug241uf7` acts as a global toggle flag enabling certain actions.
    - The goal predicate can require certain objects to hold specific predicates or
      relationships (`s0899d5u`), and these must be realized by moving or toggling states.
    - Transitioning states and establishing relations `s0899d5u` require combinations of
      the available actions, which always consume one action each.
    - The heuristic does not need to be admissible, but should aim for minimal node expansions.

    Heuristic Initialization:
    - Extract the goal facts to identify which objects must satisfy predicates `yfm3frbn`, `a4ha3ujl`, `jjtnwk3y`,
      and all `s0899d5u` relations that must be achieved.
    - Extract the set of all objects mentioned across facts for domain comprehension.
    - No static facts are used since `static` is empty.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify objects that currently have predicates `yfm3frbn`, `a4ha3ujl`, and `jjtnwk3y`.
    2. Identify which `s0899d5u` relations currently hold between objects.
    3. Determine the global flag `ug241uf7` state (True/False).
    4. For each goal predicate:
       - For `yfm3frbn` and `a4ha3ujl`: If an object lacks the goal predicate, estimate minimal toggling steps
         to achieve it, usually requiring at least 1 or 2 actions.
       - For `jjtnwk3y`: similarly estimate corrections if missing.
       - For `s0899d5u` relations, if a relation is missing, count at least 1-2 actions to establish it.
    5. Adjust the estimate based on the global flag, since some actions require `ug241uf7` to be True.
    6. Return the sum of these minimal action counts as the heuristic estimate.

    This heuristic approximates a lower bound on the number of required actions by counting
    unsatisfied goal predicates and relations, while also accounting for necessary toggling
    of global flags and intermediate states. This allows efficiently guiding the search without
    expensive explicit state-space exploration.
    """

    def __init__(self, task):
        # Store goal predicates for quick lookup
        self.goal_yfm3frbn = set()
        self.goal_a4ha3ujl = set()
        self.goal_jjtnwk3y = set()
        self.goal_s0899d5u = set()
        self.goal_ug241uf7 = False  # if goal requires ug241uf7, rare but just in case

        # Extract goals by predicate type
        for fact in task.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "yfm3frbn" and len(parts) == 2:
                self.goal_yfm3frbn.add(parts[1])
            elif pred == "a4ha3ujl" and len(parts) == 2:
                self.goal_a4ha3ujl.add(parts[1])
            elif pred == "jjtnwk3y" and len(parts) == 2:
                self.goal_jjtnwk3y.add(parts[1])
            elif pred == "s0899d5u" and len(parts) == 3:
                self.goal_s0899d5u.add((parts[1], parts[2]))
            elif pred == "ug241uf7":
                self.goal_ug241uf7 = True

        # Gather all objects from goals for completeness (to catch missing)
        self.goal_objects = set()
        for obj in self.goal_yfm3frbn:
            self.goal_objects.add(obj)
        for obj in self.goal_a4ha3ujl:
            self.goal_objects.add(obj)
        for obj in self.goal_jjtnwk3y:
            self.goal_objects.add(obj)
        for pair in self.goal_s0899d5u:
            self.goal_objects.update(pair)

    def __call__(self, node):
        state = node.state

        # Current predicate sets by object
        cur_yfm3frbn = {get_parts(f)[1] for f in state if match(f, "yfm3frbn", "*")}
        cur_a4ha3ujl = {get_parts(f)[1] for f in state if match(f, "a4ha3ujl", "*")}
        cur_jjtnwk3y = {get_parts(f)[1] for f in state if match(f, "jjtnwk3y", "*")}
        cur_s0899d5u = {(get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "s0899d5u", "*", "*")}
        cur_ug241uf7 = "(ug241uf7)" in state

        # If already goal state, heuristic is zero
        if (
            self.goal_yfm3frbn <= cur_yfm3frbn
            and self.goal_a4ha3ujl <= cur_a4ha3ujl
            and self.goal_jjtnwk3y <= cur_jjtnwk3y
            and self.goal_s0899d5u <= cur_s0899d5u
            and (not self.goal_ug241uf7 or cur_ug241uf7)
        ):
            return 0

        cost = 0

        # For each object in goal_yfm3frbn, if missing in current, this likely requires toggling
        for obj in self.goal_yfm3frbn:
            if obj not in cur_yfm3frbn:
                # To achieve yfm3frbn, object must either be turned from jjtnwk3y/a4ha3ujl 
                # Typically requires 1 or 2 actions depending on current state
                if obj in cur_jjtnwk3y or obj in cur_a4ha3ujl:
                    cost += 1
                else:
                    # If object currently lacks any related predicates, likely 2 steps needed
                    cost += 2

        # For each object in goal_a4ha3ujl, if missing in current, similar logic
        for obj in self.goal_a4ha3ujl:
            if obj not in cur_a4ha3ujl:
                if obj in cur_jjtnwk3y or obj in cur_yfm3frbn:
                    cost += 1
                else:
                    cost += 2

        # For each jjtnwk3y goal object, if missing, estimate cost
        for obj in self.goal_jjtnwk3y:
            if obj not in cur_jjtnwk3y:
                # Achieving jjtnwk3y from yfm3frbn or through relation usually costs 1 or 2 actions
                # Check if yfm3frbn currently held by the object -- can switch directly
                if obj in cur_yfm3frbn or obj in cur_a4ha3ujl:
                    cost += 1
                else:
                    cost += 2

        # For each missing s0899d5u relation in goal, each likely requires at least 1-2 actions
        for rel in self.goal_s0899d5u:
            if rel not in cur_s0899d5u:
                # Establishing relation requires yfm3frbn or jjtnwk3y states on participants plus global flag usage
                cost += 2  # safe estimate: one for changing states + one for linking

        # If goal requires ug241uf7 and not currently set, likely 1 action to set
        if self.goal_ug241uf7 and not cur_ug241uf7:
            cost += 1

        # Additional heuristic refinement:
        # The global predicate ug241uf7 gates many actions.
        # If the global flag is currently false but needed for most transitions, add +1 for toggling it.
        if not cur_ug241uf7:
            # Count of predicates that need changes but ug241uf7 is off
            relevant_obj_changes = 0
            for obj in self.goal_yfm3frbn:
                if obj not in cur_yfm3frbn:
                    relevant_obj_changes += 1
            for obj in self.goal_a4ha3ujl:
                if obj not in cur_a4ha3ujl:
                    relevant_obj_changes += 1
            for obj in self.goal_jjtnwk3y:
                if obj not in cur_jjtnwk3y:
                    relevant_obj_changes += 1
            if relevant_obj_changes > 0 or any(rel not in cur_s0899d5u for rel in self.goal_s0899d5u):
                cost += 1  # to toggle ug241uf7 at some point in plan

        return cost
