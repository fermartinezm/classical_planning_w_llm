from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at_kitchen_bread content-portion1 bread-portion2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the `secret_domain` domain ("content-portion").

    Summary:
    This heuristic estimates the approximate number of actions required to achieve the goal from the current state.
    The goal generally involves serving sandwiches or no-gluten bread portions on trays and fulfilling children 
    sandwich requests with gluten allergy considerations. 
    The heuristic counts the outstanding goal requirements that need fulfillment, estimating necessary preparations 
    such as making sandwiches (gluten or non-gluten), moving trays, and serving actions.

    Assumptions:
    - Each goal fact corresponds to a required final state of bread-portion or sandwich on a tray or a no-gluten sandwich condition.
    - Actions to make sandwiches or serve them depend on gluten/no-gluten content, waiting places, trays, and allergy predicates.
    - Preconditions for actions include facts such as "waiting" at places, "at_kitchen_bread", "at_kitchen_content", tray positions.
    - Every unmet goal fact generally requires at least one action to achieve (like making or serving a sandwich or preparing bread).
    - The heuristic is not admissible but tries to estimate the minimal effort by counting unsatisfied goal predicates and static partial supports.
    - The heuristic relies on static information about served bread portions and their relationships, extracted from static facts.

    Heuristic Initialization:
    - Extract goal facts from the task.
    - Extract static facts relevant to bread portions served pairs ("served") and no-gluten properties.
    - Analyze static properties like which bread portions can serve which others.
    - Extract sets of objects by type from static or facts seen (content-portions, bread-portions, sandwiches, trays, places, children).
    - Prepare data structures for quick membership and relation checks during heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify unmet goal conditions in the current state.
    2. For each unmet goal fact (mostly involving no_gluten_bread-2, no_gluten_sandwich-2, and similar):
       - Count how many trays/bread-portion/sandwich combinations still need preparation.
    3. Count how many "waiting" places exist—since serving or making sandwiches requires waiting places free to proceed.
    4. Check the state for "at_kitchen_bread", "at_kitchen_content", "allergic_gluten", and relevant intermediate indicators that suggest progress.
    5. Estimate the number of "make_sandwich", "make_sandwich_no_gluten", "put_on_tray", or "serve_sandwich" actions needed based on the deficiencies.
    6. Include expected serving and moving tray operations by counting unmet serving predicates or trays missing their expected contents.
    7. Sum these rough counts to produce the heuristic value.
    8. Return 0 when all goals are satisfied.

    Notes:
    - The heuristic is designed to guide greedy best-first search with efficient computation.
    - It avoids expensive path or graph search, operating mainly on set membership lookups.
    - It counts necessary actions approximately, ignoring some complex dependencies and parallelism to improve speed.
    """

    def __init__(self, task):
        """Initialize by extracting goals and static relationships."""
        self.goals = task.goals
        static_facts = task.static

        # Extract all "served" relations for quick reference.
        self.served_pairs = set()
        for fact in static_facts:
            if match(fact, "served", "*", "*"):
                parts = get_parts(fact)
                self.served_pairs.add((parts[1], parts[2]))

        # Extract which bread portions and content-portions are gluten-free from static facts.
        self.no_gluten_breads = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_contents = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "no_gluten_content", "*")
        }
        self.no_gluten_sandwich_children = set()
        for fact in static_facts:
            if match(fact, "no_gluten_sandwich", "*", "*"):
                parts = get_parts(fact)
                self.no_gluten_sandwich_children.add((parts[1], parts[2]))

        # Pre-fetch the goal facts by predicate to ease processing at runtime.
        self.goal_no_gluten_bread_2 = set()
        self.goal_no_gluten_sandwich_2 = set()
        for g in self.goals:
            parts = get_parts(g)
            predicate = parts[0]
            if predicate == "no_gluten_bread-2":
                self.goal_no_gluten_bread_2.add((parts[1], parts[2]))
            elif predicate == "no_gluten_sandwich-2":
                self.goal_no_gluten_sandwich_2.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goals fulfilled, return 0 immediately.
        if self.goals <= state:
            return 0

        # Count unmet goal "no_gluten_bread-2" facts.
        unmet_no_gluten_bread_2 = 0
        for content_portion, tray in self.goal_no_gluten_bread_2:
            fact_str = f"(no_gluten_bread-2 {content_portion} {tray})"
            if fact_str not in state:
                unmet_no_gluten_bread_2 += 1

        # Count unmet goal "no_gluten_sandwich-2" facts.
        unmet_no_gluten_sandwich_2 = 0
        for bread_portion in self.goal_no_gluten_sandwich_2:
            fact_str = f"(no_gluten_sandwich-2 {bread_portion})"
            if fact_str not in state:
                unmet_no_gluten_sandwich_2 += 1

        # Count how many places are currently waiting (places needed for making or serving).
        waiting_places = sum(1 for f in state if match(f, "waiting", "*"))

        # Count how many waiting-2 facts for content-portions and bread-portions exist.
        waiting_2_count = sum(1 for f in state if match(f, "waiting-2", "*", "*"))

        # Count how many sandwiches are currently on trays (ontray).
        ontray_count = sum(1 for f in state if match(f, "ontray", "*", "*"))

        # Count how many allergic_gluten predicates exist in state.
        allergic_gluten_count = sum(1 for f in state if match(f, "allergic_gluten", "*", "*"))

        # Count how many not_allergic_gluten predicates exist
        not_allergic_gluten_count = sum(1 for f in state if match(f, "not_allergic_gluten", "*", "*"))
        not_allergic_gluten_2_count = sum(1 for f in state if match(f, "not_allergic_gluten-2", "*", "*"))

        # Count how many sandwiches currently at kitchen sandwich places.
        at_kitchen_sandwich_count = sum(1 for f in state if match(f, "at_kitchen_sandwich", "*"))
        at_kitchen_sandwich_2_count = sum(1 for f in state if match(f, "at_kitchen_sandwich-2", "*", "*"))

        # Count how many bread and content portions are currently at kitchen positions and gluten/no-gluten status.
        at_kitchen_bread_count = sum(1 for f in state if match(f, "at_kitchen_bread", "*", "*"))
        at_kitchen_bread_2_count = sum(1 for f in state if match(f, "at_kitchen_bread-2", "*", "*", "*"))
        at_kitchen_content_count = sum(1 for f in state if match(f, "at_kitchen_content", "*"))
        at_kitchen_content_2_count = sum(1 for f in state if match(f, "at_kitchen_content-2", "*"))

        # Count no_gluten_content and no_gluten_content-2 facts present in the state.
        no_gluten_content_count = sum(1 for f in state if match(f, "no_gluten_content", "*"))
        no_gluten_content_2_count = sum(1 for f in state if match(f, "no_gluten_content-2", "*"))

        # Count number of bread portions currently served (served predicate).
        served_count = sum(1 for f in state if match(f, "served", "*", "*"))

        # Count number of notexist facts.
        notexist_count = sum(1 for f in state if match(f, "notexist", "*", "*", "*"))
        notexist_2_count = sum(1 for f in state if match(f, "notexist-2", "*"))

        # Heuristic estimate:
        # Best effort to cover the complexity:
        # - Each unmet no_gluten_bread_2 likely requires at least 1 action (making or moving tray).
        # - Each unmet no_gluten_sandwich_2 likely requires at least 1 action (making sandwich).
        # - Each waiting place or waiting-2 fact suggests preparation steps.
        # - Allergic_gluten predicates require corrective action, so each counts as 1.
        # - Sandwiches not on trays or trays missing contents add actions.
        # - Lack of content or bread at kitchen spots suggests prep steps.
        # Some actions are related (e.g. make sandwich and serve sandwich), so a weighted sum.

        # Weighting rationale (tunable):
        # - Unmet final goals weighted highest as they are directly required.
        # - Waiting places and waiting-2 reflect readiness or blocking => moderate weight.
        # - Allergic gluten and not_allergic gluten with counts moderate.
        # - At kitchen and on trays contribute moderate to low weight to account for preparation.

        heuristic_value = (
            5 * unmet_no_gluten_bread_2 +
            4 * unmet_no_gluten_sandwich_2 +
            3 * waiting_places +
            3 * waiting_2_count +
            3 * allergic_gluten_count +
            2 * not_allergic_gluten_count +
            2 * not_allergic_gluten_2_count +
            2 * (at_kitchen_sandwich_count + at_kitchen_sandwich_2_count) +
            2 * ontray_count +
            1 * at_kitchen_bread_count +
            1 * at_kitchen_bread_2_count +
            1 * at_kitchen_content_count +
            1 * at_kitchen_content_2_count +
            1 * no_gluten_content_count +
            1 * no_gluten_content_2_count +
            1 * served_count +
            1 * notexist_count +
            1 * notexist_2_count
        )

        # Ensure heuristic is never zero unless goal reached (already handled above).
        if heuristic_value == 0:
            heuristic_value = 1

        return heuristic_value
