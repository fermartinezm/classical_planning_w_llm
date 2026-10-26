from dataclasses import dataclass

BENCHMARKS = "benchmarks/"
AUTOSCALE = "/".join([BENCHMARKS, "autoscale", "training"])
IPC_2023 = "/".join([BENCHMARKS, "ipc2023-learning", "training"])
HIDDEN_DOMAINS = "/".join([BENCHMARKS, "hidden/", "training"])
UNSEEN_DOMAINS = "/".join([BENCHMARKS, "unseen-domains-strips/", "training"])

@dataclass
class DomainSuite:
    name: str
    domain: str
    instance1: str
    instance2: str
    state : str
    static : str


SUITES = {
    "blocksworld": DomainSuite(
        "blocksworld",
        f"{IPC_2023}/blocksworld/domain.pddl",
        f"{IPC_2023}/blocksworld/p01.pddl",
        f"{IPC_2023}/blocksworld/p99.pddl",
        f"{IPC_2023}/blocksworld/example-state.out",
        f"{IPC_2023}/blocksworld/example-static.out",
    ),
    "blocksworld-obfuscated": DomainSuite(
        "secret_domain",
        f"{HIDDEN_DOMAINS}/blocksworld/domain.pddl",
        f"{HIDDEN_DOMAINS}/blocksworld/p01.pddl",
        f"{HIDDEN_DOMAINS}/blocksworld/p99.pddl",
        f"{HIDDEN_DOMAINS}/blocksworld/example-state.out",
        f"{HIDDEN_DOMAINS}/blocksworld/example-static.out",
    ),
    "blocksworld-obfuscated-simple": DomainSuite(
        "secret_domain",
        f"{HIDDEN_DOMAINS}/blocksworld-simple/domain.pddl",
        f"{HIDDEN_DOMAINS}/blocksworld-simple/p01.pddl",
        f"{HIDDEN_DOMAINS}/blocksworld-simple/p99.pddl",
        f"{HIDDEN_DOMAINS}/blocksworld-simple/example-state.out",
        f"{HIDDEN_DOMAINS}/blocksworld-simple/example-static.out",
    ),
    "childsnack": DomainSuite(
        "childsnacks",
        f"{IPC_2023}/childsnack/domain.pddl",
        f"{IPC_2023}/childsnack/p01.pddl",
        f"{IPC_2023}/childsnack/p99.pddl",
        f"{IPC_2023}/childsnack/example-state.out",
        f"{IPC_2023}/childsnack/example-static.out",
    ),
    "depots": DomainSuite(
        "depots",
        f"{AUTOSCALE}/depots/domain.pddl",
        f"{AUTOSCALE}/depots/p01.pddl",
        f"{AUTOSCALE}/depots/p30.pddl",
        f"{AUTOSCALE}/depots/example-state.out",
        f"{AUTOSCALE}/depots/example-static.out",
    ),
    "driverlog": DomainSuite(
        "driverlog",
        f"{AUTOSCALE}/driverlog/domain.pddl",
        f"{AUTOSCALE}/driverlog/p01.pddl",
        f"{AUTOSCALE}/driverlog/p30.pddl",
        f"{AUTOSCALE}/driverlog/example-state.out",
        f"{AUTOSCALE}/driverlog/example-static.out",
    ),
    "floortile": DomainSuite(
        "floortile",
        f"{IPC_2023}/floortile/domain.pddl",
        f"{IPC_2023}/floortile/p01.pddl",
        f"{IPC_2023}/floortile/p99.pddl",
        f"{IPC_2023}/floortile/example-state.out",
        f"{IPC_2023}/floortile/example-static.out",
    ),
    "freecell": DomainSuite(
        "freecell",
        f"{AUTOSCALE}/freecell/domain.pddl",
        f"{AUTOSCALE}/freecell/p01.pddl",
        f"{AUTOSCALE}/freecell/p30.pddl",
        f"{AUTOSCALE}/freecell/example-state.out",
        f"{AUTOSCALE}/freecell/example-static.out",
    ),

    "grid": DomainSuite(
        "grid",
        f"{AUTOSCALE}/grid/domain.pddl",
        f"{AUTOSCALE}/grid/p01.pddl",
        f"{AUTOSCALE}/grid/p30.pddl",
        f"{AUTOSCALE}/grid/example-state.out",
        f"{AUTOSCALE}/grid/example-static.out",
    ),
   "gripper": DomainSuite(
        "gripper",
        f"{AUTOSCALE}/gripper/domain.pddl",
        f"{AUTOSCALE}/gripper/p01.pddl",
        f"{AUTOSCALE}/gripper/p30.pddl",
        f"{AUTOSCALE}/gripper/example-state.out",
        f"{AUTOSCALE}/gripper/example-static.out",
    ),
    "logistics": DomainSuite(
        "logistics",
        f"{AUTOSCALE}/logistics/domain.pddl",
        f"{AUTOSCALE}/logistics/p01.pddl",
        f"{AUTOSCALE}/logistics/p30.pddl",
        f"{AUTOSCALE}/logistics/example-state.out",
        f"{AUTOSCALE}/logistics/example-static.out",
    ),
    "miconic": DomainSuite(
        "miconic",
        f"{IPC_2023}/miconic/domain.pddl",
        f"{IPC_2023}/miconic/p01.pddl",
        f"{IPC_2023}/miconic/p99.pddl",
        f"{IPC_2023}/miconic/example-state.out",
        f"{IPC_2023}/miconic/example-static.out",
    ),
    "pipesworld-notankage": DomainSuite(
        "pipesworld-notankage",
        f"{AUTOSCALE}/pipesworld-notankage/domain.pddl",
        f"{AUTOSCALE}/pipesworld-notankage/p01.pddl",
        f"{AUTOSCALE}/pipesworld-notankage/p20.pddl",
        f"{AUTOSCALE}/pipesworld-notankage/example-state.out",
        f"{AUTOSCALE}/pipesworld-notankage/example-static.out",
    ),
    "robility": DomainSuite(
        "robility",
        f"{UNSEEN_DOMAINS}/robility/domain.pddl",
        f"{UNSEEN_DOMAINS}/robility/p01.pddl",
        f"{UNSEEN_DOMAINS}/robility/p06.pddl",
        f"{UNSEEN_DOMAINS}/robility/example-state.out",
        f"{UNSEEN_DOMAINS}/robility/example-static.out",
    ),
    "rod-rings": DomainSuite(
        "rod-rings",
        f"{UNSEEN_DOMAINS}/rod-rings/domain.pddl",
        f"{UNSEEN_DOMAINS}/rod-rings/p01.pddl",
        f"{UNSEEN_DOMAINS}/rod-rings/p99.pddl",
        f"{UNSEEN_DOMAINS}/rod-rings/example-state.out",
        f"{UNSEEN_DOMAINS}/rod-rings/example-static.out",
    ),
    "rovers": DomainSuite(
        "rovers",
        f"{IPC_2023}/rovers/domain.pddl",
        f"{IPC_2023}/rovers/p01.pddl",
        f"{IPC_2023}/rovers/p99.pddl",
        f"{IPC_2023}/rovers/example-state.out",
        f"{IPC_2023}/rovers/example-static.out",
    ),
     "satellite": DomainSuite(
        "satellite",
        f"{AUTOSCALE}/satellite/domain.pddl",
        f"{AUTOSCALE}/satellite/p01.pddl",
        f"{AUTOSCALE}/satellite/p30.pddl",
        f"{AUTOSCALE}/satellite/example-state.out",
        f"{AUTOSCALE}/satellite/example-static.out",
    ),
    "sokoban": DomainSuite(
        "sokoban",
        f"{IPC_2023}/sokoban/domain.pddl",
        f"{IPC_2023}/sokoban/p01.pddl",
        f"{IPC_2023}/sokoban/p99.pddl",
        f"{IPC_2023}/sokoban/example-state.out",
        f"{IPC_2023}/sokoban/example-static.out",
    ),
    "spanner": DomainSuite(
        "spanner",
        f"{IPC_2023}/spanner/domain.pddl",
        f"{IPC_2023}/spanner/p01.pddl",
        f"{IPC_2023}/spanner/p99.pddl",
        f"{IPC_2023}/spanner/example-state.out",
        f"{IPC_2023}/spanner/example-static.out",
    ),
    "spanner-modified": DomainSuite(
        "spanner-modified",
        f"{UNSEEN_DOMAINS}/spanner-modified/domain.pddl",
        f"{UNSEEN_DOMAINS}/spanner-modified/p01.pddl",
        f"{UNSEEN_DOMAINS}/spanner-modified/p99.pddl",
        f"{UNSEEN_DOMAINS}/spanner-modified/example-state.out",
        f"{UNSEEN_DOMAINS}/spanner-modified/example-static.out",
    ),
    "storage": DomainSuite(
        "storage",
        f"{AUTOSCALE}/storage/domain.pddl",
        f"{AUTOSCALE}/storage/p01.pddl",
        f"{AUTOSCALE}/storage/p30.pddl",
        f"{AUTOSCALE}/storage/example-state.out",
        f"{AUTOSCALE}/storage/example-static.out",
    ),
    "tpp": DomainSuite(
        "tpp",
        f"{AUTOSCALE}/tpp/domain.pddl",
        f"{AUTOSCALE}/tpp/p01.pddl",
        f"{AUTOSCALE}/tpp/p30.pddl",
        f"{AUTOSCALE}/tpp/example-state.out",
        f"{AUTOSCALE}/tpp/example-static.out",
    ),
    "transport": DomainSuite(
        "transport",
        f"{IPC_2023}/transport/domain.pddl",
        f"{IPC_2023}/transport/p01.pddl",
        f"{IPC_2023}/transport/p99.pddl",
        f"{IPC_2023}/transport/example-state.out",
        f"{IPC_2023}/transport/example-static.out",
    ),
    "visitall": DomainSuite(
        "visitall",
        f"{AUTOSCALE}/visitall/domain.pddl",
        f"{AUTOSCALE}/visitall/p01.pddl",
        f"{AUTOSCALE}/visitall/p30.pddl",
        f"{AUTOSCALE}/visitall/example-state.out",
        f"{AUTOSCALE}/visitall/example-static.out",
    ),
    "zenotravel": DomainSuite(
        "zenotravel",
        f"{AUTOSCALE}/zenotravel/domain.pddl",
        f"{AUTOSCALE}/zenotravel/p01.pddl",
        f"{AUTOSCALE}/zenotravel/p30.pddl",
        f"{AUTOSCALE}/zenotravel/example-state.out",
        f"{AUTOSCALE}/zenotravel/example-static.out",
    )
}
