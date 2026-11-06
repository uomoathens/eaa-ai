from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    metric: str
    threshold: float
    direction: str

    def margin(self, value):
        if self.direction == "max":
            return self.threshold - value
        if self.direction == "min":
            return value - self.threshold
        raise ValueError(f"unknown direction {self.direction}")


class CompliancePolicy:
    def __init__(self, requirements):
        self.requirements = [
            requirement if isinstance(requirement, Requirement) else Requirement(**requirement)
            for requirement in requirements
        ]

    @classmethod
    def from_mapping(cls, mapping):
        return cls(
            [Requirement(metric=name, threshold=float(spec["threshold"]), direction=spec["direction"]) for name, spec in mapping.items()]
        )

    def margins(self, metrics):
        return {requirement.metric: requirement.margin(metrics[requirement.metric]) for requirement in self.requirements}

    def binary_constraint(self, metrics):
        return int(all(value >= 0.0 for value in self.margins(metrics).values()))

    def hinge_penalty(self, metrics):
        return float(sum(max(0.0, -value) for value in self.margins(metrics).values()))

    def deployable(self, metrics):
        return min(self.margins(metrics).values()) >= 0.0

    def violations(self, metrics):
        return sorted(name for name, value in self.margins(metrics).items() if value < 0.0)

    def evaluate(self, metrics):
        return {
            "margins": self.margins(metrics),
            "c_ethics": self.binary_constraint(metrics),
            "hinge_penalty": self.hinge_penalty(metrics),
            "deployable": self.deployable(metrics),
            "violations": self.violations(metrics),
        }
