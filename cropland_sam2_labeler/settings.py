from dataclasses import dataclass


@dataclass
class Sam2Settings:
    backend: str = "mock"
    service_url: str = ""
    local_command: str = ""
    weights_path: str = ""
    device: str = "cpu"
    simplify_tolerance: float = 0.5
    min_area: float = 1.0
    max_hole_area: float = 0.0

    def validate(self):
        errors = []
        if self.backend not in {"mock", "http", "local"}:
            errors.append("Backend must be mock, http, or local.")
        if self.backend == "http" and not self.service_url:
            errors.append("HTTP backend requires a service URL.")
        if self.backend == "local" and not self.local_command:
            errors.append("Local backend requires a command.")
        if self.simplify_tolerance < 0:
            errors.append("Simplify tolerance cannot be negative.")
        if self.min_area < 0:
            errors.append("Minimum area cannot be negative.")
        if self.max_hole_area < 0:
            errors.append("Maximum hole area cannot be negative.")
        return errors
