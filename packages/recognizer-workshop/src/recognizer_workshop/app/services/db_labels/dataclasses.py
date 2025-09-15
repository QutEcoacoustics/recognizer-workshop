import dataclasses
import re

from typing import Any, Dict, List

@dataclasses.dataclass
class LabelName:
    """
    Labels work differently between the hoplite database and the BAW workbench
    This class stores the parts of the label in atomic form and provides converters for 
    the hoplite db and for the workbench
    
    
    """
    species: str
    call_type: str | None

    # init method
    def __init__(self, species: str, call_type: str | None = None):
        self.species = species

        # Normalize:
        # Capitalize first letter and all others lowercase
        # convert any whitespace to a single space, and trim whitespace from start and end

        if call_type is not None:
            # remove newlines, tabs, and extra spaces
            call_type = re.sub(r'\s+', ' ', call_type)
            # remove any non-word characters (except for spaces and dashes)
            call_type = re.sub(r'[^\w -]', '', call_type)
            call_type = call_type.capitalize().strip()

        self.call_type = call_type

    def get_baw_tags(self):
        return [self.species + " " + self.call_type, self.species] if self.call_type else [self.species]
    
    def get_hoplite_label_text(self):
        return f"{self.species} - {self.call_type}" if self.call_type else self.species
    
    @classmethod
    def from_hoplite_label_text(cls, label_text: str):

        parts = label_text.split(" - ")
        species = parts[0].strip()
        call_type = parts[1].strip() if len(parts) > 1 else None

        return cls(species=species, call_type=call_type)
    
    @classmethod
    def from_dict(cls, label_dict: Dict[str, Any]):

        species = label_dict["species"]
        call_type = label_dict.get("call_type", None)

        return cls(species=species, call_type=call_type)
    
    def __str__(self) -> str:
        return self.get_hoplite_label_text()
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "species": self.species,
            "call_type": self.call_type
        }
        


@dataclasses.dataclass
class LabelCounts:
    """A simple data structure to hold positive and negative counts for a label."""
    label_name: str
    positive: int = 0
    negative: int = 0

    def __str__(self) -> str:
        return f"{self.label_name}: {self.positive} positive, {self.negative} negative"


@dataclasses.dataclass
class DatabaseSummary:
    total_embeddings: int
    datasets: List[str]
    label_counts: Dict[str, LabelCounts]

    def __str__(self) -> str:
        lines = [
            "DB Summary:",
            f"  Number of embeddings: {self.total_embeddings}",
            f"  {len(self.datasets)} Dataset(s): {', '.join(self.datasets)}",
            ""
        ]
        
        if self.label_counts:
            lines.append("Label counts:")
            for label_name in sorted(self.label_counts.keys()):
                lines.append(f"  {str(self.label_counts[label_name])}")
        else:
            lines.append("No labeled embeddings in db.")
            
        return "\n".join(lines)

