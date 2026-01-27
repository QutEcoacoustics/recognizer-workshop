import dataclasses
import re

from typing import Any, Dict, List
from collections import UserDict




@dataclasses.dataclass(frozen=True)
class LabelName:
    """
    Labels work differently between the hoplite database and the BAW workbench
    This class stores the parts of the label in atomic form and provides converters for 
    the hoplite db and for the workbench
    """
    species: str
    call_type: str | None = None

    def __post_init__(self):
        """Normalize call_type after initialization."""
        if self.call_type is not None:
            # remove newlines, tabs, and extra spaces
            normalized = re.sub(r'\s+', ' ', self.call_type)
            # remove any non-word characters (except for spaces and dashes)
            normalized = re.sub(r'[^\w -]', '', normalized)
            normalized = normalized.capitalize().strip()
            
            # Use object.__setattr__ to bypass frozen restriction
            object.__setattr__(self, 'call_type', normalized)


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
class LabelProposition:
    """A data structure to hold a label proposition for a specific label on a specific segment."""
    label: LabelName
    proposition_type: str 
    local_start: float = None
    local_end: float = None
    event_cover: float = 0.0
 # e.g. 'verified_positive', 'verified_negative', 'implied_negative'

    def __post_init__(self):
        # Runtime Validation
        if not PropositionType.is_valid(self.proposition_type):
            raise ValueError(
                f"Invalid proposition_type: '{self.proposition_type}'. "
                f"Must be one of: {PropositionType._ALL}"
            )


class PropositionType:
    # Label proposition types (slightly different from hoplite's pos/neg label types)
    # we use these to decide how to resolve multiple overlapping labels of different types on the same segment
    # in order to arrive at the hoplite label types of 'positive' and 'negative'
    VERIFIED_POS = 'verified_positive'
    VERIFIED_NEG = 'verified_negative'
    IMPLIED_NEG  = 'implied_negative'

    # Optional: A set for fast validation lookup
    _ALL = {VERIFIED_POS, VERIFIED_NEG, IMPLIED_NEG}

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls._ALL


class TargetNegatives(UserDict):
    """
    A strict dictionary that enforces the schema:
    Key: LabelName
    Value: List[LabelName]
    """

    def __setitem__(self, key, value):

        if not isinstance(key, LabelName):
            raise TypeError(
                f"Invalid Key: Expected 'LabelName', got '{type(key).__name__}' ({key})"
            )

        if not isinstance(value, list):
            raise TypeError(
                f"Invalid Value: Expected a 'list', got '{type(value).__name__}'"
            )

        for i, item in enumerate(value):
            if not isinstance(item, LabelName):
                raise TypeError(
                    f"Invalid List Item at index {i}: Expected 'LabelName', "
                    f"got '{type(item).__name__}' ({item})"
                )

        super().__setitem__(key, value)

    def item_tags(self, target: LabelName) -> List[str]:
        """Get the tag representations of the items for the given key."""
        if target not in self.data:
            return []
         
        return [item.get_baw_tags()[0] for item in self.data[target]]
    
    def item_strings(self, target: LabelName) -> List[str]:
        """Get the string representations of the items for the given key."""
        if target not in self.data:
            return []
         
        return [str(item) for item in self.data[target]]

    def target_has_item(self, target: LabelName, negative: LabelName | str) -> bool:
        """Check if the specified item is in the list for the given key."""
        if target not in self.data:
            return False
        if negative in self.data[target]:
            return True
        
        # check tag representation and string representation
        if isinstance(negative, str):
            if negative in self.item_strings(target) or negative in self.item_tags(target):
                return True


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

