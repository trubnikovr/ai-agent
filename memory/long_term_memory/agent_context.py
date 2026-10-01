from dataclasses import dataclass
from typing_extensions import TypedDict


@dataclass
class AgentContext:
    user_id: str


class UserInfo(TypedDict):
    name: str
    profession: str
