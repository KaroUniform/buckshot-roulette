from dataclasses import dataclass, field
from typing import Optional

from gameplay.session import EngineSession


@dataclass
class PendingPlayer:
    name: str
    chat_id: int


@dataclass
class RoomSlot:
    players: list[PendingPlayer] = field(default_factory=list)
    session: Optional[EngineSession] = None


class RoomsManager:
    search_lobby: dict
    ROOMS: dict[int, RoomSlot]
    PLAYERS_TO_ROOMS: dict[int, int]

    def __init__(self) -> None:
        self.search_lobby = {}
        self.ROOMS = {}
        self.PLAYERS_TO_ROOMS = {}

    def create_room(self, room_id: int):
        self.ROOMS.setdefault(room_id, RoomSlot())

    def check_room(self, room_id: int):
        return self.ROOMS.get(room_id, None)

    def room_can_accept_player(self, room_id: int) -> bool:
        room = self.ROOMS.get(room_id, None)
        return bool(room) and len(room.players) < 2

    def reg_player_in_room(self, player_name: str, player_id: int, room_id: int):
        self.del_player_from_rooms(player_id)
        room = self.ROOMS.setdefault(room_id, RoomSlot())
        if len(room.players) >= 2:
            raise ValueError(f"Room {room_id} is already full")
        room.players.append(PendingPlayer(name=player_name, chat_id=player_id))
        self.PLAYERS_TO_ROOMS[player_id] = room_id

    def start_room_session(self, room_id: int) -> EngineSession:
        room = self.ROOMS.get(room_id, None)
        if room is None or len(room.players) != 2:
            raise ValueError(f"Room {room_id} is not ready to start")
        first, second = room.players
        room.session = EngineSession.new_pvp(
            first_name=first.name,
            first_chat_id=first.chat_id,
            second_name=second.name,
            second_chat_id=second.chat_id,
        )
        return room.session

    def get_session(self, room_id: int) -> EngineSession:
        room = self.ROOMS.get(room_id, None)
        if room is None or room.session is None:
            raise ValueError(f"Room {room_id} has no active session")
        return room.session

    def get_session_by_player(self, chat_id: int) -> EngineSession:
        room_id = self.get_room_id_by_player(chat_id)
        return self.get_session(room_id)

    def get_room_slot_by_player(self, chat_id: int) -> Optional[RoomSlot]:
        room_id = self.PLAYERS_TO_ROOMS.get(chat_id, None)
        if room_id is None:
            return None
        return self.ROOMS.get(room_id, None)

    def player_is_waiting(self, chat_id: int) -> bool:
        room = self.get_room_slot_by_player(chat_id)
        return room is not None and room.session is None

    def get_room_id_by_player(self, chat_id: int):
        room_id = self.PLAYERS_TO_ROOMS.get(chat_id, None)
        if room_id is None:
            raise Exception("Can't get room by player chat_id. You are not in a room")
        return room_id

    def get_players_chatid(self, chat_id: int):
        room_id = self.PLAYERS_TO_ROOMS.get(chat_id, None)
        if room_id is None:
            raise Exception("You are not in a room")
        room = self.ROOMS.get(room_id, None)
        if room is None or len(room.players) != 2:
            raise Exception("Room is not ready")
        return room.players[0].chat_id, room.players[1].chat_id

    def del_player_from_rooms(self, player_id: int):
        room_id = self.PLAYERS_TO_ROOMS.get(player_id, None)
        if room_id is None:
            return
        room = self.ROOMS.pop(room_id, None)
        if room is not None:
            for player in room.players:
                self.PLAYERS_TO_ROOMS.pop(player.chat_id, None)
        self.PLAYERS_TO_ROOMS.pop(player_id, None)
