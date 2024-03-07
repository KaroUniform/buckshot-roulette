from re import L
from .models.player_model import PlayerModel
from .scene import Scene


class RoomsManager:
    ROOMS: dict[int, Scene] = {}
    PLAYERS_TO_ROOMS: dict[int, int] = {}

    def __init__(self) -> None:
        pass

    def create_room(self, room_id: int):

        if self.ROOMS.get(room_id, None):
            return

        self.ROOMS[room_id] = Scene()

    def get_room(self, room_id: int) -> Scene:
        room = self.ROOMS.get(room_id, None)
        if not room:
            raise Exception(f"No room with this room_id: {room_id}")
        return room

    def check_room(self, room_id: int):
        room = self.ROOMS.get(room_id, None)
        return room

    def reg_player_in_room(self, player_name: str, player_id: int, room_id: int):

        self.del_player_from_rooms(player_id)

        room = self.get_room(room_id)

        if room.first_player is None:
            player_seat = 0
        else:
            player_seat = 1

        room.add_player(
            PlayerModel(name=player_name, chat_id=player_id, hp=1, max_hp=1),
            player_seat=player_seat,
        )
        self.PLAYERS_TO_ROOMS[player_id] = room_id

    def del_player_from_rooms(self, player_id: int):

        room = self.PLAYERS_TO_ROOMS.get(player_id, None)
        if room:
            self.ROOMS.pop(room, None)
        self.PLAYERS_TO_ROOMS.pop(player_id, None)

    def get_room_id_by_player(self, chat_id: int):
        room_id = self.PLAYERS_TO_ROOMS.get(chat_id, None)
        if not room_id:
            raise Exception("Can't get room by player chat_id. You are not in a room")
        return room_id

    def get_players_chatid(self, chat_id: int):
        room_id = self.PLAYERS_TO_ROOMS.get(chat_id, None)
        if room_id is None:
            raise Exception("You are not in a room")
        room = self.get_room(room_id)
        first_player = room.first_player.data.chat_id
        second_player = room.second_player.data.chat_id
        # if first_player is None or second_player is None:
        #     raise Exception(f"Not enought players: {first_player, second_player}")
        return first_player, second_player
