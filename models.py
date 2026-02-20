from pydantic import BaseModel

class SongRequest(BaseModel):
    lyrics: str

class SongResponse(BaseModel):
    success: bool
    data: dict
    message: str