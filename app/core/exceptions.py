from fastapi import HTTPException


class VectorStoreEmptyError(HTTPException):
    def __init__(self):
        super().__init__(status_code=400, detail="Vector store vazio. Faça /ingest primeiro.")
