import json
from enum import Enum


class ErrorCode(Enum):
    UNKNOWN = 10000
    INDEX_EXIST = 10001
    INDEX_NOT_FOUND = 10002
    INDEX_EMBEDDING_ERROR = 10003
    INDEX_UPLOAD_ERROR = 10004
    INDEX_DELETE_ERROR = 10005
    INDEX_UPDATE_ERROR = 10006
    INDEX_DIMENSION_NOT_MATCH = 10007


class SRAGException(Exception):
    def __init__(self, code: int = 0, msg: str = "", context: dict = {}):
        self.code = code
        self.msg = msg
        self.context = context

    def __str__(self):
        return json.dumps({"code": self.code, "msg": self.msg})

    __repr__ = __str__


class IndexExistException(SRAGException):
    def __init__(self, msg: str = "", context: dict = {}):
        super().__init__(ErrorCode.INDEX_EXIST.value, msg, context)


class IndexNotFoundException(SRAGException):
    def __init__(self, msg: str = "", context: dict = {}):
        super().__init__(ErrorCode.INDEX_NOT_FOUND.value, msg, context)


class IndexEmbeddingException(SRAGException):
    def __init__(self, msg: str = "", context: dict = {}):
        super().__init__(ErrorCode.INDEX_EMBEDDING_ERROR.value, msg, context)


class IndexUploadException(SRAGException):
    def __init__(self, msg: str = "", context: dict = {}):
        super().__init__(ErrorCode.INDEX_UPLOAD_ERROR.value, msg, context)


class IndexDeleteException(SRAGException):
    def __init__(self, msg: str = "", context: dict = {}):
        super().__init__(ErrorCode.INDEX_DELETE_ERROR.value, msg, context)


class IndexUpdateException(SRAGException):
    def __init__(self, msg: str = "", context: dict = {}):
        super().__init__(ErrorCode.INDEX_UPDATE_ERROR.value, msg, context)


class IndexDimensionNotMatchException(SRAGException):
    def __init__(self, msg: str = "", context: dict = {}):
        super().__init__(ErrorCode.INDEX_DIMENSION_NOT_MATCH.value, msg, context)


class UnknownException(SRAGException):
    def __init__(self, msg: str = "", context: dict = {}):
        super().__init__(ErrorCode.UNKNOWN.value, msg, context)
