"""Custom exception classes for consistent error handling."""

from typing import ClassVar


class APIError(Exception):
    """Base API error class."""

    status_code: ClassVar[int] = 500
    error_code: ClassVar[str] = "INTERNAL_ERROR"

    def __init__(self, message: str | None = None):
        self.message = message or "An internal error occurred"
        super().__init__(self.message)


class MissingAPIKeyError(APIError):
    """Raised when API key is not provided."""

    status_code: ClassVar[int] = 401
    error_code: ClassVar[str] = "MISSING_API_KEY"

    def __init__(self, message: str = "API key is required"):
        super().__init__(message)


class InvalidAPIKeyError(APIError):
    """Raised when API key is invalid."""

    status_code: ClassVar[int] = 401
    error_code: ClassVar[str] = "INVALID_API_KEY"

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message)


class ValidationError(APIError):
    """Raised for request validation errors."""

    status_code: ClassVar[int] = 400
    error_code: ClassVar[str] = "VALIDATION_ERROR"

    def __init__(self, message: str = "Validation error"):
        super().__init__(message)


class FileTooLargeError(APIError):
    """Raised when uploaded file exceeds size limit."""

    status_code: ClassVar[int] = 413
    error_code: ClassVar[str] = "FILE_TOO_LARGE"

    def __init__(self, message: str = "File exceeds maximum size limit"):
        super().__init__(message)


class InvalidFileTypeError(APIError):
    """Raised when file type is not allowed."""

    status_code: ClassVar[int] = 415
    error_code: ClassVar[str] = "INVALID_FILE_TYPE"

    def __init__(self, message: str = "File type is not supported"):
        super().__init__(message)


class UploadError(APIError):
    """Raised when upload to presigned URL fails."""

    status_code: ClassVar[int] = 500
    error_code: ClassVar[str] = "UPLOAD_FAILED"

    def __init__(self, message: str = "Failed to upload file to storage"):
        super().__init__(message)


class GenerationError(APIError):
    """Raised when generation fails."""

    status_code: ClassVar[int] = 500
    error_code: ClassVar[str] = "GENERATION_FAILED"

    def __init__(self, message: str = "Generation failed"):
        super().__init__(message)


class GPUOutOfMemoryError(APIError):
    """Raised when GPU runs out of memory during generation."""

    status_code: ClassVar[int] = 503
    error_code: ClassVar[str] = "GPU_OUT_OF_MEMORY"

    def __init__(self, message: str = "GPU out of memory. Try reducing resolution, duration, or wait for current jobs to complete."):
        super().__init__(message)


class JobTimeoutError(APIError):
    """Raised when a job exceeds its timeout."""

    status_code: ClassVar[int] = 504
    error_code: ClassVar[str] = "JOB_TIMEOUT"

    def __init__(self, message: str = "Job timed out"):
        super().__init__(message)
