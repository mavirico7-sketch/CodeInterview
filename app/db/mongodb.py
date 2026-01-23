"""
MongoDB connection and session repository.
"""

from datetime import datetime
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import ConnectionFailure

from app.config import get_settings
from app.models.session import InterviewSession, SessionMessage


class MongoDB:
    """MongoDB connection manager and session repository."""
    
    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._collection: Optional[AsyncIOMotorCollection] = None
    
    async def connect(self) -> None:
        """Establish connection to MongoDB."""
        settings = get_settings()
        
        self._client = AsyncIOMotorClient(
            settings.mongodb.connection_string,
            serverSelectionTimeoutMS=5000
        )
        
        # Verify connection
        try:
            await self._client.admin.command('ping')
        except ConnectionFailure as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")
        
        self._db = self._client[settings.mongodb.database]
        self._collection = self._db[settings.mongodb.collection]
        
        # Create indexes
        await self._collection.create_index("created_at")
        await self._collection.create_index("is_active")
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Get the sessions collection."""
        if self._collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._collection
    
    async def create_session(self, session: InterviewSession) -> str:
        """Create a new interview session and return its ID."""
        session_dict = session.model_dump(exclude={"live_coding": {"available_environments"}})
        
        # Convert datetime objects for MongoDB
        session_dict['created_at'] = session.created_at
        session_dict['updated_at'] = session.updated_at
        
        # Convert message timestamps
        for msg in session_dict.get('messages', []):
            if isinstance(msg.get('timestamp'), datetime):
                pass  # Already datetime
        
        result = await self.collection.insert_one(session_dict)
        return str(result.inserted_id)
    
    async def get_session(self, session_id: str) -> Optional[InterviewSession]:
        """Retrieve a session by ID."""
        try:
            object_id = ObjectId(session_id)
        except Exception:
            return None
        
        doc = await self.collection.find_one({"_id": object_id})
        
        if doc is None:
            return None
        
        # Remove MongoDB _id before creating Pydantic model
        doc.pop('_id', None)
        
        return InterviewSession(**doc)
    
    async def update_session(self, session_id: str, session: InterviewSession) -> bool:
        """Update an existing session."""
        try:
            object_id = ObjectId(session_id)
        except Exception:
            return False
        
        session_dict = session.model_dump(exclude={"live_coding": {"available_environments"}})
        session_dict['updated_at'] = datetime.utcnow()
        
        result = await self.collection.replace_one(
            {"_id": object_id},
            session_dict
        )
        
        return result.modified_count > 0
    
    async def add_message(self, session_id: str, message: SessionMessage) -> bool:
        """Add a message to a session."""
        try:
            object_id = ObjectId(session_id)
        except Exception:
            return False
        
        message_dict = message.model_dump()
        
        result = await self.collection.update_one(
            {"_id": object_id},
            {
                "$push": {"messages": message_dict},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return result.modified_count > 0
    
    async def update_session_field(self, session_id: str, field: str, value) -> bool:
        """Update a specific field in a session."""
        try:
            object_id = ObjectId(session_id)
        except Exception:
            return False
        
        result = await self.collection.update_one(
            {"_id": object_id},
            {
                "$set": {
                    field: value,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return result.modified_count > 0
    
    async def increment_tokens(self, session_id: str, tokens: int) -> bool:
        """Increment the total token count for a session."""
        try:
            object_id = ObjectId(session_id)
        except Exception:
            return False
        
        result = await self.collection.update_one(
            {"_id": object_id},
            {
                "$inc": {"total_tokens_used": tokens},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return result.modified_count > 0


# Global database instance
_db_instance: Optional[MongoDB] = None


async def get_db() -> MongoDB:
    """Get the global database instance."""
    global _db_instance
    
    if _db_instance is None:
        _db_instance = MongoDB()
        await _db_instance.connect()
    
    return _db_instance


async def close_db() -> None:
    """Close the global database connection."""
    global _db_instance
    
    if _db_instance is not None:
        await _db_instance.disconnect()
        _db_instance = None

