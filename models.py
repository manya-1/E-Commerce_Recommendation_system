import json
from datetime import datetime

class Product:
    """Class representing a product in the e-commerce system"""
    
    def __init__(self, id=None, name=None, category=None, price=None, 
                 description=None, attributes=None, created_at=None):
        self.id = id
        self.name = name
        self.category = category
        self.price = price
        self.description = description
        self.attributes = self._parse_attributes(attributes)
        self.created_at = created_at or datetime.now()
    
    def _parse_attributes(self, attributes):
        """Parse attributes from string to dictionary"""
        if isinstance(attributes, str):
            try:
                return json.loads(attributes.replace("'", "\""))
            except:
                # If attributes string is not valid JSON, try to evaluate it as a Python dict
                try:
                    return eval(attributes)
                except:
                    return {}
        elif isinstance(attributes, dict):
            return attributes
        else:
            return {}
    
    def to_dict(self):
        """Convert product to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "description": self.description,
            "attributes": self.attributes,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create product from dictionary"""
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            category=data.get("category"),
            price=data.get("price"),
            description=data.get("description"),
            attributes=data.get("attributes"),
            created_at=data.get("created_at")
        )
    
    @classmethod
    def from_db_row(cls, row):
        """Create product from database row"""
        return cls(
            id=row[0],
            name=row[1],
            category=row[2],
            price=row[3],
            description=row[4],
            attributes=row[5],
            created_at=row[6]
        )


class Customer:
    """Class representing a customer in the e-commerce system"""
    
    def __init__(self, id=None, name=None, email=None, age=None, gender=None,
                 location=None, segment=None, preferences=None, created_at=None):
        self.id = id
        self.name = name
        self.email = email
        self.age = age
        self.gender = gender
        self.location = location
        self.segment = segment
        self.preferences = self._parse_preferences(preferences)
        self.created_at = created_at or datetime.now()
    
    def _parse_preferences(self, preferences):
        """Parse preferences from string to dictionary"""
        if isinstance(preferences, str):
            try:
                return json.loads(preferences.replace("'", "\""))
            except:
                # If preferences string is not valid JSON, try to evaluate it as a Python dict
                try:
                    return eval(preferences)
                except:
                    return {}
        elif isinstance(preferences, dict):
            return preferences
        else:
            return {}
    
    def to_dict(self):
        """Convert customer to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "gender": self.gender,
            "location": self.location,
            "segment": self.segment,
            "preferences": self.preferences,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create customer from dictionary"""
        return cls(
            id=data.get("id"),
            name=data.get("name"),
            email=data.get("email"),
            age=data.get("age"),
            gender=data.get("gender"),
            location=data.get("location"),
            segment=data.get("segment"),
            preferences=data.get("preferences"),
            created_at=data.get("created_at")
        )
    
    @classmethod
    def from_db_row(cls, row):
        """Create customer from database row"""
        return cls(
            id=row[0],
            name=row[1],
            email=row[2],
            age=row[3],
            gender=row[4],
            location=row[5],
            segment=row[6],
            preferences=row[7],
            created_at=row[8]
        )


class Interaction:
    """Class representing a customer-product interaction"""
    
    def __init__(self, id=None, customer_id=None, product_id=None, 
                 interaction_type=None, timestamp=None, details=None):
        self.id = id
        self.customer_id = customer_id
        self.product_id = product_id
        self.interaction_type = interaction_type
        self.timestamp = timestamp or datetime.now()
        self.details = self._parse_details(details)
    
    def _parse_details(self, details):
        """Parse details from string to dictionary"""
        if isinstance(details, str):
            try:
                return json.loads(details.replace("'", "\""))
            except:
                # If details string is not valid JSON, try to evaluate it as a Python dict
                try:
                    return eval(details)
                except:
                    return {}
        elif isinstance(details, dict):
            return details
        else:
            return {}
    
    def to_dict(self):
        """Convert interaction to dictionary"""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "product_id": self.product_id,
            "interaction_type": self.interaction_type,
            "timestamp": self.timestamp,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create interaction from dictionary"""
        return cls(
            id=data.get("id"),
            customer_id=data.get("customer_id"),
            product_id=data.get("product_id"),
            interaction_type=data.get("interaction_type"),
            timestamp=data.get("timestamp"),
            details=data.get("details")
        )
    
    @classmethod
    def from_db_row(cls, row):
        """Create interaction from database row"""
        return cls(
            id=row[0],
            customer_id=row[1],
            product_id=row[2],
            interaction_type=row[3],
            timestamp=row[4],
            details=row[5]
        )


class Recommendation:
    """Class representing a product recommendation for a customer"""
    
    def __init__(self, id=None, customer_id=None, product_id=None, 
                 score=None, algorithm=None, timestamp=None):
        self.id = id
        self.customer_id = customer_id
        self.product_id = product_id
        self.score = score
        self.algorithm = algorithm
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self):
        """Convert recommendation to dictionary"""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "product_id": self.product_id,
            "score": self.score,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create recommendation from dictionary"""
        return cls(
            id=data.get("id"),
            customer_id=data.get("customer_id"),
            product_id=data.get("product_id"),
            score=data.get("score"),
            algorithm=data.get("algorithm"),
            timestamp=data.get("timestamp")
        )
    
    @classmethod
    def from_db_row(cls, row):
        """Create recommendation from database row"""
        return cls(
            id=row[0],
            customer_id=row[1],
            product_id=row[2],
            score=row[3],
            algorithm=row[4],
            timestamp=row[5]
        )
