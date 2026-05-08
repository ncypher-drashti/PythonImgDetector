from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Person(BaseModel):
    name: str
    age: int

@app.post("/greet")
def greet(person: Person):
    return {
        "message": f"Hello {person.name}!",
        "next_year": person.age + 1
    }