from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import sys

# Import planetterp_core (now in same directory)
from planetterp_core import (
    load_model, get_courses, get_course, get_professor, get_course_grades,
    extract_course_ids, initialize_index, search, generate_response,
    generate_chat_name, get_greeting
)
import google.generativeai as genai
from dotenv import load_dotenv
import uuid
import datetime
import random

load_dotenv()

app = FastAPI(title="PlanetTerp Chatbot API", version="1.0.0")

# CORS middleware - Updated for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://*.vercel.app",  # Allow all Vercel apps
        "https://your-app-name.vercel.app"  # Replace with your actual Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rest of your code stays the same...
# (keeping the existing code from your main.py)

# Global variables
model = None
index = None
course_ids = []
chat_model = None

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    response: str
    chat_name: Optional[str] = None

class FunFactResponse(BaseModel):
    fact: str

def get_random_umd_fact():
    umd_facts = [
        "UMD's mascot Testudo is a diamondback terrapin, Maryland's state reptile.",
        "McKeldin Mall is the largest academic mall in the country.",
        "UMD has the oldest continuously operating airport in the world - College Park Airport.",
        "UMD's school colors (red, white, black, and gold) come from the Maryland state flag.",
        "UMD's campus spans over 1,300 acres.",
        "The 'M Circle' flowerbed is 57 feet in diameter.",
        "UMD is one of only 62 members of the Association of American Universities.",
        "The Xfinity Center can hold over 17,000 fans for basketball games.",
        "UMD's campus has over 8,000 trees of 400+ species.",
        "Morrill Hall is UMD's oldest academic building, completed in 1898.",
        "The Clarice Smith Performing Arts Center covers 318,000 square feet.",
        "The fear of turtles is called chelonaphobia.",
        "Testudo statues around campus are considered good luck, especially during finals week.",
    ]
    return random.choice(umd_facts)

def initialize_chat_model():
    global chat_model
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        chat_model = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
        ).start_chat(history=[])
        return chat_model
    except Exception as e:
        print(f"Error initializing chat model: {e}")
        return None

def ensure_index_initialized():
    global model, index, course_ids
    if model is None:
        model = load_model()
    if index is None:
        courses = get_courses()
        if courses:
            index, course_ids = initialize_index(model, courses)

@app.on_event("startup")
async def startup_event():
    global model, index, course_ids, chat_model
    print("Initializing PlanetTerp Chatbot...")
    
    # Initialize the model
    try:
        model = load_model()
        if model:
            print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    # Initialize the index
    try:
        courses = get_courses()
        if courses:
            index, course_ids = initialize_index(model, courses)
            print(f"Index initialized with {len(course_ids)} courses")
    except Exception as e:
        print(f"Error initializing index: {e}")
    
    # Initialize chat model
    try:
        initialize_chat_model()
        if chat_model:
            print("Chat model initialized successfully")
    except Exception as e:
        print(f"Error initializing chat model: {e}")

@app.get("/")
async def root():
    return {"message": "PlanetTerp Chatbot API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None, 
        "index_loaded": index is not None,
        "chat_model_loaded": chat_model is not None
    }

@app.get("/fun-fact", response_model=FunFactResponse)
async def get_fun_fact():
    return FunFactResponse(fact=get_random_umd_fact())

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global chat_model
    
    try:
        # Ensure index is initialized
        ensure_index_initialized()
        
        if not chat_model:
            chat_model = initialize_chat_model()
        
        if not chat_model:
            raise HTTPException(status_code=500, detail="Chat model not available")
        
        # Create chat history for the model
        chat_history = []
        for msg in request.chat_history:
            role = "user" if msg.role == "user" else "model"
            chat_history.append({"role": role, "parts": [msg.content]})
        
        # Create a new chat model with history
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        genai.configure(api_key=api_key)
        current_chat = genai.GenerativeModel(
            'gemini-2.0-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
        ).start_chat(history=chat_history)
        
        # Extract course IDs from the query
        direct_course_ids = extract_course_ids(request.message)
        
        # If no direct matches, use semantic search
        semantic_course_ids = [] if direct_course_ids else search(
            request.message, 
            index, 
            course_ids, 
            model
        ) if index and course_ids else []
        
        # Combine results
        all_course_ids = direct_course_ids + semantic_course_ids
        data = {"courses": [], "professors": [], "grades": []}

        # Get course details
        for course_id in all_course_ids[:3]:  # Limit to top 3 results
            try:
                course = get_course(course_id)
                if course:
                    data["courses"].append(course)
                    
                    # Get grades for this course
                    try:
                        grades = get_course_grades(course_id)
                        if grades:
                            data["grades"].extend(grades[:5])  # Limit to 5 most recent grade entries
                    except Exception as e:
                        print(f"Error getting grades for {course_id}: {e}")
                    
                    # Get professors for this course
                    for prof_name in course.get("professors", [])[:5]:  # Limit to 5 professors
                        try:
                            prof = get_professor(prof_name)
                            if prof:
                                data["professors"].append(prof)
                        except Exception as e:
                            print(f"Error getting professor {prof_name}: {e}")
            except Exception as e:
                print(f"Error processing course {course_id}: {e}")
        
        # Generate response
        response = generate_response(request.message, data, current_chat)
        
        # Generate chat name if this is the first message
        chat_name = None
        if not request.chat_history:
            try:
                chat_name = generate_chat_name(request.message)
            except Exception as e:
                print(f"Error generating chat name: {e}")
                chat_name = "New Chat"
        
        return ChatResponse(response=response, chat_name=chat_name)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/greeting")
async def get_greeting_endpoint():
    try:
        return {"greeting": get_greeting()}
    except Exception as e:
        print(f"Error getting greeting: {e}")
        return {"greeting": "Welcome to PlanetTerp Chatbot!"}

# Health check endpoint with more details
@app.get("/debug")
async def debug_info():
    return {
        "environment_variables": {
            "GOOGLE_API_KEY": "set" if os.getenv('GOOGLE_API_KEY') else "not set",
            "PORT": os.getenv('PORT', 'not set')
        },
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "files_in_directory": os.listdir('.') if os.path.exists('.') else []
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)