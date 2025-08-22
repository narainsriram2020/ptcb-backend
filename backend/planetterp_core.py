import requests
import numpy as np
import re
import os
import json
import datetime
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Fixed: Use GOOGLE_API_KEY (matches your Railway environment variable)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

PLANETTERP_BASE_URL = "https://api.planetterp.com/v1"

def torch_initialization():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def get_courses():
    try:
        response = requests.get(f"{PLANETTERP_BASE_URL}/courses", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching courses: {e}")
    return []

def get_course(course_id):
    try:
        response = requests.get(f"{PLANETTERP_BASE_URL}/course", 
                              params={"name": course_id}, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching course {course_id}: {e}")
    return None

def get_professor(name):
    try:
        response = requests.get(f"{PLANETTERP_BASE_URL}/professor", 
                               params={"name": name, "reviews": "true"}, timeout=10)
        if response.status_code == 200:
            professor_data = response.json()
            # Sort reviews by recency
            return sort_professor_reviews(professor_data)
    except Exception as e:
        print(f"Error fetching professor {name}: {e}")
    return None

def sort_professor_reviews(professor_data):
    if professor_data and 'reviews' in professor_data and professor_data['reviews']:
        # Sort reviews by date in descending order
        professor_data['reviews'].sort(key=lambda x: x.get('created', ''), reverse=True)
    return professor_data

def filter_recent_grades(grades, years=4):
    # Get current year
    current_year = datetime.datetime.now().year
    
    recent_grades = [g for g in grades if g.get('semester') and 
                    int(g.get('semester', '000000')[:4]) >= (current_year - years)]

    recent_grades.sort(key=lambda x: x.get('semester', '000000'), reverse=True)
    
    return recent_grades

def get_course_grades(course_id):
    try:
        response = requests.get(f"{PLANETTERP_BASE_URL}/grades", 
                              params={"course": course_id}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "error" in data:
                return []
            return filter_recent_grades(data)
    except Exception as e:
        print(f"Error fetching grades for {course_id}: {e}")
    return []

def extract_course_ids(query):
    # Look for standard course patterns like CMSC330, MATH140, etc.
    pattern = r'\b[A-Z]{4}\d{3}[A-Z]?\b'
    matches = re.findall(pattern, query.upper())
    return matches

# Model and semantic search functions
def load_model():
    try:
        # Reduce resource usage for deployment
        import torch
        torch.set_num_threads(1)
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def initialize_index(model, courses):
    if not courses or not model:
        return None, []
        
    try:
        # Create descriptions
        texts = []
        ids = []
        for course in courses:
            course_id = course.get('course_id') or course.get('name')
            title = course.get('title', '')
            desc = course.get('description', '')
            texts.append(f"{course_id}: {title}. {desc}")
            ids.append(course_id)
        
        # Create embeddings
        embeddings = model.encode(texts)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype('float32'))
        
        return index, ids
    except Exception as e:
        print(f"Error initializing index: {e}")
        return None, []

def search(query, index, course_ids, model, k=5):
    if index is None or not model:
        return []
    
    try:
        query_embedding = model.encode([query])
        _, indices = index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        results = []
        for idx in indices[0]:
            if idx < len(course_ids):
                course_id = course_ids[idx]
                if course_id not in results:  # Avoid duplicates
                    results.append(course_id)
        
        return results
    except Exception as e:
        print(f"Error in search: {e}")
        return []

def get_rating_description(rating):
    """Convert numerical rating to descriptive text"""
    if not rating or rating < 0:
        return " (No rating available)"
    elif rating >= 4.5:
        return f" (Rating: {rating:.1f}/5.0 - Excellent)"
    elif rating >= 4.0:
        return f" (Rating: {rating:.1f}/5.0 - Very Good)"
    elif rating >= 3.5:
        return f" (Rating: {rating:.1f}/5.0 - Good)"
    elif rating >= 3.0:
        return f" (Rating: {rating:.1f}/5.0 - Average)"
    elif rating >= 2.0:
        return f" (Rating: {rating:.1f}/5.0 - Below Average)"
    else:
        return f" (Rating: {rating:.1f}/5.0 - Poor)"

# Generate response
def generate_response(query, data, chat_model):
    try:
        system = """
        You are a UMD assistant using PlanetTerp data. Be concise but helpful about courses and professors.
        Focus on the most recent grades, ratings, and recommendations from the past 3-4 years (2021-2025).
        Explicitly mention the recency of the data (e.g., "According to Spring 2024 data...").
        Remember context from previous questions. Keep in mind what class or professors you have already suggested during the conversation and use that as context. 
        Do not bring up random data out of nowhere continue conversation on whatever data is being currently discussed. 
        If you don't have information about a specific course or professor, simply say so and explain that
        they should visit the PlanetTerp website to get more info, but this should be the last resort option.
        Sound very laid back and chill like you are another student.
        When a student asks about what professor they should take for a certain course give them your personal evaluation of the best professor.
        When mentioning professors, use the rating_description field which includes the professor name and their numerical rating with quality description.
        When mentioning professor names, format them as <strong>Professor Name</strong> to make them bold.
        """
        
        # Add rating descriptions to professors
        for prof in data["professors"]:
            if "average_rating" in prof:
                prof["rating_description"] = get_rating_description(prof["average_rating"])
            else:
                prof["rating_description"] = " (No rating available)"
        
        # Create a clean context
        clean_context = {
            "courses": data["courses"],
            "professors": [],
            "grades": data["grades"]
        }
        
        # Add professors with proper rating formatting
        for prof in data["professors"]:
            clean_prof = prof.copy()
            if "average_rating" in prof:
                rating_desc = get_rating_description(prof["average_rating"])
                clean_prof["rating_description"] = f"{prof['name']}{rating_desc}"
            else:
                clean_prof["rating_description"] = f"{prof['name']} (No rating available)"
            clean_context["professors"].append(clean_prof)
        
        prompt = f"""
        PlanetTerp Data: {json.dumps(clean_context, indent=2, ensure_ascii=False)}
        Question: {query}
        """
        
        response = chat_model.send_message(system + prompt)
        return response.text
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error: {str(e)}. Please try asking your question again."

def generate_chat_name(query):
    try:
        course_ids = extract_course_ids(query)
        if course_ids:
            return f"{', '.join(course_ids)} Question"
        
        words = query.split()
        if len(words) <= 4:
            return query
        else:
            return ' '.join(words[:4]) + "..."
    except Exception as e:
        print(f"Error generating chat name: {e}")
        return "New Chat"

def get_greeting():
    try:
        hour = datetime.datetime.now().hour
        if hour > 5 and hour < 11:
            return "Good morning"
        elif hour >= 11 and hour < 17:
            return "Good afternoon"
        else:
            return "Good evening"
    except Exception as e:
        print(f"Error getting greeting: {e}")
        return "Hello"