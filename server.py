import os
import sys
import logging
import asyncio
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from app.interviewer import InterviewBot
from app.analyzer import analyze_results, summary_results
from app.utils import get_cv, get_job_description
from app.prompts import evaluate_code
from app.speech_to_text import transcribe_audio_with_overlap

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# FastAPI app setup
app = FastAPI()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# State management
class InterviewState:
    def __init__(self):
        self.cv_text = ""
        self.job_description = ""
        self.interview_bot = None
        self.results = {"INTRODUCTION": {}, "PROJECT": {}, "CODING": {}, "TECHNICAL": {}, "OUTRO": {}}
        self.stop_interview = asyncio.Event()

# Pydantic model for event validation
class EventData(BaseModel):
    type: str
    code: str = None
    ques: str = None
    audio_data: str = None

# Root endpoint
@app.get("/")
async def root():
    return {"data": "HELLO WORLD"}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = InterviewState()

    async def handle_interview():
        if state.interview_bot:
            try:
                await state.interview_bot.conduct_interview(websocket)
            except asyncio.CancelledError:
                logging.info("Interview task cancelled")
            except Exception as e:
                logging.error(f"Error during interview: {e}")
            finally:
                state.interview_bot = None

    try:
        while True:
            try:
                data = await websocket.receive_json()
                logging.info(f"Received data: {data}")
                event = EventData(**data)  # Validate incoming data
                await handle_event(event, websocket, state, handle_interview)
            except ValidationError as e:
                logging.error(f"Validation error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        if state.interview_bot:
            state.interview_bot.stop_interview.set()
            state.interview_bot = None

# Event handler
async def handle_event(event: EventData, websocket: WebSocket, state: InterviewState, handle_interview):
    if event.type == 'upload_cv':
        await handle_upload_cv(websocket, state)
    elif event.type == 'analyze_jd':
        await handle_analyze_jd(websocket, state)
    elif event.type == 'start_interview':
        await handle_start_interview(websocket, state, handle_interview)
    elif event.type == 'answer':
        await handle_answer(event, state)
    elif event.type == 'coding':
        await handle_coding(event, websocket, state)
    elif event.type == 'end_interview':
        await handle_end_interview(websocket, state)
    elif event.type == 'get_analysis':
        await handle_get_analysis(websocket, state)
    elif event.type == 'get_summary_analysis':
        await handle_summary_analysis(websocket, state)
    elif event.type == 'test_coding_question':
        await handle_test_coding_question(websocket)
    elif event.type == 'audio':
        await handle_audio_transcription(event, websocket, state)

# Event-specific handlers
async def handle_upload_cv(websocket, state):
    state.cv_text = await get_cv(await websocket.receive_json(), websocket)
    await websocket.send_json({'type': 'cv_uploaded', 'message': 'CV data received', 'cv_text': state.cv_text})

async def handle_analyze_jd(websocket, state):
    state.job_description = await get_job_description(await websocket.receive_json(), websocket)
    await websocket.send_json({'type': 'jd_analyzed', 'message': 'Received JD Successfully', 'job_description': state.job_description})

async def handle_start_interview(websocket, state, handle_interview):
    if not state.interview_bot:
        state.interview_bot = InterviewBot(state.cv_text, state.job_description, state.results)
        asyncio.create_task(handle_interview())
        await websocket.send_json({"type": "interview_started", "message": "Interview started"})

async def handle_answer(event, state):
    if state.interview_bot:
        state.interview_bot.current_answer = event.code  # Assuming answer is passed in `code`
        state.interview_bot.answer_event.set()

async def handle_coding(event, websocket, state):
    try:
        if state.interview_bot:
            response = evaluate_code(state.interview_bot.llm, event.ques, event.code)
            await websocket.send_json({"type": "code_evaluation", "result": response})
            if response and response.get("RESULT"):
                state.interview_bot.coding_event.set()
    except Exception as e:
        logging.error(f"Error in coding handler: {e}")
        await websocket.send_json({"type": "coding_error", "message": str(e)})

async def handle_end_interview(websocket, state):
    if state.interview_bot:
        state.interview_bot.stop_interview.set()
        await asyncio.sleep(0.1)
        state.interview_bot.stop_interview.clear()
        await websocket.send_json({"type": "interview_end", "message": "Interview ended"})
    logging.info("Interview concluded")

async def handle_get_analysis(websocket, state):
    result = analyze_results(state.results)
    await websocket.send_json({"type": "analysis", "result": result})

async def handle_summary_analysis(websocket, state):
    summary = summary_results(state.results)
    await websocket.send_json({"type": "summary_analysis", "result": summary})

async def handle_test_coding_question(websocket):
    question = random.choice([
        "Q1. Print Hello World",
        "Q2. Print Hello Anish",
        "Q3. Print Hello Duniya"
    ])
    await websocket.send_json({"type": "test_coding_question", "message": question})

async def handle_audio_transcription(event, websocket, state):
    try:
        if not event.audio_data:
            await websocket.send_json({"type": "error", "message": "No audio data provided"})
            return

        # Transcribe audio using overlapping chunks
        transcription = transcribe_audio_with_overlap(event.audio_data)
        await websocket.send_json({"type": "transcription_complete", "transcription": transcription})

        # Add transcription to the current answer
        if transcription and state.interview_bot:
            state.interview_bot.current_answer = transcription
            state.interview_bot.answer_event.set()
    except Exception as e:
        logging.error(f"Error during audio transcription: {e}")
        await websocket.send_json({"type": "transcription_error", "message": str(e)})

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(content={"error": "Internal server error"}, status_code=500)

# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
