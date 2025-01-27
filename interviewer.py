import os
import re
import sys
from dotenv import load_dotenv
from app.utils import *
from app.prompts import *
from app.analyzer import *
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import asyncio
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import random
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class InterviewBot:
    def __init__(self, cv_text, job_description, results):
        self.stages = ["INTRODUCTION", "PROJECT", "CODING", "TECHNICAL", "OUTRO"]
        self.message_history = ChatMessageHistory()
        self.result = results
        self.answer_event = asyncio.Event()
        self.current_answer = None
        self.stop_interview = asyncio.Event()
        self.coding_event = asyncio.Event()

        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        except Exception as e:
            raise Exception(f"Failed to initialize Google Generative AI: {str(e)}")

        try:
            self.cv_parts = get_resume_in_parts(self.model, cv_text)
            self.job_description = get_summarized_jd(self.model, job_description)
        except Exception as e:
            raise Exception(f"Failed to process CV or job description: {str(e)}")

    @staticmethod
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        return re.sub(r"[^a-z0-9',!?-]", " ", text)

    def get_ai_response(self, prompt, input_text):
        try:
            chain = ChatPromptTemplate.from_messages([
                ("human", prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]) | self.model

            chain_with_history = RunnableWithMessageHistory(
                chain,
                lambda session_id: self.message_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            response = chain_with_history.invoke(
                {"input": input_text},
                {"configurable": {"session_id": "interview_session"}}
            )

            return self.clean_text(response.content)
        except Exception as e:
            raise Exception(f"Failed to get AI response: {str(e)}")

    async def start_coding_stage(self, websocket):
        try:
            q1 = random.choice([
                "Write a python code to Print Hello World",
                "Write a python code to Print Hello Anish",
                "Write a python code to Print Hello Duniya"
            ])
            await websocket.send_json({'type': 'coding_question', 'message': q1})
            await asyncio.wait_for(self.coding_event.wait(), timeout=300)  # 5-minute timeout
            self.coding_event.clear()
        except asyncio.TimeoutError:
            await websocket.send_json({'type': 'coding_timeout', 'message': 'Coding question timed out'})
        except Exception as e:
            await websocket.send_json({'type': 'coding_error', 'message': f'Error in coding stage: {str(e)}'})

    async def conduct_interview(self, websocket):
        for stage in self.stages:
            print(f"\n--- {stage} STAGE ---")
            try:
                self.message_history = ChatMessageHistory()
                if stage == "TECHNICAL":
                    prompt = PROMPTS[stage].format(
                        skills=self.cv_parts[stage],
                        job_description=self.job_description
                    )
                elif stage == "CODING":
                    for i in range(2):
                        await self.start_coding_stage(websocket=websocket)
                    await websocket.send_json({'type': 'coding_ended', 'message': "The End of Coding Round"})
                    continue
                else:
                    try:
                        cleaned_text = self.cv_parts[stage].replace("{", "").replace("}", "")
                        prompt = PROMPTS[stage].format(variable=cleaned_text)
                    except Exception as e:
                        print(e)

                while True:
                    if self.stop_interview.is_set():
                        return

                    response = self.get_ai_response(prompt, "Ask your question or exit the interview")
                    next_phase = response.find("next phase")
                    if next_phase != -1:
                        response = response[:next_phase]
                        move_to_new_phase = True
                    else:
                        move_to_new_phase = False

                    if response:
                        if "exit" in response or "interview concluded" in response:
                            break
                    if not response:
                        break

                    print("Interviewer:", response)
                    await websocket.send_json({'type': 'interview_question', 'question': response})

                    try:
                        await asyncio.wait_for(self.answer_event.wait(), timeout=300)  # 5-minute timeout
                        answer = self.current_answer
                        self.current_answer = None
                        self.answer_event.clear()
                    except asyncio.TimeoutError:
                        await websocket.send_json({'type': 'answer_timeout', 'message': 'Answer timed out'})
                        break

                    print("You:", answer)
                    if answer and "exit" in answer.lower():
                        return

                    self.message_history.add_user_message(answer)

                    if move_to_new_phase:
                        break
                    if stage != "CODING":
                        self.result[stage][response] = answer
                    await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error in {stage} stage: {str(e)}")
                await websocket.send_json({'type': 'stage_error', 'message': f'Error in {stage} stage: {str(e)}'})
                continue

        print("\nInterview concluded. Thank you for your time!")
        await websocket.send_json({'type': 'interview_end', 'message': 'Interview completed'})
