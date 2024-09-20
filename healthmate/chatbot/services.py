import os
import google.generativeai as genai
from patients.models import Patient

class ChatBotService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.chat = None
        self.history = []
        self.patient_context_initialized = False

    def get_patient_data(self):
        return Patient.objects.first()

    def start_chat(self):
        self.chat = genai.GenerativeModel("gemini-1.5-flash").start_chat(
            history=[
                {"role": "user", "parts": "Hello, I will share you some of my details. Remember them."},
            ]
        )

    def initialize_patient_context(self):
        patient = self.get_patient_data()
        patient_context = (
            f"My name is {patient.first_name} {patient.last_name}. "
            f"I was born on {patient.date_of_birth.strftime('%Y-%m-%d')}. "
            f"My medical condition is {patient.medical_condition}. "
            f"I am taking {patient.medication_regimen}. "
            f"My next appointment is on {patient.next_appointment.strftime('%Y-%m-%d %H:%M')} "
            f"with {patient.doctor_name}."
        )

        initial_message = f"Here are my details: {patient_context}"
        response = self.chat.send_message(initial_message)
        self.history.append({"role": "user", "parts": initial_message})
        self.history.append({"role": "model", "parts": response.text})
        self.patient_context_initialized = True

    def send_message(self, message):
        if self.chat is None:
            self.start_chat()

        if not self.patient_context_initialized:
            self.initialize_patient_context()

        response = self.chat.send_message(message)

        self.history.append({"role": "user", "parts": message})
        self.history.append({"role": "model", "parts": response.text})

        return response.text

    def update_patient_context(self):
        self.patient_context_initialized = False 
