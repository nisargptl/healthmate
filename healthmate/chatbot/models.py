# chatbot/models.py
from django.db import models

class ConversationHistory(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    date = models.DateField(auto_now_add=True)
    session_id = models.CharField(max_length=100)  # Track different sessions/conversations
    patient_email = models.EmailField()  # To associate the conversation with the patient

    def __str__(self):
        return f"{self.date} - {self.patient_email}"
