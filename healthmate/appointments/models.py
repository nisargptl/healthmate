from django.db import models
from patients.models import Patient

class Appointment(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    appointment_time = models.DateTimeField()
    doctor_name = models.CharField(max_length=100)
    status = models.CharField(max_length=50)  # e.g., "Scheduled", "Rescheduled", etc.

    def __str__(self):
        return f'{self.patient.first_name} {self.patient.last_name} - {self.appointment_time}'
