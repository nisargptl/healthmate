# chatbot/views.py
from django.shortcuts import render
from django.http import JsonResponse
from .services import ChatBotService

chatbot_service = ChatBotService()

def landing_page(request):
    if request.method == 'POST':
        user_message = request.POST.get('message')
        bot_response = chatbot_service.send_message(user_message)
        return JsonResponse({"response": bot_response})

    return render(request, 'landing_page.html')
