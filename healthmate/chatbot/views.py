# chatbot/views.py
from django.shortcuts import render
from django.http import JsonResponse
from .core.healthmate_graph import compile_graph
import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, RemoveMessage

graph = compile_graph()
config = {"configurable": {"thread_id": str(uuid.uuid4())}}


first_user_message = True

def landing_page(request):
    global first_user_message
    if request.method == 'POST':
        additional_message = None
        user_message = request.POST.get('message')
        if first_user_message==True:
            input_message = {"messages": [HumanMessage(content=user_message)], "current_state":"Orchestrator", "message_counter":0}
            first_user_message=False
        else:
            input_message = {"messages": [HumanMessage(content=user_message)]}
        for output in graph.stream(input_message, config=config, stream_mode="updates"):
            print()
            print(output)
            if 'assistant' in output:
                bot_response = output['assistant'].get('messages', [])[0].content
            elif 'appt_rescheduler' in output:
                bot_response = output['appt_rescheduler'].get('messages', [])[0].content
            if 'change_state' in output:
                additional_message = output['change_state'].get('messages', [])[0].content
        return JsonResponse({"response": bot_response, "additional_info": additional_message})

    return render(request, 'landing_page.html')
