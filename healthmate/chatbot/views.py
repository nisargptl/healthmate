from django.shortcuts import render
from django.http import JsonResponse
from .core.healthmate_graph import compile_graph, save_graph
from .models import ConversationHistory 
import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import datetime

graph = compile_graph()
save_graph(graph)
def format_date_to_iso(date_str):
    month_replacements = {
        "Jan.": "Jan", "Feb.": "Feb", "Mar.": "Mar", "Apr.": "Apr", "May.": "May", "Jun.": "Jun",
        "Jul.": "Jul", "Aug.": "Aug", "Sept.": "Sep", "Oct.": "Oct", "Nov.": "Nov", "Dec.": "Dec"
    }
    for long_month, short_month in month_replacements.items():
        date_str = date_str.replace(long_month, short_month)
    try:
        date_obj = datetime.datetime.strptime(date_str, "%b %d, %Y")
    except ValueError:
        date_obj = datetime.datetime.today()
    return date_obj.strftime("%Y-%m-%d")


config = {"configurable": {"thread_id": str(uuid.uuid4())}}
first_user_message = True
def landing_page(request):
    global first_user_message
    summary = ""
    session_id = request.session.session_key or str(uuid.uuid4()) 

    if request.method == 'POST':
        conversation_history = request.session.get('conversation_history', {})
        if not isinstance(conversation_history, dict):
            conversation_history = {}  

        current_date = str(datetime.date.today())
        additional_message = None
        user_message = request.POST.get('message')
        bot_response = ""

        if first_user_message:
            input_message = {"messages": [HumanMessage(content=user_message)], "current_state": "Orchestrator", "message_counter": 0}
            first_user_message = False
        else:
            input_message = {"messages": [HumanMessage(content=user_message)]}

        for output in graph.stream(input_message, config=config, stream_mode="updates"):
            if "final_state" in output:
                if summary != output['final_state'].get('summary', []):
                    summary = output['final_state'].get('summary', [])
                    print(summary)
            elif 'assistant' in output:
                bot_response = output['assistant'].get('messages', [])[0].content
            elif 'appt_rescheduler' in output:
                bot_response = output['appt_rescheduler'].get('messages', [])[0].content
            elif 'treatment_change' in output:
                bot_response = output['treatment_change'].get('messages', [])[0].content
            elif 'change_state' in output:
                bot_response = output['change_state'].get('messages', [])[1].content
                additional_message = output['change_state'].get('messages', [])[0].content

        if bot_response != "":
            if current_date not in conversation_history:
                conversation_history[current_date] = [] 
            conversation_history[current_date].append({'sender': 'user', 'message': user_message})
            conversation_history[current_date].append({'sender': 'bot', 'message': bot_response})
            
            request.session['conversation_history'] = conversation_history

            ConversationHistory.objects.create(
                session_id=session_id,
                date=current_date,
                user_message=user_message,
                bot_response=bot_response,
            )

            return JsonResponse({"response": bot_response, "history_dates": list(conversation_history.keys()), "additional_info": additional_message})
        else:
            return JsonResponse({"response": ""})

    history_dates = ConversationHistory.objects.values_list('date', flat=True).distinct()
    history_dates = list(history_dates) 
    return render(request, 'landing_page.html', {'history_dates': history_dates})

def get_conversation_by_date(request):
    if request.method == 'POST':
        selected_date = request.POST.get('selected_date')
        session_id = request.session.session_key
        conversations = ConversationHistory.objects.filter(date=format_date_to_iso(selected_date)).values('user_message', 'bot_response')
        conversation = []
        for convo in conversations:
            conversation.append({'sender': 'user', 'message': convo['user_message']})
            conversation.append({'sender': 'bot', 'message': convo['bot_response']})
        return JsonResponse({"conversation": conversation})
    
def search_conversation_history(request):
    if request.method == 'POST':
        # Retrieve the search keyword from the POST request
        search_keyword = request.POST.get('search_keyword', "").lower()
        case_sensitive = request.POST.get('case_sensitive', 'false') == 'true'
        filter_user = request.POST.get('filter_user', 'false') == 'true'
        filter_bot = request.POST.get('filter_bot', 'false') == 'true'
        
        session_id = request.session.session_key
        matching_conversations = []

        # Retrieve conversation history from session
        conversation_history = request.session.get('conversation_history', {})

        # Search through each conversation for the keyword
        for date, conversations in conversation_history.items():
            for convo in conversations:
                message = convo['message']
                sender = convo['sender']

                # Apply case sensitivity if needed
                if not case_sensitive:
                    message = message.lower()
                    search_keyword = search_keyword.lower()

                # Apply sender filters
                if filter_user and sender != 'user':
                    continue
                if filter_bot and sender != 'bot':
                    continue

                # Check if the message contains the search keyword
                if search_keyword in message:
                    matching_conversations.append({'date': date, 'sender': sender, 'message': convo['message'], 'keyword': search_keyword})

        # Return the matching conversations as a JSON response
        return JsonResponse({"matching_conversations": matching_conversations})
