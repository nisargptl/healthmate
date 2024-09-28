# chatbot/views.py
from django.shortcuts import render
from django.http import JsonResponse
from .core.healthmate_graph import compile_graph, display_graph
import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage, RemoveMessage

graph = compile_graph()
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

def save_graph_as_image(graph, file_path='graph.png'):
    dot = graph.get_graph().draw_mermaid_png()
    with open(file_path, 'wb') as f:
        f.write(dot)
    print(f"Graph image saved at: {file_path}")
    return file_path

save_graph_as_image(graph)
first_user_message = True

def landing_page(request):
    global first_user_message
    summary =""
    if request.method == 'POST':
        additional_message = None
        user_message = request.POST.get('message')
        # print(user_message)
        if first_user_message==True:
            input_message = {"messages": [HumanMessage(content=user_message)], "current_state":"Orchestrator", "message_counter":0}
            first_user_message=False
        else:
            input_message = {"messages": [HumanMessage(content=user_message)]}
        for output in graph.stream(input_message, config=config, stream_mode="updates"):
            # print()
            # print(output)
            if "final_state" in output:
                if(summary != output['final_state'].get('summary', [])):
                    summary = output['final_state'].get('summary', [])
                    print(summary)
            if 'assistant' in output:
                bot_response = output['assistant'].get('messages', [])[0].content
            elif 'appt_rescheduler' in output:
                bot_response = output['appt_rescheduler'].get('messages', [])[0].content
            if 'change_state' in output:
                bot_response = output['change_state'].get('messages', [])[1].content
                additional_message = output['change_state'].get('messages', [])[0].content
        if bot_response:
            return JsonResponse({"response": bot_response, "additional_info": additional_message})
        else:
            return JsonResponse({"response": ""})

    return render(request, 'landing_page.html')
