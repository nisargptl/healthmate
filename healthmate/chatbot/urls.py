from django.urls import path
from .views import landing_page, get_conversation_by_date, search_conversation_history

urlpatterns = [
    path('', landing_page, name='landing_page'),
    path('get_conversation_by_date/', get_conversation_by_date, name='get_conversation_by_date'),
    path('search_conversation_history/', search_conversation_history, name='search_conversation_history'),  # New URL path for searching

]
