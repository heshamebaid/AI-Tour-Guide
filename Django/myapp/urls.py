from django.urls import path
from . import views

urlpatterns=[
    path('', views.login_view, name="login"),
    path('signup/',views.signup,name="signup"),
    path('home/',views.home,name="home"),
    path('translator/',views.upload_image,name="translator"),
    path('chatbot/',views.chatbot_view,name="chatbot"),
    path('chatbot/new/', views.chatbot_new, name="chatbot_new"),
    path('chatbot/stream/', views.chatbot_stream, name="chatbot_stream"),
    path('talk-to-pharos/', views.talk_to_pharos_view, name="talk_to_pharos"),
    path('place-details/', views.place_details_view, name="place_details"),

    # API endpoints for RAG chatbot
    path('api/chat/', views.chatbot_api, name="chatbot_api"),
    path('api/rag-status/', views.rag_status, name="rag_status"),
    path('api/place-details/', views.place_details_api, name="place_details_api"),
]