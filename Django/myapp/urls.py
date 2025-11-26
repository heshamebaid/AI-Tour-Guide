from django.urls import path
from . import views

urlpatterns=[
    path('', views.login_view, name="login"),
    path('signup/',views.signup,name="signup"),
    path('home/',views.home,name="home"),
    path('translator/',views.upload_image,name="translator"),
    path('chatbot/',views.chatbot_view,name="chatbot"),
    path('talk-to-pharos/', views.talk_to_pharos_view, name="talk_to_pharos"),
]