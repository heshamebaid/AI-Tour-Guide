import os

from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.shortcuts import redirect, render

from .forms import ChatbotForm, ImageUploadForm
from .models import UploadedImage

import requests

PHAROS_SERVICE_URL = os.environ.get("PHAROS_SERVICE_URL", "http://localhost:8050").rstrip("/")

def login_view(request):
    if request.method=='POST':
        username=request.POST.get('username-input')
        password=request.POST.get('password-input')
        user_to_check=authenticate(username=username,password=password)
        if user_to_check is not None:
            login(request,user_to_check)
            return redirect('home')
        else:
            error={'error':"You entered a wrong password"}
            return render(request, 'login.html',error)

    return render(request, 'login.html')


def signup(request):
    if request.method=='POST':
        username=request.POST.get('username-input')
        email=request.POST.get('email-input')
        password=request.POST.get('password-input')
        password_1=request.POST.get('password-input-1')
        x= User.objects.filter(username=username)
        if x:
            error={'error':"Username is already taken"}
            return render(request, 'signup.html', error)
        x= User.objects.filter(email=email)
        if x:
            error={'error':"Email is already taken"}
            return render(request, 'signup.html', error)
        if password!=password_1:
            error={'error':"Both passwords doesn't match"}
            return render(request, 'signup.html', error)
        user_to_create=User.objects.create_user(username,email,password)
        user_to_create.save()
        return redirect('login')
    
    return render(request, 'signup.html')
    
def home(request):
    return render(request, 'home.html')

def translator(request):
    return render(request, 'translator.html')

def chatbot(request):
    return render(request, 'chatbot.html')

def upload_image(request):
    story = None
    explanation = None
    symbols = []
    error = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            api_url = "http://localhost:8000/translate"
            try:
                with open(instance.image.path, "rb") as img_file:
                    files = {"file": (instance.image.name, img_file, "image/jpeg")}
                    api_response = requests.post(api_url, files=files)
                    if api_response.ok:
                        data = api_response.json()
                        story = data.get("story")
                        # If LLM provides explanation separately, extract it; else, explanation can be None
                        explanation = data.get("explanation")
                        # Each symbol: Gardiner Code, confidence, Hieroglyph, Description, symbol_image_base64
                        symbols = data.get("classifications", [])
                        error = data.get("error")
                    else:
                        error = f"Translation failed: {api_response.text}"
            except Exception as e:
                error = f"Error calling translation API: {e}"
            return render(request, 'translator.html', {
                'form': form,
                'story': story,
                'explanation': explanation,
                'symbols': symbols,
                'error': error
            })
    else:
        form = ImageUploadForm()
    return render(request, 'translator.html', {
        'form': form,
        'story': story,
        'explanation': explanation,
        'symbols': symbols,
        'error': error
    })

def chatbot_view(request):
    response = None
    if request.method == 'POST':
        form = ChatbotForm(request.POST)
        if form.is_valid():
            user_input = form.cleaned_data['user_input']
            api_url = "http://localhost:8080/chat"
            try:
                api_response = requests.post(api_url, json={"query": user_input})
                if api_response.ok:
                    bot_response = api_response.json().get("answer", "No answer")
                else:
                    bot_response = f"Chatbot failed: {api_response.text}"
            except Exception as e:
                bot_response = f"Error calling chatbot API: {e}"
            chat = form.save(commit=False)
            chat.bot_response = bot_response
            chat.save()
            response = bot_response
    else:
        form = ChatbotForm()
    return render(request, 'chatbot.html', {
        'form': form,
        'response': response
    })


def talk_to_pharos_view(request):
    context = {
        "pharos_service_url": PHAROS_SERVICE_URL,
    }
    return render(request, 'pharos/talk_to_pharos.html', context)