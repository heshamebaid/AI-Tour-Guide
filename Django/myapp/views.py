import os
import json

from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.shortcuts import redirect, render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .forms import ChatbotForm, ImageUploadForm
from .models import UploadedImage, Chatbot
from .rag_service import get_rag_service

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
    """
    Enhanced chatbot view using Agentic_RAG pipeline.
    Provides document search, web search, and image search capabilities.
    """
    response_data = None
    
    if request.method == 'POST':
        form = ChatbotForm(request.POST)
        if form.is_valid():
            user_input = form.cleaned_data['user_input']
            
            # Get feature toggle values from form
            use_agent = request.POST.get('use_agent', 'true').lower() == 'true'
            include_images = request.POST.get('include_images', 'true').lower() == 'true'
            
            # Use the RAG service for intelligent responses
            rag_service = get_rag_service()
            
            if rag_service.is_ready():
                try:
                    # Get response from RAG pipeline with selected features
                    result = rag_service.chat(
                        user_input=user_input,
                        use_agent=use_agent,
                        include_images=include_images
                    )
                    
                    if result.get("success", False):
                        bot_response = result.get("answer", "Sorry, I couldn't generate a response.")
                        sources = result.get("sources", [])
                        images = result.get("images", [])
                        web_info = result.get("web_info")
                        documents_found = result.get("documents_found", 0)
                        
                        response_data = {
                            "success": True,
                            "answer": bot_response,
                            "sources": sources,
                            "images": images,
                            "web_info": web_info,
                            "documents_found": documents_found
                        }
                    else:
                        # RAG returned an error
                        response_data = {
                            "success": False,
                            "error": result.get("error", "Unknown error occurred"),
                            "answer": result.get("answer", "")
                        }
                        bot_response = result.get("answer", result.get("error", "Error"))
                        sources = []
                        images = []
                        web_info = None
                        documents_found = 0
                        
                except Exception as e:
                    # Network or processing error
                    response_data = {
                        "success": False,
                        "error": f"Network or processing error: {str(e)}"
                    }
                    bot_response = f"Error: {str(e)}"
                    sources = []
                    images = []
                    web_info = None
                    documents_found = 0
            else:
                # RAG service not available - try fallback API
                try:
                    api_url = "http://localhost:8080/chat"
                    api_response = requests.post(api_url, json={"query": user_input}, timeout=30)
                    if api_response.ok:
                        bot_response = api_response.json().get("answer", "No answer")
                        response_data = {
                            "success": True,
                            "answer": bot_response,
                            "sources": [],
                            "images": [],
                            "documents_found": 0
                        }
                    else:
                        error_msg = f"API Error {api_response.status_code}: The LLM service is unavailable. Please check your OPENROUTER_API_KEY configuration."
                        response_data = {
                            "success": False,
                            "error": error_msg,
                            "answer": error_msg
                        }
                        bot_response = error_msg
                except requests.exceptions.ConnectionError:
                    error_msg = "Cannot connect to the AI service. Please check if the server is running."
                    response_data = {
                        "success": False,
                        "error": error_msg,
                        "answer": error_msg
                    }
                    bot_response = error_msg
                except requests.exceptions.Timeout:
                    response_data = {
                        "success": False,
                        "error": "The request timed out. Please try again."
                    }
                    bot_response = "Timeout Error"
                except Exception as e:
                    response_data = {
                        "success": False,
                        "error": f"Error: {str(e)}"
                    }
                    bot_response = f"Error: {e}"
                
                sources = []
                images = []
                web_info = None
                documents_found = 0
            
            # Save to database
            chat = form.save(commit=False)
            chat.bot_response = bot_response
            chat.set_sources(sources)
            chat.set_images(images)
            chat.web_info = web_info if web_info else ""
            chat.documents_found = documents_found
            chat.save()
    else:
        form = ChatbotForm()
    
    # Get recent chat history
    chat_history = Chatbot.objects.order_by('-timestamp')[:20]
    
    return render(request, 'chatbot.html', {
        'form': form,
        'response': response_data,
        'chat_history': chat_history
    })


@csrf_exempt
def chatbot_api(request):
    """
    API endpoint for AJAX chatbot requests.
    Returns JSON response with full RAG results.
    """
    if request.method != 'POST':
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        data = json.loads(request.body)
        user_input = data.get("query", "")
        use_agent = data.get("use_agent", True)
        include_images = data.get("include_images", True)
    except json.JSONDecodeError:
        user_input = request.POST.get("query", "")
        use_agent = True
        include_images = True
    
    if not user_input:
        return JsonResponse({"error": "No query provided"}, status=400)
    
    rag_service = get_rag_service()
    
    if not rag_service.is_ready():
        return JsonResponse({
            "error": "RAG service not available",
            "status": rag_service.get_status()
        }, status=503)
    
    result = rag_service.chat(
        user_input=user_input,
        use_agent=use_agent,
        include_images=include_images
    )
    
    # Save to database
    chat = Chatbot(
        user_input=user_input,
        bot_response=result.get("answer", ""),
        documents_found=result.get("documents_found", 0)
    )
    chat.set_sources(result.get("sources", []))
    chat.set_images(result.get("images", []))
    chat.web_info = result.get("web_info", "")
    chat.save()
    
    return JsonResponse(result)


def rag_status(request):
    """Check RAG service status."""
    rag_service = get_rag_service()
    return JsonResponse(rag_service.get_status())


def talk_to_pharos_view(request):
    context = {
        "pharos_service_url": PHAROS_SERVICE_URL,
    }
    return render(request, 'pharos/talk_to_pharos.html', context)