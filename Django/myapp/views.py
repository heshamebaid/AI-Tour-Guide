import os
import json

from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.shortcuts import redirect, render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

from .forms import ChatbotForm, ImageUploadForm
from .models import UploadedImage, Chatbot
from .rag_service import get_rag_service

import requests

PHAROS_SERVICE_URL = os.environ.get("PHAROS_SERVICE_URL", "http://localhost:8050").rstrip("/")
# URL the browser can reach (e.g. localhost:8050); Docker internal hostnames like pharos-service:8050 don't work in the browser
PHAROS_PUBLIC_URL = os.environ.get("PHAROS_PUBLIC_URL", PHAROS_SERVICE_URL).rstrip("/")

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


def chatbot_new(request):
    """New ChatGPT-style streaming chat interface."""
    return render(request, 'chatbot_new.html')


@csrf_exempt
def chatbot_stream(request):
    """
    Streaming endpoint for chatbot - Server-Sent Events (SSE).
    Streams LLM response token by token with query rewriting optimization.
    """
    if request.method != 'POST':
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        data = json.loads(request.body)
        message = data.get("message", "")
        include_images = data.get("include_images", True)
        search_web = data.get("search_web", False)
    except json.JSONDecodeError:
        message = request.POST.get("message", "")
        include_images = True
        search_web = False
    
    if not message:
        return JsonResponse({"error": "No message provided"}, status=400)
    
    rag_service = get_rag_service()
    
    if not rag_service.is_ready():
        return JsonResponse({
            "error": "RAG service not available",
            "status": rag_service.get_status()
        }, status=503)
    
    def event_stream():
        """Generate SSE events from RAG streaming response."""
        full_response = ""
        sources = []
        images = []
        
        try:
            for chunk in rag_service.chat_streaming(message, include_images=include_images, search_web=search_web):
                chunk_type = chunk.get("type", "")
                content = chunk.get("content", "")
                
                if chunk_type == "token":
                    full_response += content
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                elif chunk_type == "sources":
                    sources = content
                    yield f"data: {json.dumps({'type': 'sources', 'content': content})}\n\n"
                elif chunk_type == "images":
                    images = content
                    yield f"data: {json.dumps({'type': 'images', 'content': content})}\n\n"
                elif chunk_type == "done":
                    yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
                elif chunk_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'content': content})}\n\n"
            
            # Save to database after streaming completes
            try:
                chat = Chatbot(
                    user_input=message,
                    bot_response=full_response,
                    documents_found=len(sources)
                )
                chat.set_sources(sources)
                chat.set_images(images)
                chat.save()
            except Exception as save_error:
                print(f"Error saving chat: {save_error}")
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    return response


def talk_to_pharos_view(request):
    context = {
        "pharos_service_url": PHAROS_PUBLIC_URL,
    }
    return render(request, 'pharos/talk_to_pharos.html', context)


# --- Place details (location → LLM-generated visitor info) ---
OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")
OPEN_ROUTER_MODEL = os.environ.get("OPEN_ROUTER_MODEL") or os.environ.get("LLM_MODEL") or "openai/gpt-4o-mini"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"


def _reverse_geocode(lat: float, lon: float) -> str:
    """Resolve lat/lon to a human-readable place name using Nominatim (no API key)."""
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "AI-Tour-Guide/1.0 (Educational)"},
            timeout=5,
        )
        if not r.ok:
            return f"Location ({lat:.4f}, {lon:.4f})"
        data = r.json()
        return data.get("display_name") or data.get("name") or f"Location ({lat:.4f}, {lon:.4f})"
    except Exception:
        return f"Location ({lat:.4f}, {lon:.4f})"


def _place_details_llm(place_name_or_address: str) -> str:
    """Call OpenRouter to generate visitor-friendly place details."""
    if not OPEN_ROUTER_API_KEY:
        return "Place details are unavailable: OPEN_ROUTER_API_KEY is not set."
    system = (
        "You are an AI tour guide for visitors. Given a place name or address, provide a concise, "
        "visitor-friendly overview: name, significance, what to see, practical tips, and any "
        "ancient Egyptian or historical connection if relevant. Use clear short paragraphs. "
        "If the place is unknown, say so politely and suggest checking the name or trying a nearby landmark."
    )
    payload = {
        "model": OPEN_ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Tell me about this place for a visitor:\n\n{place_name_or_address[:2000]}"},
        ],
    }
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        if not r.ok:
            return f"Sorry, the guide service returned an error (HTTP {r.status_code})."
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content") or "No description generated."
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"


def place_details_view(request):
    """Render the Place details page."""
    return render(request, "place_details.html")


@csrf_exempt
def place_details_api(request):
    """
    API: POST JSON { lat, lng } or { place_name }.
    Returns { success, place_name, details, error }.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}
    lat = data.get("lat")
    lng = data.get("lng")
    place_name = (data.get("place_name") or "").strip()
    if lat is not None and lng is not None:
        try:
            lat, lng = float(lat), float(lng)
        except (TypeError, ValueError):
            return JsonResponse({"success": False, "error": "Invalid lat/lng"}, status=400)
        place_name = _reverse_geocode(lat, lng)
    if not place_name:
        return JsonResponse({"success": False, "error": "Provide lat/lng or place_name"}, status=400)
    details = _place_details_llm(place_name)
    return JsonResponse({
        "success": True,
        "place_name": place_name,
        "details": details,
    })