
from .widgets import DatePickerInput
from django.forms import ModelForm
from .models import UploadedImage, Chatbot
from django import forms


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image']

class ChatbotForm(forms.ModelForm):
    class Meta:
        model = Chatbot
        fields = ['user_input']
        widgets = {
            'user_input': forms.Textarea(attrs={
                'rows': 2,
                'placeholder': 'Ask something about Ancient Egypt...',
                'class': 'chat-input',  
                'maxlength': 2000,
            })
        }


