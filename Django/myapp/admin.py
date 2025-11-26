from django.contrib import admin

# Register your models here.

from .models import Chatbot,UploadedImage
# Register your models here.

admin.site.register(Chatbot)
admin.site.register(UploadedImage)