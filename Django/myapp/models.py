from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        if self.image:
            return self.image.name
        return f"Image uploaded at {self.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')}"
    
class Chatbot(models.Model):
    user_input = models.CharField(max_length=1000)
    bot_response = models.CharField(max_length=1000)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User Input: {self.user_input[:50]}"